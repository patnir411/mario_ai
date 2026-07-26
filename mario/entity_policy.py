"""Scaled entity-transformer generalist policy (attention over {player, enemies, terrain}).

Promotes the V4 152K experiment to a real, importable policy with a SHARED-TRUNK value
head (procedure-cloning aux target) and a soft-action head. ~1-3M params — trains in
minutes/epoch on the M2 MPS, <2GB. Used by scripts/train_generalist.py (BC + DAgger) and
scripts/eval_heldout.py.

Reads the shared schema from mario.entity (N_TOK tokens × D_TOK features).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from mario.actions import SMB1_N_ACTIONS as N_ACTIONS
from mario.entity import D_TOK, N_TOK

# Inference bias: discourage NOOP/left/down/up under uncertainty (proven in V4/production).
ACTION_BIAS = [-0.6, 0, 0, 0, 0, 0, -1.0, -0.8, -0.8]


class EntityTransformer(nn.Module):
    """Self-attention over entity tokens → (action logits, value)."""

    def __init__(self, d_model: int = 128, nhead: int = 4, layers: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.cfg = {"d_model": d_model, "nhead": nhead, "layers": layers, "dropout": dropout}
        self.proj = nn.Linear(D_TOK, d_model)
        self.tok_emb = nn.Parameter(0.02 * torch.randn(N_TOK, d_model))  # learned type/pos
        enc = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=4 * d_model,
                                         dropout=dropout, batch_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(enc, layers)
        self.norm = nn.LayerNorm(d_model)
        self.policy_head = nn.Linear(d_model, N_ACTIONS)
        self.value_head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor):
        if x.ndim != 2 or x.shape[1] != N_TOK * D_TOK:
            raise ValueError(
                f"EntityTransformer expects (batch,{N_TOK * D_TOK}), "
                f"got {tuple(x.shape)}")
        b = x.shape[0]
        t = x.view(b, N_TOK, D_TOK)
        h = self.proj(t) + self.tok_emb[None]
        h = self.enc(h)
        h = self.norm(h.mean(1))          # mean-pool tokens
        return self.policy_head(h), self.value_head(h).squeeze(-1)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)[0]


class TemporalEntityTransformer(nn.Module):
    """Entity-transformer with TEMPORAL CONTEXT — attends over the last `ctx_k` frames of
    entity tokens (K·N_TOK tokens with a learned time embedding), then decides from the CURRENT
    (last) frame's pooled tokens.  A short observation history is a testable way
    to expose motion phase that one frame may omit; it is not assumed to make the
    representation Markov.  Past actions are intentionally omitted in this
    implementation. ``ctx_k=1`` provides the direct architecture ablation."""

    def __init__(self, ctx_k: int = 4, d_model: int = 128, nhead: int = 4, layers: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.cfg = {"ctx_k": ctx_k, "d_model": d_model, "nhead": nhead,
                    "layers": layers, "dropout": dropout}
        self.ctx_k = ctx_k
        self.proj = nn.Linear(D_TOK, d_model)
        self.tok_emb = nn.Parameter(0.02 * torch.randn(N_TOK, d_model))    # per-token type/pos
        self.time_emb = nn.Parameter(0.02 * torch.randn(ctx_k, d_model))   # per-frame (recency)
        enc = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=4 * d_model,
                                         dropout=dropout, batch_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(enc, layers)
        self.norm = nn.LayerNorm(d_model)
        self.policy_head = nn.Linear(d_model, N_ACTIONS)
        self.value_head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor):
        # x: (B, ctx_k, N_TOK*D_TOK) — a stack of the last ctx_k entity-obs frames
        expected = (self.ctx_k, N_TOK * D_TOK)
        if x.ndim != 3 or tuple(x.shape[1:]) != expected:
            raise ValueError(
                f"TemporalEntityTransformer expects "
                f"(batch,{expected[0]},{expected[1]}), got {tuple(x.shape)}")
        b = x.shape[0]
        t = x.view(b, self.ctx_k, N_TOK, D_TOK)
        h = self.proj(t) + self.tok_emb[None, None] + self.time_emb[None, :, None]
        h = self.enc(h.view(b, self.ctx_k * N_TOK, -1))
        cur = h.view(b, self.ctx_k, N_TOK, -1)[:, -1].mean(1)   # decide from the CURRENT frame
        cur = self.norm(cur)
        return self.policy_head(cur), self.value_head(cur).squeeze(-1)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)[0]


def n_params(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())


def mps_parity_ok(net: "EntityTransformer", x32) -> dict:
    """CPU-vs-MPS argmax agreement on a small batch (deployment-safety check)."""
    import numpy as np
    x = torch.from_numpy(np.asarray(x32, np.float32))
    net_cpu = net.to("cpu").eval()
    with torch.no_grad():
        lc = net_cpu.logits(x)
    out = {"mps_available": False, "argmax_agreement": 1.0, "max_abs_logit_diff": 0.0}
    if torch.backends.mps.is_available():
        net_mps = net.to("mps").eval()
        with torch.no_grad():
            lm = net_mps.logits(x.to("mps")).cpu()
        out = {"mps_available": True,
               "argmax_agreement": float((lc.argmax(1) == lm.argmax(1)).float().mean()),
               "max_abs_logit_diff": float((lc - lm).abs().max())}
        net.to("cpu")
    return out


def save_policy(net: nn.Module, path, extra: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {"state": net.state_dict(), "cfg": net.cfg}
    if extra:
        blob.update(extra)
    torch.save(blob, path)


def load_policy_checkpoint(path, device: str = "cpu"):
    """Load a policy plus the non-tensor provenance stored beside its weights."""
    blob = torch.load(path, map_location=device, weights_only=True)
    cfg = blob.get("cfg", {})
    net = TemporalEntityTransformer(**cfg) if "ctx_k" in cfg else EntityTransformer(**cfg)
    net.load_state_dict(blob["state"])
    metadata = {key: value for key, value in blob.items() if key != "state"}
    return net.to(device).eval(), metadata


def load_policy(path, device: str = "cpu"):
    return load_policy_checkpoint(path, device=device)[0]


class EntityPolicyPrior:
    """Wraps a trained EntityTransformer as a PRIOR for tree search ("policy proposes,
    search disposes"). Returns log π(a|s) over actions for a live RAM state, so beam_search
    can (a) add a prior bonus to each child's score and (b) expand only the top-k actions per
    node — turning a wide search narrow without losing the solvable lineage."""

    def __init__(self, net: EntityTransformer, device: str = "cpu", temperature: float = 1.0):
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.net = net.to(device).eval()
        self.device = device
        self.temp = temperature

    @torch.no_grad()
    def logp(self, ram, info=None):
        from mario.entity import entity_obs
        x = torch.tensor(entity_obs(ram, info), device=self.device)[None]
        lg = self.net.logits(x).squeeze(0) / self.temp
        return torch.log_softmax(lg, dim=0).cpu().numpy()


class TemporalEntityPolicyPrior:
    """Search prior for a TemporalEntityTransformer: returns log π(a|s) given a K-frame stack of
    entity obs. beam_search owns the per-node history bookkeeping and calls logp_stack. `ctx_k`
    tells the search how many recent frames to maintain per node."""

    def __init__(self, net: TemporalEntityTransformer, device: str = "cpu", temperature: float = 1.0):
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.net = net.to(device).eval()
        self.device = device
        self.temp = temperature
        self.ctx_k = int(net.cfg.get("ctx_k", 1))

    @torch.no_grad()
    def logp_stack(self, stack) -> np.ndarray:
        x = torch.tensor(np.asarray(stack, np.float32), device=self.device)[None]  # (1,K,238)
        lg = self.net.logits(x).squeeze(0) / self.temp
        return torch.log_softmax(lg, dim=0).cpu().numpy()


class EntityController:
    """Closed-loop wrapper: RAM → action index (argmax of biased logits)."""

    def __init__(self, net: EntityTransformer, device: str = "cpu"):
        self.net = net.to(device).eval()
        self.device = device
        self.bias = torch.tensor(ACTION_BIAS, device=device)

    def act(self, ram, info=None) -> int:
        from mario.entity import entity_obs
        x = torch.tensor(entity_obs(ram, info), device=self.device)[None]
        with torch.no_grad():
            logits = self.net.logits(x) + self.bias
        return int(logits.argmax(1))

    def act_with_entropy(self, ram, info=None):
        """Return (action, entropy_nats, value) — entropy flags uncertain states for DAgger."""
        from mario.entity import entity_obs
        x = torch.tensor(entity_obs(ram, info), device=self.device)[None]
        with torch.no_grad():
            logits, value = self.net(x)
            p = torch.softmax(logits, dim=1)
            ent = float(-(p * torch.log(p.clamp_min(1e-9))).sum(1))
            a = int((logits + self.bias).argmax(1))
        return a, ent, float(value)


class TemporalEntityController:
    """Closed-loop wrapper for TemporalEntityTransformer: maintains a rolling K-frame history of
    entity obs and feeds the stack each step. Call reset() at the start of every episode."""

    def __init__(self, net: TemporalEntityTransformer, device: str = "cpu"):
        self.net = net.to(device).eval()
        self.device = device
        self.k = int(net.cfg.get("ctx_k", 1))
        self.bias = torch.tensor(ACTION_BIAS, device=device)
        self.hist: list = []

    def reset(self) -> None:
        self.hist = []

    def _stack(self, ram, info):
        from mario.entity import entity_obs
        self.hist.append(entity_obs(ram, info))
        h = self.hist[-self.k:]
        while len(h) < self.k:                 # left-pad the start of an episode
            h = [h[0]] + h
        return torch.tensor(np.stack(h), device=self.device)[None]   # (1, K, OBS_DIM_ENTITY)

    def act(self, ram, info=None, biased: bool = True) -> int:
        with torch.no_grad():
            lg = self.net.logits(self._stack(ram, info)).squeeze(0)
        if biased:
            lg = lg + self.bias
        return int(lg.argmax())

    def act_with_entropy(self, ram, info=None):
        with torch.no_grad():
            lg, v = self.net(self._stack(ram, info))
            lg = lg.squeeze(0)
            p = torch.softmax(lg, 0)
            ent = float(-(p * torch.log(p.clamp_min(1e-9))).sum())
            a = int((lg + self.bias).argmax())
        return a, ent, float(v)
