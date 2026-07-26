"""Input-perturbation consistency regularizer for entity-policy training.

This loss penalizes changes in the action distribution after Gaussian observation
noise:

    L_consistency = E_x E_delta ||softmax(f(x+delta)) - softmax(f(x))||^2

It is a local output-sensitivity heuristic.  It does *not* model emulator
dynamics, constrain the closed-loop state-transition Jacobian, prove contraction,
or implement Mehta et al.'s Stable-BC method.  Keep that distinction explicit
when interpreting the V5 negative experiment.

``sigma`` is expressed in normalized entity-feature units.  Both comparison
forwards run with dropout disabled, then the caller's train/eval mode is
restored.  Otherwise independent dropout masks would be measured as if they
were input sensitivity.
"""
from __future__ import annotations

import torch


def input_consistency_penalty(logits_clean: torch.Tensor, net, x: torch.Tensor,
                              sigma: float = 0.05) -> torch.Tensor:
    """Mean squared change in the action distribution under a small input perturbation.

    `logits_clean` supplies the output dtype/device for the zero-sigma fast path.
    For positive sigma the clean logits are recomputed with dropout disabled so
    the penalty isolates the declared input perturbation.
    Returns a non-negative scalar suitable for a small auxiliary loss weight.
    """
    if sigma <= 0:
        return logits_clean.new_zeros(())
    noise = sigma * torch.randn_like(x)
    was_training = net.training
    net.eval()
    try:
        stable_clean = net.logits(x)
        logits_pert = net.logits(x + noise)
    finally:
        net.train(was_training)
    p_clean = torch.softmax(stable_clean, dim=1)
    p_pert = torch.softmax(logits_pert, dim=1)
    return ((p_pert - p_clean) ** 2).sum(dim=1).mean()
