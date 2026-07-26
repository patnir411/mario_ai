# Canonical solution manifests

This directory is the small, reviewable evidence surface for replayable action
sequences. Datasets, checkpoints, videos, savestates, search prefixes, and run
directories remain ignored.

For SMB1, `W-S.json` is a positive solution only when:

1. `solved` is true;
2. `path` is an integer action sequence with a positive `chunk_frames`;
3. deterministic replay from seed 0 reaches the flag.

Run the complete acceptance gate with:

```bash
./venv/bin/python scripts/verify_stock_solutions.py --expect-verified 30
```

As of 2026-07-25, 30 of 32 stock levels pass. `6-2.json` is an unsolved
partial attempt and `6-3.json` is quarantined negative evidence after its
search-time success claim failed replay. Neither counts as solved.

SMA4 and SML manifests use their adapter-specific replay gates and legally
obtained local ROMs. Fresh-boot roots can be replayed after supplying the
matching ROM. Snapshot-root manifests record ROM and snapshot hashes, but the
savestates themselves remain under ignored `runs/`; they are therefore locally
replay-auditable/declarative evidence, not clean-clone-verifiable artifacts.
