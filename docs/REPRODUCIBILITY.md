# Reproducibility Notes

This project aims for *explicit-state reproducibility*:
- The learned model is the `acts.jsonl` snapshot (plus `ledger.jsonl` history).
- No hidden weights / embeddings are trained.

## What is deterministic

Given:
- identical `acts.jsonl`
- identical `EngineConfig`
- identical prompt + plan_trace constraints

Then:
- concept execution (`concept_csv`) is deterministic
- validators are deterministic
- ledger chain verification is deterministic

## What can vary

In streaming training runs:
- the HF stream order depends on shuffle buffer and seed
- OS scheduling affects wall-clock time and `tokens_per_s`

## Freeze a run

Use:

```bash
python3 scripts/freeze_run.py --run <RUN_DIR>
```

This creates `<RUN_DIR>/freeze_manifest.json` with:
- latest snapshot + SHA256
- ledger SHA256 + chain verification
- python/platform info
- git revision + dirty flag

## Validate a frozen snapshot

Run a deterministic pack on the frozen snapshot:

```bash
python3 scripts/demo_sota_pack.py --acts <RUN_DIR>/snapshots/stepXXXXXX_acts.jsonl --pack sota_v8 --json
```

Compare:
- `sha256_acts`
- `sha256_transcript_text` (if you export transcripts in your own harness)
- metric keys (pass rates, concept depth rates, etc.)

