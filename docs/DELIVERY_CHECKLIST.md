# ACT Delivery Checklist (CPU-only, no GD)

This repo is designed to produce a **single** explicit model artifact (`acts.jsonl` + `ledger.jsonl`) that:
- Learns via **structural edits** (ADD/MERGE/PRUNE/REWRITE) only.
- Keeps all learned capability **explicit and auditable** (no hidden weights/embeddings).
- Runs **CPU-only**.

## 1) Freeze a run (repro manifest)

After a run finishes (or when you decide to freeze a snapshot):

```bash
python3 scripts/freeze_run.py --run <RUN_DIR>
cat <RUN_DIR>/freeze_manifest.json
```

This records:
- latest snapshot path + SHA256
- ledger hash + chain verification (WORM)
- python + platform
- git rev + dirty flag

## 2) Run a “convincing” deterministic demo (SOTA-ish pack)

Run the same validator pack used during training (example: `sota_v8`):

```bash
python3 scripts/demo_sota_pack.py --run <RUN_DIR> --pack sota_v8 --show_transcripts 3
```

Or pin a snapshot file:

```bash
python3 scripts/demo_sota_pack.py --acts <RUN_DIR>/snapshots/stepXXXXXX_acts.jsonl --pack sota_v8
```

Key output signals:
- `utility_pass_rate` and category pass rates
- `concept_*_rate` (deep/composed/very_deep)
- `concept_static_depth_max`
- `concept_cross_context_reuse_example` (created in one tag, used in others)
- CPU throughput estimate (`tokens_per_s_est`)

## 3) Inspect the internal language (concept_csv)

Pick a concept id from demo output (e.g. `cross_context_example.concept_id`), then:

```bash
python3 scripts/inspect_concept_csv.py --run <RUN_DIR> --concept_id <CONCEPT_ID>
```

This prints:
- interface schema + validator id
- static depth + call deps
- the concept program ops (CSV_GET_INPUT / CSV_CALL / CSV_PRIMITIVE / CSV_RETURN)

## 4) Training (streaming FLAN)

Canonical pattern:

```bash
nohup env PYTHONUNBUFFERED=1 python3 scripts/train_hf_ka.py \
  --source Open-Orca/FLAN --split train --on_the_fly \
  --streaming --shuffle --shuffle_buffer 10000 \
  --steps 100000 --mode pure --window 2000 \
  --selection_mode survival \
  --holdout_frac 0.05 --holdout_eval_windows 3 --holdout_eval_tokens 4000 \
  --utility_weight 1.0 --skill_suite_pack sota_v8 \
  --agency_suite_enabled \
  --concept_csv_mining_enabled --concept_csv_mining_max_ops 96 \
  --concept_csv_composed_enabled --concept_csv_budget 128 \
  --out <RUN_DIR> \
  > <RUN_DIR>/train.log 2>&1 & echo $! > <RUN_DIR>/nohup_pid
```

Monitor:

```bash
tail -f <RUN_DIR>/train.log
tail -f <RUN_DIR>/report.jsonl
ps -p $(cat <RUN_DIR>/nohup_pid) -o pid,etime,%cpu,%mem,command
```

