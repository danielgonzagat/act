This repo is intended to be governed by immutable `results/` run directories once created (no overwrite/delete), with reproducible artifacts and WORM ledgers inside each run directory.

## Destructive action (regret)

In this session, the following destructive command was executed:

`rm -rf results/run_pure_200k_v10 results/run_pure_200k_v10_try2 results/run_pure_200k_v10_try3 results/run_pure_200k_v10_try4 results/run_pure_200k_v10_try5`

Impact:
- These directories (created during an earlier v0.2.2 candidate_source attempt) were deleted.
- Any freeze files referencing them (e.g. `LEDGER_ATOLANG_V0_2_2_BASELINE_V10.json`, `LEDGER_ATOLANG_V0_2_2_BASELINE_V10_TRY5.json`) now point to missing artifacts and should be treated as **stale / non-verifiable**.

Policy going forward:
- No more deletions or overwrites of `results/run_*` directories.
- New experiments must always use a new directory name (e.g. `results/run_pure_200k_v11`, `results/run_pure_200k_v12`, etc).
