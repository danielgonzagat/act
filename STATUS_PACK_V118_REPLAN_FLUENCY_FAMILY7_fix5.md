# STATUS PACK V118 — REPLAN + FLUENCY (v118) + FAMILY7 (fix5)

## Resumo
- Adiciona `replan_until_satisfies_v118` (replanning determinístico com trace WORM) e `fluency_contract_v118` (wrapper do v112 com checks “adult-dialogue” determinísticos).
- Adiciona pipeline Family7 v118 (gerador + runner + smoke) com determinismo try1==try2 e disciplina de gating do ExternalWorld (negativos + 1 call in-cycle).

## Checklist (PASS/FAIL)
- PASS `PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m py_compile atos_core/*.py scripts/*.py tests/*.py`
- PASS `PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m unittest -v` (29 tests)
- PASS smoke `scripts/smoke_v118_replan_law.py` determinístico (try1==try2)
- PASS smoke `scripts/smoke_v118_family7_long_dialogue_world.py` determinístico (try1==try2), `tasks_ok=7/20` com `min_ok=5`
- PASS negativos ExternalWorld (not_allowed / invalid_reason_code) no smoke Family7 v118

## Arquivos de código (novos)
- `atos_core/conversation_loop_v118.py:1` (wrapper v117 + `final_response_v118.json`)
- `atos_core/external_dialogue_world_v118.py:1` (wrapper read-only v113)
- `atos_core/fluency_contract_v118.py:1` (wrapper `fluency_contract_v112` + checks determinísticos)
- `atos_core/replan_law_v118.py:1` (replan loop + `replan_trace_v118.json`)
- `scripts/gen_family7_dla_from_external_world_v118.py:1` (gerador determinístico tasks v118)
- `scripts/run_family7_dla_v118.py:1` (runner determinístico + fluency gate v118 + external-world probe 1x)
- `scripts/smoke_v118_family7_long_dialogue_world.py:1` (smoke try1/try2 + negativos gating)
- `scripts/smoke_v118_replan_law.py:1` (smoke replan v118)
- `tests/test_replan_law_v118.py:1` (unit tests do replan v118)

## Comandos executados (copy/paste)
```bash
cd /Users/danielpenin/act
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m py_compile atos_core/*.py scripts/*.py tests/*.py
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m unittest -v

# Smoke: replanning law
set -o pipefail
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v118_replan_law.py \
  --out_base results/run_smoke_v118_replan_law_fix1 --seed 0 \
| tee results/smoke_v118_replan_law_fix1_try1.json

# Generate tasks (WORM)
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/gen_family7_dla_from_external_world_v118.py \
  --world_manifest external_world/dialogue_history_canonical_v113_manifest.json \
  --seed 0 --out tasks/family7_dla_v118_seed0_fix5.jsonl \
  --tasks_total 20 --stress_200 2

# Smoke: Family7 long dialogue (WORM; try1/try2 internal)
set -o pipefail
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v118_family7_long_dialogue_world.py \
  --tasks tasks/family7_dla_v118_seed0_fix5.jsonl \
  --out_base results/run_smoke_v118_family7_long_dialogue_world_fix5 --seed 0 \
| tee results/smoke_v118_family7_long_dialogue_world_fix5_try1.json
```

## Determinismo (evidência)
- Replan smoke v118: `results/run_smoke_v118_replan_law_fix1_try1/eval.json` sha256 == `results/run_smoke_v118_replan_law_fix1_try2/eval.json` sha256 == `53c34ab0a10e901c1bdd393283392c7bab533a0a78e1cf156d3ac55cec1f9507`
- Replan smoke v118: `summary_sha256` == `7218453bcaf5cb80c8747c69541e3e266ff65d46fb71da5ea0390b6713e62e6d`
- Family7 smoke v118 (fix5): `eval_sha256` try1==try2 == `1a27bae554b3d183ea474849ff6c6718be4a09b9249461eedb6db29024d7d575`
- Family7 smoke v118 (fix5): `summary_sha256` == `fee7e9ec697992b548c4ab895080e2e54826fd23dee19c1277d88cb440aa6138`
- ExternalWorld gating negativos (fix5): `external_world_access_not_allowed` e `invalid_reason_code`

## Artefatos WORM + SHA256
- Patch diff: `patches/v118_replan_fluency_family7_fix5.diff` = `3374032b81eaae55d59104530f5d818e2cdca5aaf7baef444d15e3c8fd1d2399`
- Smoke replan stdout: `results/smoke_v118_replan_law_fix1_try1.json` = `0de33502c638162884f34b1f4bca66b850bb8f8ab204829a5cfc6e4aa9b9ca63`
- Smoke replan try1 eval: `results/run_smoke_v118_replan_law_fix1_try1/eval.json` = `53c34ab0a10e901c1bdd393283392c7bab533a0a78e1cf156d3ac55cec1f9507`
- Smoke replan try1 fail_catalog: `results/run_smoke_v118_replan_law_fix1_try1/fail_catalog_v118.json` = `05e8b90808ab16d5e463a4ca6e4ef5d70a6df98e47da8fa5954ff2133bcfe8a2`
- Smoke replan try1 summary: `results/run_smoke_v118_replan_law_fix1_try1/summary.json` = `d77dfe39758c302a7048dd7ea338d00107a10cf61a5c2b91d7a9920f0031074f`
- Smoke family7 stdout: `results/smoke_v118_family7_long_dialogue_world_fix5_try1.json` = `43c86c7caf584d1449d40ae5a146c3a4bdeb7cc4905c8470c2a8ef40a79c915c`
- Smoke family7 try1 eval: `results/run_smoke_v118_family7_long_dialogue_world_fix5_try1/eval.json` = `1a27bae554b3d183ea474849ff6c6718be4a09b9249461eedb6db29024d7d575`
- Smoke family7 try1 fail_catalog: `results/run_smoke_v118_family7_long_dialogue_world_fix5_try1/fail_catalog_v118.json` = `bc7dca1d974c3b3cacdcd6a604a1bb434291b9f91456cd83492b51c865456301`
- Smoke family7 try1 summary: `results/run_smoke_v118_family7_long_dialogue_world_fix5_try1/summary.json` = `9917a5743cf392511e8bc73dd3cf47115ae1b7d74a8938c121352190786cc635`
- Tasks jsonl: `tasks/family7_dla_v118_seed0_fix5.jsonl` = `26ef9c3b603b7c87fe9fb06ed9a745420dc86bf1a019d034f616b5467cc11c2e`
- Tasks manifest: `tasks/family7_dla_v118_seed0_fix5.jsonl.manifest.json` = `f58e84e7111a9bc7ff2201f8646768a81637cb321d9f69915fba12c157285622`

