# STATUS PACK V119 — FLUENCY-AS-SURVIVAL LAW + DISCOURSE VARIANTS (fix1)

## Resumo
- Torna `fluency_contract_v118` uma lei dentro do runtime (não só no runner): `atos_core/conversation_loop_v119.py` só aceita `final_response_v116.ok==true` quando `fluency_ok==true`; caso contrário registra FAIL_EVENT_V119 e tenta novo attempt determinístico.
- Elimina o colapso dominante `consecutive_prefix2_repeat` com variação determinística (sem RNG) via `atos_core/discourse_variants_v119.py`, incluindo preâmbulo variável para o prompt de opções A/B/C.
- Fecha o bug de resolução de manifest do mundo externo (v111 vs v113) no gerador Family7 v118, com resolução determinística e auditável.
- Adiciona um hard gate explícito `scripts/smoke_v119_family7_hard_gate.py` que exige `tasks_ok == tasks_total` (sem “min_ok”).

## Checklist (PASS/FAIL)
- PASS `PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m py_compile atos_core/*.py scripts/*.py`
- PASS `PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m unittest -v` (35 tests)
- PASS smoke `scripts/smoke_v119_ack_spam_200.py` determinístico (try1==try2), `fluency_ok=true`
- PASS hard gate `scripts/smoke_v119_family7_hard_gate.py` determinístico (try1==try2), `tasks_ok=5/5`
- PASS gerador v118 com manifests v113 e v111 (resolução determinística do canonical jsonl)

## Arquivos de código (novos)
- `atos_core/conversation_loop_v119.py:1` (fluency-as-survival no runtime + FAIL_EVENT_V119 por tentativa rejeitada)
- `atos_core/discourse_variants_v119.py:1` (bank determinístico + anti-colisão por prefix2; inclui preâmbulo para opções)
- `atos_core/world_manifest_resolver_v119.py:1` (resolver determinístico do canonical jsonl a partir de manifest v111/v113)
- `scripts/smoke_v119_ack_spam_200.py:1` (prova determinística: 200 turns de “ok” sem colapsar fluency)
- `scripts/smoke_v119_family7_hard_gate.py:1` (hard gate: 100% pass em subset ack-like do Family7)
- `tests/test_discourse_variants_v119.py:1` (determinismo + anti-colisão + sequência ack-spam)
- `tests/test_world_manifest_resolver_v119.py:1` (resolução v111/v113 em tmpdir)

## Arquivos alterados (retrocompatíveis)
- `atos_core/conversation_loop_v110.py:1` (opt-in `discourse_variants_v119_enabled=False` default; aplica variantes em caminhos de clarificação/reformulação/opções)
- `scripts/gen_family7_dla_from_external_world_v118.py:1` (usa resolver v119; injeta campos auditáveis `world_*_sha256` nas tasks)

## Comandos executados (copy/paste)
```bash
cd /Users/danielpenin/act
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m py_compile atos_core/*.py scripts/*.py
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m unittest -v

# Smoke: ack-spam 200 (WORM; try1/try2 internal)
set -o pipefail
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v119_ack_spam_200.py \
  --out_base results/run_smoke_v119_ack_spam_200_fix3 --seed 0 \
| tee results/smoke_v119_ack_spam_200_fix3_try1.json

# Smoke: Family7 hard gate (WORM; try1/try2 internal)
set -o pipefail
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v119_family7_hard_gate.py \
  --tasks tasks/family7_dla_v113_seed0_fix2.jsonl \
  --out_base results/run_smoke_v119_family7_hard_gate_fix2 --seed 0 --top_n 5 \
| tee results/smoke_v119_family7_hard_gate_fix2_try1.json

# Evidence: generator v118 works for both manifests (v113 + v111)
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/gen_family7_dla_from_external_world_v118.py \
  --world_manifest external_world/dialogue_history_canonical_v113_manifest.json \
  --seed 0 --out results/tmp_v119_tasks_v118_from_v113_manifest_seed0.jsonl \
  --tasks_total 20 --stress_200 2 --stress_500 0

PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/gen_family7_dla_from_external_world_v118.py \
  --world_manifest external_world/manifests/world_manifest_v111.json \
  --seed 0 --out results/tmp_v119_tasks_v118_from_v111_manifest_seed0.jsonl \
  --tasks_total 20 --stress_200 2 --stress_500 0

# Patch diff (deterministic file order)
(
  diff -u patches/v119_base/atos_core/conversation_loop_v110.py atos_core/conversation_loop_v110.py || true; \
  diff -u /dev/null atos_core/conversation_loop_v119.py || true; \
  diff -u /dev/null atos_core/discourse_variants_v119.py || true; \
  diff -u /dev/null atos_core/world_manifest_resolver_v119.py || true; \
  diff -u patches/v119_base/scripts/gen_family7_dla_from_external_world_v118.py scripts/gen_family7_dla_from_external_world_v118.py || true; \
  diff -u /dev/null scripts/smoke_v119_ack_spam_200.py || true; \
  diff -u /dev/null scripts/smoke_v119_family7_hard_gate.py || true; \
  diff -u /dev/null tests/test_discourse_variants_v119.py || true; \
  diff -u /dev/null tests/test_world_manifest_resolver_v119.py || true \
) > patches/v119_fluency_survival_disourse_variants_fix1.diff
```

## Determinismo (evidência)
- Ack-spam 200 (fix3): try1==try2
  - `results/run_smoke_v119_ack_spam_200_fix3_try1/eval.json` sha256 == `82dcf776ab08b7592f94fa1bb5cc31fcfc8b8c6bcccb900f16306db46bb4b10b`
  - `summary_sha256` == `48a232d3abc89b05257fbe032ec11bd1843b125503ab06ddfed124e96777f08f`
  - `assistant_prefix2_max_consecutive.n` == `1` (no `smoke_summary.json`)
- Family7 hard gate (fix2): try1==try2
  - `tasks_ok == tasks_total == 5`
  - `results/run_smoke_v119_family7_hard_gate_fix2_try1/eval.json` sha256 == `b76ecd9c43e2ba930b2cc25dbb64cc0256383a94bd0653299034daddb06d2f71`
  - `summary_sha256` == `22bdde41b91b70f84dd2877ea8ff29351aee265af87fa84cabbf0cc78ec481f8`

## Artefatos WORM + SHA256
- Patch diff: `patches/v119_fluency_survival_disourse_variants_fix1.diff` = `b68e30500eecc37a543083ffef90f624974d28725c17bb9b6a796d94d2ac6996`
- Smoke ack-spam stdout: `results/smoke_v119_ack_spam_200_fix3_try1.json` = `5a6f2bedf33c2f533d6ae576bb2fb6445d9125d3ca9f98d5e7de3f64fd079886`
- Smoke ack-spam try1 eval: `results/run_smoke_v119_ack_spam_200_fix3_try1/eval.json` = `82dcf776ab08b7592f94fa1bb5cc31fcfc8b8c6bcccb900f16306db46bb4b10b`
- Smoke ack-spam try1 smoke_summary: `results/run_smoke_v119_ack_spam_200_fix3_try1/smoke_summary.json` = `be91cc7776079e5bbb4644c1dd93a2648e8671752e83740780afd852fcb3d0e1`
- Smoke ack-spam try1 summary: `results/run_smoke_v119_ack_spam_200_fix3_try1/summary.json` = `7c20fd1f0a0e11fb7aa0be56081ab4b5dd75d4b4f4fb3b12e7bd876f55778dd9`
- Smoke Family7 hard gate stdout: `results/smoke_v119_family7_hard_gate_fix2_try1.json` = `9a50b472cfeb20ee93ed25bebd244cc8e6cac987ed89b7286f4d8bfaff50652a`
- Smoke Family7 hard gate try1 eval: `results/run_smoke_v119_family7_hard_gate_fix2_try1/eval.json` = `b76ecd9c43e2ba930b2cc25dbb64cc0256383a94bd0653299034daddb06d2f71`
- Smoke Family7 hard gate try1 summary: `results/run_smoke_v119_family7_hard_gate_fix2_try1/summary.json` = `9e5a89ad25659c2f10c217fecfc05df5d8476bc28ea9edd00775bd043d2a0ab9`
- Tasks Family7 input: `tasks/family7_dla_v113_seed0_fix2.jsonl` = `72a91a0d4bb15b31ac4812e6b25d9c84ac9854f91b91ec493bf60face49773b1`
- Tasks Family7 manifest: `tasks/family7_dla_v113_seed0_fix2.jsonl_manifest.json` = `e3c28b45c150c648ecc2c17aca1525737b30ce198eb97f62d7f2799cf2b972a0`
- Generator v118 (manifest v113) output: `results/tmp_v119_tasks_v118_from_v113_manifest_seed0.jsonl` = `0c323e27f45a2fcd443bd324627e4761257a4914f3dc3baebe3cdbf8a8549636`
- Generator v118 (manifest v111) output: `results/tmp_v119_tasks_v118_from_v111_manifest_seed0.jsonl` = `e346b9083788657c5c45295c32d269c25262ba0221a5e544ff20d9ba69aa19c8`

## GO / NO-GO
- GO: fluência virou lei (sem relaxar thresholds) e o colapso dominante por `consecutive_prefix2_repeat` foi removido por variação determinística auditável.
