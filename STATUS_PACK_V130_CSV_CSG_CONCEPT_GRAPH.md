# STATUS PACK V130 — CSV/CSG CONCEPT GRAPH

## Resumo
- Adiciona wrappers V130 para CSV/CSG (`atos_core/csg_v130.py`) mantendo semântica/execução do V87 e isolando o baseline por `concept_id_prefix="concept_csv_v130_"`.
- Smoke determinístico V130 (`scripts/smoke_csv_csg_v130.py`) prova: (1) execução via conceito com CSG embed == replay inline por expansão do CSG; (2) bloqueio explícito por match (`match_disallowed`); (3) logs WORM hash-chained (`concepts.jsonl`, `concept_evidence.jsonl`, `concept_telemetry.jsonl`).
- Verificador de cadeia (`scripts/verify_csv_csg_v130_chain.py`) valida hash-chain dos 3 JSONL.

## Checklist (PASS/FAIL)
- PASS `PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m py_compile atos_core/*.py scripts/*.py tests/*.py`
- PASS `PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m unittest -v`
- PASS smoke `scripts/smoke_csv_csg_v130.py` determinístico (try1==try2)
- PASS replay/expand: output via conceito == output via expansão inline
- PASS negative: match disallowed (`match_disallowed`)
- PASS verify chains (`scripts/verify_csv_csg_v130_chain.py`)

## Arquivos de código (novos)
- `atos_core/csg_v130.py:1`
- `scripts/smoke_csv_csg_v130.py:1`
- `scripts/verify_csv_csg_v130_chain.py:1`
- `tests/test_csg_v130.py:1`

## Comandos executados (copy/paste)
```bash
cd /Users/danielpenin/act
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m py_compile atos_core/*.py scripts/*.py tests/*.py
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m unittest -v

set -o pipefail
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_csv_csg_v130.py \
  --out_base results/run_csv_csg_v130_seed0 \
  --seed 0 \
| tee results/smoke_csv_csg_v130_try1.json

PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/verify_csv_csg_v130_chain.py \
  --run_dir results/run_csv_csg_v130_seed0_try1 \
| tee results/verify_csv_csg_v130_chain_try1.json

PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/verify_csv_csg_v130_chain.py \
  --run_dir results/run_csv_csg_v130_seed0_try2 \
| tee results/verify_csv_csg_v130_chain_try2.json

# Patch diff (ordem determinística; somente arquivos V130)
(
  diff -u /dev/null atos_core/csg_v130.py; \
  diff -u /dev/null scripts/smoke_csv_csg_v130.py; \
  diff -u /dev/null scripts/verify_csv_csg_v130_chain.py; \
  diff -u /dev/null tests/test_csg_v130.py \
) > patches/v130_csv_csg_concept_graph.diff
```

## Determinismo (evidência)
- `summary_sha256` (try1==try2) = `20d231ba45101168196ab3535857eeda6c1f4ad2f8d18ec7f3e23348561586d6`
- `results/run_csv_csg_v130_seed0_try1/eval.json` sha256 == `619e955d1257538d984110d1d7be68b0a5772836f444b1bcca8c8c9c89286a85`
- `results/run_csv_csg_v130_seed0_try2/eval.json` sha256 == `619e955d1257538d984110d1d7be68b0a5772836f444b1bcca8c8c9c89286a85`

## Artefatos WORM + SHA256
- Patch diff: `patches/v130_csv_csg_concept_graph.diff` = `f960aad9c0f6830d64e565e241d304cda47e9c5901e7ae3f1e049a3231c66c64`
- Smoke stdout: `results/smoke_csv_csg_v130_try1.json` = `fdc5c204b4866a48b5729bcbc1c52c24eac3523350b6e1bf97f84fa99b53bfe3`
- Smoke try1 eval: `results/run_csv_csg_v130_seed0_try1/eval.json` = `619e955d1257538d984110d1d7be68b0a5772836f444b1bcca8c8c9c89286a85`
- Smoke try1 summary: `results/run_csv_csg_v130_seed0_try1/smoke_summary.json` = `e7b461cf6f215605192ebc16f9c4882a720c1ce55ab885f3a2123f2c2597590a`
- Smoke try1 concepts: `results/run_csv_csg_v130_seed0_try1/concepts.jsonl` = `fa25e7dd3c87c0db49e60cd9a844513c8cede23e87fb5202b63c173ed39c5ec7`
- Smoke try1 evidence: `results/run_csv_csg_v130_seed0_try1/concept_evidence.jsonl` = `490a6770b752ff4525d13b0a460a4e74c2bac5ae8f9253d2c296fb1e7c5d4a1d`
- Smoke try1 telemetry: `results/run_csv_csg_v130_seed0_try1/concept_telemetry.jsonl` = `2f4565e9b60f2889b01e4123ad91dc7776c6672d6c2949a84b69dde60e877bec`
- Verify chain try1: `results/verify_csv_csg_v130_chain_try1.json` = `ed8aeabfbf1a751b56f2e895003ef0558251638b1e139e83d9bf1655d059b8b0`
- Verify chain try2: `results/verify_csv_csg_v130_chain_try2.json` = `bc919461f4ad77b7d3d98e10622f6a48cbd76f4ec915d2d2e4f24809179f28db`

## GO / NO-GO
- GO: CSV/CSG V130 com identidade por hash e replay auditável, logs WORM hash-chained com verificação, smoke determinístico try1==try2 e bloqueio explícito por match.
