# Internal Language (ACT / concept_csv)

This system does not learn in weights. It learns by creating/editing explicit **Acts**.

## Act (explicit learned object)

An `Act` is a JSON-serializable object stored in `acts.jsonl`:
- `kind`: e.g. `predictor`, `rewrite_rule`, `concept_csv`, `goal`
- `program`: list of `Instruction` ops (deterministic bytecode)
- `evidence`: interface + provenance (audit)
- `cost`: explicit overhead (MDL-style)
- `active`: on/off (prune without deletion)

## concept_csv (callable semantic object)

`concept_csv` is the unit of explicit semantic learning:
- Has an explicit **interface**: `input_schema`, `output_schema`, `validator_id`
- Is **invocable** by the runtime via `CSV_CALL`
- Can be **composed** (concept calling other concepts), creating a hierarchy

### Core ops (examples)

- `CSV_GET_INPUT`: binds an input field into a local variable
- `CSV_PRIMITIVE`: executes a named primitive function (`fn`) on variables
- `CSV_CALL`: calls another `concept_csv` by `concept_id` with a bind map
- `CSV_RETURN`: returns a variable as the concept output

## Auditing and provenance

### Ledger (WORM)
Every training window appends to `ledger.jsonl`:
- patch applied (ADD/MERGE/PRUNE/REWRITE) or `null`
- snapshot hash (`acts_hash`)
- snapshot path (explicit state)
- a chained `prev_hash` for tamper-evidence

### Snapshots
Every window writes a full explicit snapshot:
- `snapshots/stepXXXXXX_acts.jsonl`

## Inspecting “what the model knows”

Use:

```bash
python3 scripts/inspect_concept_csv.py --run <RUN_DIR> --concept_id <CONCEPT_ID>
```

This prints the exact internal program for a concept, its call dependencies, and its stored static depth.

