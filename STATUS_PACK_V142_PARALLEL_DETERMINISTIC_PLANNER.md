```markdown
# STATUS_PACK_V142_PARALLEL_DETERMINISTIC_PLANNER

## Summary
- Implements a parallel deterministic planning loop for ARC solver that distributes search budget across 16 CPU cores
- Maintains full auditability with separate logs per worker merged into WORM-compliant ledger
- Adds reachability pruning heuristics (shape + palette + stage) for early termination
- Supports ambiguity detection (fail-closed: UNKNOWN when multiple distinct test outputs)
- Deterministic: same seed + inputs → same outputs regardless of execution timing

## Architecture

### Components
1. **parallel_planner_v142.py** - Core parallel search coordinator
   - `SearchBudgetV142`: Explicit budget for deterministic search
   - `WorkerBudgetV142`: Per-worker budget slice with branch assignment
   - `ParallelPlannerV142`: Main coordinator using ProcessPoolExecutor
   - `ReachabilityStateV142`: Abstract state for pruning

2. **run_arc_parallel_v142.py** - CLI runner for ARC tasks
   - Integrates with existing ARC dataset structure
   - Supports concept/macro banks
   - Produces WORM-compliant ledger

3. **test_parallel_planner_v142.py** - Unit tests (25 tests, all passing)

### Budget Distribution
The search budget is divided among workers by splitting the operator index space:
```
Worker 0: operators[0:25]   -> programs_start=0, programs_end=125
Worker 1: operators[25:50]  -> programs_start=125, programs_end=250
Worker 2: operators[50:75]  -> programs_start=250, programs_end=375
Worker 3: operators[75:100] -> programs_start=375, programs_end=500
```
Each worker explores programs whose first operator index falls in its assigned slice.

### Pruning Heuristics
- **Shape reachability**: Prune if min_steps_to_shape_change > steps_left
- **Palette reachability**: Prune if min_steps_to_modify > steps_left
- **Stage awareness**: Different stages (grid/objset/obj/bbox/patch) have different costs

### WORM Compliance
- Each search produces a `ParallelSearchResultV142` with merged logs
- Logs are merged deterministically by (timestamp, worker_id, event)
- Ledger entries chain via prev_hash for tamper detection
- Timestamps use deterministic counters (not wall-clock) for reproducibility

## Commands
```bash
# Compile check
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m py_compile atos_core/parallel_planner_v142.py scripts/run_arc_parallel_v142.py tests/test_parallel_planner_v142.py

# Run unit tests
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m unittest tests.test_parallel_planner_v142 -v

# Run parallel solver on ARC-AGI training (example)
python3 scripts/run_arc_parallel_v142.py \
  --arc_root ARC-AGI \
  --split training \
  --limit 100 \
  --jobs 16 \
  --max_programs 2000 \
  --max_depth 4 \
  --seed 0 \
  --out_base results/run_arc_parallel_v142_training_100
```

## Test Results
```
test_ambiguity_different_outputs ... ok
test_ambiguity_limit_respected ... ok
test_no_ambiguity_same_output ... ok
test_no_ambiguity_single_program ... ok
test_budget_distribution_deterministic ... ok
test_planner_deterministic_same_seed ... ok
test_write_to_ledger_chain_hashes ... ok
test_write_to_ledger_creates_file ... ok
test_config_to_dict ... ok
test_planner_result_to_dict ... ok
test_planner_search_returns_result ... ok
test_can_reach_target_needs_modification ... ok
test_can_reach_target_same_state ... ok
test_can_reach_target_shape_grow ... ok
test_min_steps_shape_change_grow ... ok
test_min_steps_to_modify_grid ... ok
test_min_steps_to_modify_objset ... ok
test_reachability_state_key_deterministic ... ok
test_budget_to_dict_schema_version ... ok
test_distribute_budget_deterministic ... ok
test_distribute_budget_even_split ... ok
test_distribute_budget_more_workers_than_ops ... ok
test_log_entry_to_dict ... ok
test_merge_worker_logs_deterministic_order ... ok
test_merge_worker_logs_hash_deterministic ... ok

Ran 25 tests in 0.033s - OK
```

## Files Added (V142)
- `atos_core/parallel_planner_v142.py`
- `scripts/run_arc_parallel_v142.py`
- `tests/test_parallel_planner_v142.py`

## Integration with Existing Components

### Engine Integration
The parallel planner works alongside `engine.py`:
- Engine handles single-threaded concept execution
- Parallel planner distributes search across workers
- Each worker can invoke Engine for program evaluation

### Ledger Integration
Compatible with existing ledger format:
- Entries include `prev_hash` for chain verification
- WORM compliance: write-once semantics
- Deterministic hashes via `canonical_json_dumps`

### Solver Integration
Designed to complement `arc_solver_v141.py`:
- Uses same operator definitions (`OP_DEFS_V141`)
- Same pruning heuristics (reachability)
- Shares concept/macro bank format

## Next Steps

### Priority 1: Integrate with arc_solver_v141
- Connect worker search to actual `solve_arc_task_v141` evaluation
- Share apply_cache across workers for efficiency
- Implement early termination on first train-perfect solution

### Priority 2: Operator Coverage
Analyze failure_counts from V140/V141 runs:
- 193 MISSING_OPERATOR (training)
- 140 MISSING_OPERATOR (evaluation)
Identify and implement missing operators based on failure analysis.

### Priority 3: Concept Composition
Extend ToC (Theory of Concepts) for:
- Concept recombination from successful programs
- Simplification via MDL principle
- Cross-task transfer with dependency tracking

### Priority 4: Evaluation Metrics
Add telemetry for:
- Programs evaluated per second per worker
- Pruning effectiveness (nodes pruned / total)
- Cache hit rates across workers

## Invariants Preserved
✓ Deterministic (seed-controlled)
✓ No neural components
✓ Fail-closed on ambiguity
✓ WORM-compliant logging
✓ Explicit budget (no hidden timeouts)
✓ All state in JSON-serializable structures

```
