from __future__ import annotations

from train_structural import FailureCluster, FailureClusterStore, _induce_operator_from_failure_cluster


def test_failure_cluster_missing_operator_is_eligible_immediately() -> None:
    store = FailureClusterStore(threshold=10, cooldown_steps=0, max_induce_attempts=3)

    sig_missing = {"cluster_ctx_key": "ck", "reason": "missing_operator_nav_clearing", "world_features": {}}
    store.observe(sig_missing, step=1, witness_env_id="MiniGrid-Test-v0", witness_seed=123)

    eligible = store.eligible_clusters(step=1)
    assert eligible, "missing_operator clusters must not wait for recurrence under strict burn"
    assert eligible[0].signature["reason"] == "missing_operator_nav_clearing"

    store2 = FailureClusterStore(threshold=10, cooldown_steps=0, max_induce_attempts=3)
    sig_other = {"cluster_ctx_key": "ck2", "reason": "no_path", "world_features": {}}
    store2.observe(sig_other, step=1, witness_env_id="MiniGrid-Test-v0", witness_seed=123)
    assert store2.eligible_clusters(step=1) == []


def test_failure_cluster_witnesses_are_sorted_unique_and_bounded() -> None:
    store = FailureClusterStore(threshold=10, cooldown_steps=0, max_induce_attempts=3)
    sig = {"cluster_ctx_key": "ck", "reason": "missing_operator_nav_clearing", "world_features": {}}
    c = store.observe(sig, step=1, witness_env_id="B", witness_seed=2)
    c = store.observe(sig, step=2, witness_env_id="A", witness_seed=1)
    c = store.observe(sig, step=3, witness_env_id="A", witness_seed=1)
    assert c.witnesses == [{"env_id": "A", "seed": 1}, {"env_id": "B", "seed": 2}]


def test_induce_operator_nav_clearing_v5() -> None:
    cluster = FailureCluster(
        cluster_id="fc_test",
        signature={
            "reason": "missing_operator_nav_clearing_v5",
            "world_features": {"doors_locked_bucket": "0", "keys_bucket": "0", "boxes_bucket": "1", "balls_bucket": "0"},
            "plan_fail_concept": {"name": "NAVIGATE_TO"},
        },
    )
    op = _induce_operator_from_failure_cluster(10, cluster)
    assert op is not None
    assert op.impl == "NAV_CLEARING_V5"
    assert op.signature.get("covers_reason") == "missing_operator_nav_clearing_v5"


def test_induce_operator_unbox_key_v1() -> None:
    cluster = FailureCluster(
        cluster_id="fc_test_unbox",
        signature={
            "reason": "missing_operator_unbox_key",
            "world_features": {"doors_locked_bucket": "1", "keys_bucket": "0", "boxes_bucket": "1", "balls_bucket": "0"},
            "plan_fail_concept": {"name": "UNBOX_KEY_FOR_DOOR"},
        },
    )
    op = _induce_operator_from_failure_cluster(10, cluster)
    assert op is not None
    assert op.impl == "UNBOX_KEY_V1"
    assert op.signature.get("covers_reason") == "missing_operator_unbox_key"


def test_induce_operator_nav_clearing_requires_blockers() -> None:
    cluster = FailureCluster(
        cluster_id="fc_test_no_blockers",
        signature={
            "reason": "missing_operator_nav_clearing",
            "world_features": {"doors_locked_bucket": "0", "keys_bucket": "0", "boxes_bucket": "0", "balls_bucket": "0"},
            "plan_fail_concept": {"name": "NAVIGATE_TO"},
        },
    )
    op = _induce_operator_from_failure_cluster(10, cluster)
    assert op is None

