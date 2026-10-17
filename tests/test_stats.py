from permissionlab.stats import cluster_bootstrap, paired_cluster_bootstrap


def rows(task_values):
    return [{"task_id": t, "v": v} for t, values in task_values.items() for v in values]


def test_constant_rate_has_zero_width_interval():
    r = cluster_bootstrap(rows({"a": [0, 0], "b": [0, 0]}), lambda x: x["v"], n_boot=200)
    assert (r["point"], r["lo"], r["hi"]) == (0.0, 0.0, 0.0)


def test_resampling_tasks_not_episodes_gives_honest_width():
    # One task always drifts, one never: many reps must not shrink the interval.
    r = cluster_bootstrap(rows({"a": [1] * 50, "b": [0] * 50}), lambda x: x["v"], n_boot=500)
    assert r["point"] == 0.5 and r["lo"] == 0.0 and r["hi"] == 1.0 and r["n_clusters"] == 2


def test_none_values_are_skipped_and_seed_is_deterministic():
    data = rows({"a": [1, None], "b": [0], "c": [1]})
    first = cluster_bootstrap(data, lambda x: x["v"], n_boot=300, seed=7)
    assert first == cluster_bootstrap(data, lambda x: x["v"], n_boot=300, seed=7)
    assert first["n"] == 3


def test_paired_difference_of_identical_conditions_is_zero():
    data = rows({"a": [1, 0], "b": [0, 0], "c": [1, 1]})
    d = paired_cluster_bootstrap(data, data, lambda x: x["v"], n_boot=200)
    assert (d["point"], d["lo"], d["hi"]) == (0.0, 0.0, 0.0)
