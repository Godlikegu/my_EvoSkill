from myevoskill.datasets import DatasetManifest, FamilySplit


def test_family_split_rejects_overlap():
    try:
        FamilySplit(
            family="optics",
            distill_train=["task_a", "task_b"],
            transfer_val=["task_b", "task_c"],
        )
    except ValueError as exc:
        assert "overlapping split entries" in str(exc)
    else:
        raise AssertionError("expected overlap validation error")


def test_manifest_detects_validation_leakage():
    manifest = DatasetManifest.from_splits(
        [
            FamilySplit(
                family="optics",
                distill_train=["task_a", "task_b"],
                transfer_val=["task_c"],
            )
        ]
    )
    try:
        manifest.assert_no_validation_leakage("optics", ["task_b", "task_c"])
    except ValueError as exc:
        assert "validation leakage detected" in str(exc)
    else:
        raise AssertionError("expected validation leakage detection")

