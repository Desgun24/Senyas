import pandas as pd

from senyas.dataset import assert_no_group_leakage, assign_group_splits


def synthetic_manifest() -> pd.DataFrame:
    rows = []
    for signer in range(20):
        for class_id in range(5):
            rows.append({"signer_id": str(signer), "id_label": class_id, "vid_path": f"{class_id}/{signer}.MOV"})
    return pd.DataFrame(rows)


def test_group_split_is_disjoint_and_close_to_requested_ratio() -> None:
    split = assign_group_splits(synthetic_manifest(), seed=7, trials=300)
    assert_no_group_leakage(split)
    ratios = split["split"].value_counts(normalize=True).to_dict()
    assert abs(ratios["train"] - 0.70) <= 0.06
    assert abs(ratios["validation"] - 0.15) <= 0.06
    assert abs(ratios["test"] - 0.15) <= 0.06
    assert split.groupby("split")["id_label"].nunique().eq(5).all()


def test_group_split_is_reproducible() -> None:
    first = assign_group_splits(synthetic_manifest(), seed=42, trials=100)
    second = assign_group_splits(synthetic_manifest(), seed=42, trials=100)
    assert first["split"].tolist() == second["split"].tolist()

