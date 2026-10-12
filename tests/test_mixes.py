import pandas as pd
import pytest

from build_mixes import MIXES, cer_path


def load_nodes(mix):
    folder = cer_path(mix) / "nodes"
    return {p.stem: pd.read_csv(p, index_col="datetime") for p in sorted(folder.glob("*/*.csv"))}


@pytest.mark.parametrize("mix", [m for m in MIXES if m != "full"])
def test_mix_only_changes_load_shape(mix):
    if not cer_path(mix).exists():
        pytest.skip("run scripts/build_mixes.py first")
    full, other = load_nodes("full"), load_nodes(mix)
    assert full.keys() == other.keys()
    for member, df in full.items():
        assert other[member]["consumption"].sum() == pytest.approx(df["consumption"].sum(), rel=1e-9)
        for col in ("production", "price_imp", "price_exp"):
            if col in df:
                assert (other[member][col] == df[col]).all()
        changed = not (other[member]["consumption"] == df["consumption"]).all()
        assert changed == (member in MIXES[mix])
