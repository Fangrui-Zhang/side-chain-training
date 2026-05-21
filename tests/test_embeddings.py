import pandas as pd
import pytest

from dina2.embeddings import _index_filename, _select_manifest_rows


def _manifest(n=10):
    return pd.DataFrame(
        {
            "protein_id": [f"p{i}" for i in range(n)],
            "sequence": ["ACD"] * n,
        }
    )


def test_select_manifest_rows_supports_offset_count_and_shards():
    selected = _select_manifest_rows(_manifest(), start=2, count=6, shard_index=1, num_shards=3)

    assert selected["protein_id"].tolist() == ["p3", "p6"]


def test_select_manifest_rows_rejects_partial_shard_config():
    with pytest.raises(ValueError, match="shard-index and num-shards"):
        _select_manifest_rows(_manifest(), shard_index=0)


def test_index_filename_uses_shard_suffix_only_for_sharded_runs():
    assert _index_filename(None, None) == "embedding_index.csv"
    assert _index_filename(3, 20) == "embedding_index_shard_0003_of_0020.csv"
