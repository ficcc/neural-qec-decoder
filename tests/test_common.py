import json

from experiments.common import load_rows, my_points, out_path


def test_my_points_partition():
    pts = list(range(10))
    parts = [my_points(pts, f"{i}/3") for i in range(3)]
    assert sorted(x for p in parts for x in p) == pts


def test_load_rows_merges_shards(tmp_path):
    for i in range(2):
        out_path("e9", f"{i}/2", root=tmp_path).write_text(json.dumps({"decoder": "a", "k": i}) + "\n")
    rows = load_rows("e9", root=tmp_path)
    assert sorted(r["k"] for r in rows) == [0, 1]
