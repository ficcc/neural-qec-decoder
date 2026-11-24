from experiments.jobs import E1_GRID, E2_TEST_P, E2_TRAIN_GRID, all_jobs, ckpt_dir


def test_job_names_unique_and_counts():
    jobs = all_jobs()
    names = [n for n, _ in jobs]
    assert len(names) == len(set(names))
    by = lambda prefix: [c for n, c in jobs if n.startswith(prefix)]
    assert len(by("e1/")) == 9            # d in {3,5,7} x 3 seeds
    assert len(by("e2/")) == 27 + 6       # 3 scen x 3 inst x 3 d, + S2 i0 seeds 1,2 x 3 d
    assert len(by("e3/")) == 5            # S0 d35 x3 seeds, S1 pool d35, S1 pool d7
    assert len(by("e4/")) == 12           # 4 budgets x 3 seeds
    assert len(by("e6/")) == 5            # S4 d in {3,5,7} seed 0, + d=5 seeds 1,2


def test_gpu_split_is_d5_and_up():
    gpu = [n for n, c in all_jobs() if max(c.d_list) >= 5]
    cpu = [n for n, c in all_jobs() if max(c.d_list) < 5]
    assert len(gpu) + len(cpu) == len(all_jobs())
    assert all("d3" in n for n in cpu)


def test_grids_below_threshold_for_tests():
    assert max(E2_TEST_P) < max(E2_TRAIN_GRID) < max(E1_GRID)


def test_e4_uses_fixed_budgets():
    e4 = [c for n, c in all_jobs() if n.startswith("e4/")]
    assert sorted({c.fixed_shots for c in e4}) == [1_000, 10_000, 100_000, 1_000_000]


def test_ckpt_dir_layout():
    assert str(ckpt_dir("e1/d5_s0")).replace("\\", "/") == "checkpoints/e1/d5_s0"


def test_select_by_distance_and_prefix():
    from experiments.train_jobs import select
    jobs = all_jobs()
    gpu = select(jobs, min_d=5, max_d=99, only=())
    assert all(max(c.d_list) >= 5 for _, c in gpu)
    e1_cpu = select(jobs, min_d=0, max_d=3, only=("e1/",))
    assert [n for n, _ in e1_cpu] == ["e1/d3_s0", "e1/d3_s1", "e1/d3_s2"]


def test_pending_skips_finished(tmp_path, monkeypatch):
    from experiments import train_jobs
    monkeypatch.setattr(train_jobs, "ckpt_dir", lambda n: tmp_path / n)
    jobs = all_jobs()[:2]
    done = tmp_path / jobs[0][0]; done.mkdir(parents=True)
    (done / "best.pt").write_bytes(b"x"); (done / "log.json").write_text("{}")
    assert [n for n, _ in train_jobs.pending(jobs)] == [jobs[1][0]]


def test_timing_override_never_writes_real_checkpoints(tmp_path, monkeypatch):
    from experiments import train_jobs
    seen = []
    monkeypatch.setattr(train_jobs, "ckpt_dir", lambda n: tmp_path / n)
    monkeypatch.setattr(train_jobs, "train_gnn", lambda cfg, out: seen.append((cfg.steps, out)))
    train_jobs.main(["--only", "e1/d3_s0", "--max-d", "3", "--steps", "7", "--device", "cpu"])
    assert seen and seen[0][0] == 7
    assert seen[0][1].relative_to(tmp_path).parts[0] == "_timing"


def test_shards_partition_jobs():
    from experiments.train_jobs import shard
    jobs = all_jobs()
    parts = [shard(jobs, i, 3) for i in range(3)]
    names = [n for p in parts for n, _ in p]
    assert sorted(names) == sorted(n for n, _ in jobs) and len(names) == len(set(names))


def test_ablation_jobs_separate_and_deeper():
    from experiments.jobs import ablation_jobs
    abl = dict(ablation_jobs())
    assert not set(abl) & {n for n, _ in all_jobs()}
    cfg = abl["abl/d7_L8_H96_s0"]
    assert (cfg.layers, cfg.hidden, cfg.d_list) == (8, 96, (7,))


def test_e6_grid_below_s4_threshold():
    from experiments.jobs import E6_TEST_P, E6_TRAIN_GRID
    assert max(E6_TRAIN_GRID) <= 0.0035 and max(E6_TEST_P) < max(E6_TRAIN_GRID)


def test_warmstart_jobs_differ_from_scratch_only_in_init():
    from dataclasses import replace
    from experiments.jobs import warmstart_jobs
    jobs, scratch = dict(warmstart_jobs()), dict(all_jobs())
    assert sorted(jobs) == ["ws/d7_from_d35_s0", "ws/d7_from_d35_s1", "ws/d7_from_d35_s2", "ws/d7_from_d5_s0"]
    for name, cfg in jobs.items():
        s = int(name[-1])
        assert replace(cfg, init_from=None) == scratch[f"e1/d7_s{s}"]          # same recipe, same seeds
        src = f"e3/S0_d35_s{s}" if "d35" in name else "e1/d5_s0"
        assert cfg.init_from.replace("\\", "/").endswith(f"{src}/best.pt")
        assert src in scratch


def test_train_jobs_warmstart_flag_lists_them(capsys, tmp_path, monkeypatch):
    from experiments import train_jobs
    monkeypatch.setattr(train_jobs, "ckpt_dir", lambda n: tmp_path / n)   # nothing finished, whatever is on disk
    train_jobs.main(["--warmstart", "--list", "--device", "cpu"])
    out = capsys.readouterr().out
    assert "ws/d7_from_d35_s0" in out and "e1/d7_s0" not in out
