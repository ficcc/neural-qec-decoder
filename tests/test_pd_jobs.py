import shutil
import subprocess
from pathlib import Path

import pytest

from experiments import train_jobs
from experiments.jobs import E1_GRID, predecoder_jobs


def test_predecoder_jobs_names_and_configs():
    jobs = dict(predecoder_jobs())
    assert list(jobs) == ["pd/d35_s0", "pd/d35_s1", "pd/d35_s2", "pd/d7_s0"]
    for name, cfg in jobs.items():
        assert (cfg.scenario, cfg.instance, cfg.p_grid, cfg.steps) == ("S0", 0, E1_GRID, 12000)
        assert cfg.d_list == ((7,) if name == "pd/d7_s0" else (3, 5))
        assert cfg.seed == int(name[-1])


def test_train_jobs_predecoder_flag_lists_them(capsys, tmp_path, monkeypatch):
    monkeypatch.setattr(train_jobs, "ckpt_dir", lambda n: tmp_path / n)   # nothing finished, whatever is on disk
    train_jobs.main(["--predecoder", "--list", "--device", "cpu"])
    out = capsys.readouterr().out
    assert all(n in out for n in ("pd/d35_s0", "pd/d35_s1", "pd/d35_s2", "pd/d7_s0"))
    assert "e1/" not in out and "ws/" not in out


def test_predecoder_timing_never_writes_real_checkpoints(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(train_jobs, "ckpt_dir", lambda n: tmp_path / n)
    monkeypatch.setattr(train_jobs, "train_pd", lambda cfg, out: seen.append((cfg.steps, Path(out))))
    train_jobs.main(["--predecoder", "--only", "pd/d35_s0", "--steps", "3", "--device", "cpu"])
    assert seen and seen[0][0] == 3
    assert seen[0][1].relative_to(tmp_path).parts[0] == "_timing"


def _working_bash():
    """A bash that can actually run (on Windows the first `bash` on PATH may be the WSL stub)."""
    for cand in (shutil.which("bash"), r"C:\Program Files\Git\bin\bash.exe", "/usr/bin/bash"):
        if cand:
            try:
                r = subprocess.run([cand, "-c", "echo ok"], capture_output=True, text=True, timeout=20)
            except OSError:
                continue
            if r.returncode == 0 and r.stdout.strip() == "ok":
                return cand
    return None


def test_pod_script_has_predecoder_mode_and_is_valid_bash():
    text = Path("scripts/pod_run.sh").read_text()
    assert "predecoder)" in text and "PREDECODER_DONE" in text and "--predecoder" in text
    assert b"\r" not in Path("scripts/pod_run.sh").read_bytes()
    bash = _working_bash()
    if bash is None:
        pytest.skip("no working bash available for the syntax check")
    assert subprocess.run([bash, "-n", "scripts/pod_run.sh"]).returncode == 0
