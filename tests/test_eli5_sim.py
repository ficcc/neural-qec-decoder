"""Checks the toy simulations behind docs/eli5/index.html against exact results (needs Node)."""
import json
import math
import shutil
import subprocess

import pytest

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")


def run_js(body: str):
    code = "const S=require('./docs/eli5/sim.js');" + body
    out = subprocess.run([NODE, "-e", code], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def binom_tail(n, p):
    """Probability that more than half of n qubits flip (majority vote fails)."""
    return sum(math.comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(n // 2 + 1, n + 1))


def test_decoder_fixes_up_to_half_the_qubits():
    r = run_js("""
      const w = S.weights([0.1,0.1,0.1,0.1,0.1]);
      const res = (e) => S.logicalError(e, S.decode(S.syndrome(e), w));
      console.log(JSON.stringify({one: res([0,0,1,0,0]), two: res([0,1,0,1,0]), three: res([1,1,1,0,0]), none: res([0,0,0,0,0]), all: res([1,1,1,1,1])}));
    """)
    assert r == {"one": False, "two": False, "three": True, "none": False, "all": True}


def test_alarms_ring_exactly_where_neighbours_differ():
    s = run_js("console.log(JSON.stringify(S.syndrome([0,1,1,0,1])))")
    assert s == [1, 0, 1, 1]


@pytest.mark.parametrize("n,p", [(3, 0.1), (5, 0.2), (7, 0.25), (9, 0.3)])
def test_failure_rate_matches_the_exact_binomial_formula(n, p):
    f = run_js(f"""
      const pr = Array({n}).fill({p});
      console.log(JSON.stringify(S.failureRate(pr, S.weights(pr), 200000, 7)));
    """)
    assert abs(f - binom_tail(n, p)) < 0.004


def test_bigger_buildings_are_safer_below_half():
    f = run_js("""
      const out = [3,5,7,9].map(n => { const pr = Array(n).fill(0.1); return S.failureRate(pr, S.weights(pr), 200000, 3); });
      console.log(JSON.stringify(out));
    """)
    assert f[0] > f[1] > f[2] > f[3]


def test_map_estimator_recovers_uniform_and_uneven_noise():
    est = run_js("""
      console.log(JSON.stringify({even: S.estimateP(Array(7).fill(0.1), 300000, 5),
                                  bad: S.estimateP([0.05,0.05,0.05,0.25,0.05,0.05,0.05], 300000, 5)}));
    """)
    assert all(abs(x - 0.1) < 0.02 for x in est["even"])
    for got, true in zip(est["bad"], [0.05, 0.05, 0.05, 0.25, 0.05, 0.05, 0.05]):
        assert abs(got - true) < 0.03


def test_true_map_is_never_worse_than_the_factory_map():
    true_p = [0.30, 0.04, 0.04, 0.04, 0.04, 0.04, 0.30]
    r = run_js(f"""
      const tp = {json.dumps(true_p)};
      const factory = S.weights(Array(7).fill(0.1));
      console.log(JSON.stringify({{factory: S.failureRate(tp, factory, 200000, 9), oracle: S.failureRate(tp, S.weights(tp), 200000, 9)}}));
    """)
    assert r["oracle"] <= r["factory"] + 0.002


def test_redrawn_map_is_close_to_the_true_map_with_enough_examples():
    true_p = [0.30, 0.04, 0.04, 0.04, 0.04, 0.04, 0.30]
    r = run_js(f"""
      const tp = {json.dumps(true_p)};
      const est = S.estimateP(tp, 100000, 11);
      console.log(JSON.stringify({{redrawn: S.failureRate(tp, S.weights(est), 200000, 13), oracle: S.failureRate(tp, S.weights(tp), 200000, 13)}}));
    """)
    assert abs(r["redrawn"] - r["oracle"]) < 0.003


def test_the_learner_needs_far_more_practice_than_the_map_redrawer():
    r = run_js("""
      const tp = Array(9).fill(0.2);
      const oracle = S.failureRate(tp, S.weights(tp), 100000, 21);
      const few = S.tableFailure(S.trainTable(tp, 30, 1), tp, 100000, 22);
      const lots = S.tableFailure(S.trainTable(tp, 300000, 1), tp, 100000, 22);
      const redrawn30 = S.failureRate(tp, S.weights(S.estimateP(tp, 300, 4)), 100000, 22);
      console.log(JSON.stringify({oracle, few, lots, redrawn30}));
    """)
    assert r["few"] - r["oracle"] > 0.03          # 30 examples: the learner is clearly worse
    assert r["lots"] - r["oracle"] < 0.01         # plenty of examples: it catches up
    assert r["redrawn30"] - r["oracle"] < 0.03    # the map-redrawer is already good with few examples


def test_simulation_is_reproducible_for_a_seed():
    a = run_js("const p=Array(5).fill(0.2); console.log(JSON.stringify(S.failureRate(p,S.weights(p),5000,99)))")
    b = run_js("const p=Array(5).fill(0.2); console.log(JSON.stringify(S.failureRate(p,S.weights(p),5000,99)))")
    assert a == b
