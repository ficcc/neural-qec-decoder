/* Toy simulations for the interactive explainer: a row of qubits with one alarm between each pair.
 * A qubit "flips" with some probability; an alarm rings when its two neighbours differ.
 * The detective chooses the cheapest correction that explains the alarms (minimum-weight decoding,
 * the one-dimensional version of matching). Tested against exact formulas in tests/test_eli5_sim.py. */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.Sim = factory();
})(typeof self !== "undefined" ? self : this, function () {
  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  const clip = (p) => Math.min(0.49, Math.max(1e-4, p));

  /* alarm j rings when qubits j and j+1 differ */
  function syndrome(e) {
    const s = new Array(e.length - 1);
    for (let j = 0; j < s.length; j++) s[j] = e[j] ^ e[j + 1];
    return s;
  }
  /* the two corrections consistent with the alarms: A (first qubit untouched) and its complement B */
  function candidates(s) {
    const A = [0];
    for (let j = 0; j < s.length; j++) A.push(A[j] ^ s[j]);
    return [A, A.map((x) => 1 - x)];
  }
  const weights = (p) => p.map((x) => Math.log((1 - clip(x)) / clip(x)));
  function cost(c, w) {
    let t = 0;
    for (let i = 0; i < c.length; i++) if (c[i]) t += w[i];
    return t;
  }
  function decode(s, w) {
    const [A, B] = candidates(s);
    return cost(B, w) < cost(A, w) ? B : A;     // a tie goes to A
  }
  /* both corrections explain the alarms, so what is left over is all-clear or the whole row flipped */
  const logicalError = (e, corr) => e[0] !== corr[0];
  const sample = (p, rand) => p.map((x) => (rand() < x ? 1 : 0));

  function failureRate(truep, w, trials, seed) {
    const rand = mulberry32(seed);
    let bad = 0;
    for (let t = 0; t < trials; t++) {
      const e = sample(truep, rand);
      if (logicalError(e, decode(syndrome(e), w))) bad++;
    }
    return bad / trials;
  }

  /* Redraw the map from alarm statistics alone (the same estimator the project uses for matching):
   * qubit j sits between alarms j-1 and j; the two end qubits touch only one alarm. */
  function estimateP(truep, shots, seed) {
    const n = truep.length, m = n - 1, rand = mulberry32(seed);
    const mean = new Array(m).fill(0), pair = new Array(m).fill(0);
    for (let t = 0; t < shots; t++) {
      const s = syndrome(sample(truep, rand));
      for (let j = 0; j < m; j++) {
        mean[j] += s[j];
        if (j + 1 < m) pair[j] += s[j] & s[j + 1];
      }
    }
    for (let j = 0; j < m; j++) { mean[j] /= shots; pair[j] /= shots; }
    const p = new Array(n).fill(0.01);
    for (let j = 1; j <= n - 2; j++) {          // interior qubits: from the pair of alarms next to them
      const x = mean[j - 1], y = mean[j], xy = pair[j - 1];
      const a = xy - x * y, b = 1 - 2 * x - 2 * y + 4 * xy;
      p[j] = b > 0 ? clip(0.5 - 0.5 * Math.sqrt(Math.max(0, 1 - (4 * a) / b))) : 1e-4;
    }
    p[0] = clip(0.5 - 0.5 * ((1 - 2 * mean[0]) / (1 - 2 * p[1])));
    p[n - 1] = clip(0.5 - 0.5 * ((1 - 2 * mean[m - 1]) / (1 - 2 * p[n - 2])));
    return p;
  }

  /* A "student" with no map: remember, for every alarm pattern seen in practice, which answer was right. */
  const key = (s) => s.reduce((k, v) => (k << 1) | v, 0);
  function trainTable(truep, shots, seed) {
    const rand = mulberry32(seed), table = new Map();
    for (let t = 0; t < shots; t++) {
      const e = sample(truep, rand), k = key(syndrome(e));
      const c = table.get(k) || [0, 0];
      c[e[0]]++;
      table.set(k, c);
    }
    return table;
  }
  function tableFailure(table, truep, trials, seed) {
    const rand = mulberry32(seed);
    let bad = 0;
    for (let t = 0; t < trials; t++) {
      const e = sample(truep, rand), c = table.get(key(syndrome(e)));
      const guess = c ? (c[1] > c[0] ? 1 : 0) : 0;   // never seen this pattern: just guess
      if (guess !== e[0]) bad++;
    }
    return bad / trials;
  }

  /* exact chance that a majority of n equally shaky qubits flip (a tie counts half) */
  function majorityFailure(n, p) {
    const C = (a, b) => { let r = 1; for (let i = 1; i <= b; i++) r = (r * (a - b + i)) / i; return r; };
    let t = 0;
    for (let k = 0; k <= n; k++) {
      const pk = C(n, k) * Math.pow(p, k) * Math.pow(1 - p, n - k);
      if (2 * k > n) t += pk; else if (2 * k === n) t += pk / 2;
    }
    return t;
  }

  return { mulberry32, syndrome, candidates, weights, decode, logicalError, sample, failureRate,
           estimateP, trainTable, tableFailure, majorityFailure };
});
