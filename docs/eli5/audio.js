/* Tiny sound effects for the explainer (Web Audio, no files). Off until the reader switches it on.
 * Every sound is paired with something visible on the page, so nothing depends on hearing it. */
(function (root) {
  let ctx = null, on = false;
  function ensure() {
    try {
      if (!ctx) { const C = root.AudioContext || root.webkitAudioContext; if (!C) return false; ctx = new C(); }
      if (ctx.state === "suspended") ctx.resume();
      return true;
    } catch (e) { return false; }
  }
  function tone(freq, dur, type, vol, delay) {
    if (!on || !ensure()) return;
    const t = ctx.currentTime + (delay || 0), o = ctx.createOscillator(), g = ctx.createGain();
    o.type = type || "sine";
    o.frequency.setValueAtTime(freq, t);
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(vol || 0.1, t + 0.012);
    g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
    o.connect(g); g.connect(ctx.destination);
    o.start(t); o.stop(t + dur + 0.03);
  }
  const A = {
    set(v) { on = !!v; if (on) { ensure(); tone(660, 0.09, "sine", 0.08); } },
    isOn() { return on; },
    flip() { tone(200, 0.09, "triangle", 0.14); },                         // a part flips
    alarm(i, delay) { tone(820 + 55 * (i || 0), 0.16, "square", 0.04, delay || 0); tone(820 + 55 * (i || 0), 0.16, "square", 0.04, (delay || 0) + 0.2); },
    pop(i) { tone(520 + 40 * (i || 0), 0.1, "sine", 0.12); },              // helper clears something
    ok() { [523, 659, 784].forEach((f, k) => tone(f, 0.22, "sine", 0.1, k * 0.11)); },
    bad() { tone(170, 0.45, "sawtooth", 0.07); tone(128, 0.45, "sawtooth", 0.06, 0.12); },
    tick() { tone(980, 0.04, "sine", 0.05); }
  };
  root.Sound = A;
})(typeof self !== "undefined" ? self : this);
