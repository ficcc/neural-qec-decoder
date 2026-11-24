"""The sound module must be silent and harmless where Web Audio is missing (needs Node)."""
import shutil
import subprocess

import pytest

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")


def test_sound_calls_are_safe_without_web_audio():
    code = ("globalThis.self=globalThis;require('./docs/eli5/audio.js');const S=globalThis.Sound;"
            "S.flip();S.alarm(2);S.ok();S.bad();S.pop(1);S.tick();S.set(true);S.flip();S.ok();"
            "console.log(JSON.stringify({on:S.isOn()}))")
    out = subprocess.run([NODE, "-e", code], capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    assert '"on":true' in out.stdout
