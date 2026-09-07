"""The shared-memory colour transport and its handshake.

The native harness cannot be built or run in CI, so these tests stand in a fake
harness — a small Python script that speaks the same line protocol — to exercise
the parts that live on the Python side: the mapping, the zero-copy buffer, the
capability handshake, and the fallback to files against an older harness that
rejects the new flag.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from dlss5_converter.evaluator import Harness
from dlss5_converter.settings import NeuralSettings

# A fake harness that supports shared memory: it maps the same named section and
# checks the value the parent wrote, so a transport bug surfaces as an ERROR.
_FAKE_NEW = """
import sys, mmap, numpy as np
a = sys.argv[1:]
def val(f): return a[a.index(f)+1]
w = int(val('--width')); h = int(val('--height'))
shm = None
if '--colour-shmem' in a:
    i = a.index('--colour-shmem'); name = a[i+1]; nbytes = int(a[i+2])
    try: shm = mmap.mmap(-1, nbytes, tagname=name)
    except Exception: shm = None
notes = 'DLSS feature created in DLAA mode' + (' shmem:colour' if shm else '')
print('READY ' + notes, flush=True)
for line in sys.stdin:
    s = line.strip()
    if s.startswith('FRAME'):
        p = s.split()[1]
        if p == '@shmem' and shm:
            v = float(np.frombuffer(shm, dtype=np.float16, count=w*h*4)[0])
            if abs(v - 0.25) > 1e-3:
                print('ERROR shmem mismatch %r' % v, flush=True); sys.exit(1)
        print('FRAME_OK 1', flush=True)
    elif s.startswith('WRITE'): print('WRITE_OK 42', flush=True)
    elif s.startswith('QUIT') or s == '': break
print('BYE', flush=True)
"""

# A fake harness that predates the flag: it Fails on the unknown argument, just
# as the real one does, so the parent must retry without it.
_FAKE_OLD = """
import sys
a = sys.argv[1:]
if '--colour-shmem' in a:
    print('ERROR Unknown argument: --colour-shmem', flush=True); sys.exit(1)
print('READY DLSS feature created in DLAA mode', flush=True)
for line in sys.stdin:
    s = line.strip()
    if s.startswith('FRAME'): print('FRAME_OK 1', flush=True)
    elif s.startswith('WRITE'): print('WRITE_OK 7', flush=True)
    elif s.startswith('QUIT') or s == '': break
print('BYE', flush=True)
"""


def _harness(script: Path, use_shmem: bool) -> Harness:
    h = Harness(
        script.parent / "dummy.exe", width=4, height=2,
        depth_path=Path("d"), motion_path=Path("m"),
        neural=NeuralSettings(), frames=1, use_shmem=use_shmem,
    )
    # Drive the fake script through the interpreter in place of a real exe.
    h._command = [sys.executable, str(script)] + h._command[1:]
    return h


@pytest.fixture
def fake_new(tmp_path):
    path = tmp_path / "fake_new.py"
    path.write_text(_FAKE_NEW, encoding="utf-8")
    return path


@pytest.fixture
def fake_old(tmp_path):
    path = tmp_path / "fake_old.py"
    path.write_text(_FAKE_OLD, encoding="utf-8")
    return path


def test_shared_memory_is_a_zero_copy_writable_view():
    import mmap

    h = Harness(
        Path("dummy.exe"), width=4, height=2, depth_path=Path("d"),
        motion_path=Path("m"), neural=NeuralSettings(), frames=1, use_shmem=True,
    )
    h._shmem = mmap.mmap(-1, h._colour_bytes, tagname="dlss5_unit_shm")
    h._shmem_active = True
    try:
        buf = h.colour_buffer((2, 4, 4))
        assert buf.shape == (2, 4, 4) and buf.dtype == np.float16
        assert buf.flags.writeable
        buf[:] = 1.5
        raw = np.frombuffer(h._shmem, dtype=np.float16)
        assert np.all(raw[: 2 * 4 * 4] == 1.5)  # writes land straight in the map
    finally:
        h._teardown_shmem()  # must not raise BufferError with the view alive
    assert h._shmem is None and h._shmem_active is False


def test_new_harness_negotiates_shared_memory(fake_new, tmp_path):
    colour_path = tmp_path / "colour.bin"
    h = _harness(fake_new, use_shmem=True)
    h.__enter__()
    try:
        assert h._shmem_active is True
        assert "shmem:colour" in h.notes
        buf = h.colour_buffer((2, 4, 4))
        buf[:] = 0.25  # the fake asserts this exact value over the mapping
        h.commit_colour(buf, colour_path, (0.0, 0.0))  # raises if it mismatched
        assert not colour_path.exists()  # nothing written to disk on the shmem path
    finally:
        h.__exit__(None, None, None)


def test_old_harness_falls_back_to_files(fake_old, tmp_path):
    colour_path = tmp_path / "colour.bin"
    h = _harness(fake_old, use_shmem=True)
    h.__enter__()
    try:
        assert h._shmem_active is False  # the flag was rejected; retried on files
        buf = h.colour_buffer((2, 4, 4))
        buf[:] = 0.5
        h.commit_colour(buf, colour_path, (0.0, 0.0))
        assert colour_path.exists()
        assert colour_path.stat().st_size == h._colour_bytes
    finally:
        h.__exit__(None, None, None)


def test_use_shmem_false_never_maps_anything(fake_new, tmp_path):
    # The single-image path opts out and must stay on files even against a
    # harness that would happily do shared memory.
    colour_path = tmp_path / "colour.bin"
    h = _harness(fake_new, use_shmem=False)
    h.__enter__()
    try:
        assert h._shmem_active is False
        assert h._shmem is None
        buf = h.colour_buffer((2, 4, 4))
        buf[:] = 0.9
        h.commit_colour(buf, colour_path, (0.0, 0.0))
        assert colour_path.exists()
    finally:
        h.__exit__(None, None, None)
