"""Unicode-safe image read/write.

``cv2.imread`` / ``cv2.imwrite`` open the file with the C runtime's narrow
``fopen``, which on Windows uses the system ANSI code page. A path under a
folder whose name has Cyrillic (or any non-Latin) characters therefore can't be
opened: ``imread`` returns ``None`` and the app reports "Could not read … —
unsupported or corrupt", so the photo simply refuses to load. Reported on
Discord for 0.3.0.

The fix is to keep the byte I/O in Python — whose ``open`` is fully Unicode on
Windows — and hand OpenCV only the encoded bytes, via ``imdecode`` / ``imencode``.
Those touch no filenames, so they are locale-independent. The decode/encode
behaviour (codecs, flags) is otherwise identical to ``imread``/``imwrite``.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np


def imread(path: str | os.PathLike[str], flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray | None:
    """Drop-in for ``cv2.imread`` that works with any-locale paths.

    Returns ``None`` exactly where ``cv2.imread`` would — a missing/empty file
    or bytes no codec accepts — so callers keep their existing None checks.
    """
    try:
        with open(os.fspath(path), "rb") as handle:
            raw = handle.read()
    except OSError:
        return None
    if not raw:
        return None
    return cv2.imdecode(np.frombuffer(raw, np.uint8), flags)


def imwrite(
    path: str | os.PathLike[str],
    image: np.ndarray,
    params: Sequence[int] | None = None,
) -> bool:
    """Drop-in for ``cv2.imwrite`` that works with any-locale paths.

    The codec is chosen from the file extension, exactly as ``cv2.imwrite`` does.
    Returns ``False`` (rather than raising) when encoding or writing fails, to
    match the bool contract callers branch on.
    """
    target = Path(path)
    try:
        ok, buffer = cv2.imencode(target.suffix, image, list(params or []))
    except cv2.error:
        return False
    if not ok:
        return False
    try:
        with open(os.fspath(target), "wb") as handle:
            handle.write(buffer.tobytes())
    except OSError:
        return False
    return True
