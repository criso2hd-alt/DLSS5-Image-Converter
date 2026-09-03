"""DLSS 5 Image & Video Converter."""

from __future__ import annotations

import os

__version__ = "0.1.0"

# OpenCV ships the OpenEXR codec but disables it at runtime unless this is set
# (opencv/opencv#21326). EXR matters more here than in a normal photo app: the
# neural pass works in linear HDR, so an EXR round-trip is the only lossless way
# to get a result back out. Must be set before cv2 is first imported anywhere.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
