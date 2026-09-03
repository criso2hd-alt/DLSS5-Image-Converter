import os
import sys

# A PyInstaller --windowed build has no console, so sys.stdout and sys.stderr
# are None. Anything that writes to them then dies with
# "'NoneType' object has no attribute 'write'" — which is what huggingface_hub's
# tqdm progress bar does during the first model download, and it surfaced to
# users as a failed download with a baffling message.
#
# This was invisible during development because every test run redirected
# stderr to a file, which handed the process a real stream. Guard first, before
# importing anything that might write while being imported.
for _name in ("stdout", "stderr"):
    if getattr(sys, _name, None) is None:
        setattr(sys, _name, open(os.devnull, "w", encoding="utf-8"))

# Right after the streams are safe and before anything heavy is imported: turn a
# silent "crashes to desktop" into a written crash.log. A windowed build has no
# console and the guard above sends stderr to the void, so without this a crash -
# Python or native - leaves nothing to diagnose.
from dlss5_converter import crashlog  # noqa: E402

crashlog.install()

from dlss5_converter.app import main  # noqa: E402 - must follow the guard above

if __name__ == "__main__":
    # `DLSS5Converter.exe --selftest` prints what the runtime can actually do and
    # exits. It exists because "the window opened" proves very little: the
    # downloaded PyTorch is only *located* at startup, not imported, so a missing
    # dependency would not surface until the first conversion. It doubles as the
    # thing to paste into a bug report.
    if "--selftest" in sys.argv:
        from dlss5_converter.selftest import run_selftest

        sys.exit(run_selftest())
    main()
