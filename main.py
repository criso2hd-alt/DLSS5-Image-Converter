import sys

from dlss5_converter.app import main

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
