"""Prove the runtime actually works, rather than merely appearing to.

Written because a frozen build can pass every startup check and still fail on the
first conversion: ``bootstrap.is_ready`` uses ``find_spec``, which *locates*
PyTorch without importing it, so a missing transitive dependency stays invisible
until something tries to use it.

Output goes to stderr, which a windowed build still writes to a redirect, so:

    DLSS5Converter.exe --selftest 2> report.txt
"""

from __future__ import annotations

import sys
from pathlib import Path


def _line(text: str = "") -> None:
    print(text, file=sys.stderr, flush=True)


def run_selftest() -> int:
    """Return 0 if the app can do its job, 1 otherwise."""
    failures = 0

    _line("DLSS 5 Image Converter - self test")
    _line("=" * 46)

    from . import bootstrap, paths

    _line(f"frozen           : {paths.is_frozen()}")
    _line(f"app folder       : {paths.app_dir()}")
    _line(f"pytorch folder   : {bootstrap.runtime_dir()}")
    _line(f"models folder    : {paths.model_cache_dir()}")
    _line("")

    # --- the runtime, imported for real ---------------------------------
    bootstrap.activate()
    try:
        import torch

        _line(f"torch            : {torch.__version__}")
        _line(f"torch location   : {Path(torch.__file__).parent}")
        cuda = torch.cuda.is_available()
        _line(f"cuda available   : {cuda}")
        if cuda:
            _line(f"device           : {torch.cuda.get_device_name(0)}")
            result = (torch.randn(256, 256, device="cuda") @ torch.randn(256, 256, device="cuda"))
            _line(f"gpu matmul       : ok {tuple(result.shape)}")
        else:
            _line("gpu matmul       : SKIPPED - no CUDA device")
            failures += 1
    except Exception as error:  # noqa: BLE001 - reporting is the whole point
        _line(f"torch            : FAILED - {type(error).__name__}: {error}")
        failures += 1

    # --- everything the depth stage needs -------------------------------
    for name in ("transformers", "safetensors", "huggingface_hub", "cv2", "numpy", "PIL"):
        try:
            __import__(name)
            _line(f"{name:<17}: ok")
        except Exception as error:  # noqa: BLE001
            _line(f"{name:<17}: FAILED - {error}")
            failures += 1

    # --- the actual workload --------------------------------------------
    #
    # Imports succeeding is not the same as inference working, and this is the
    # step that uses the downloaded runtime in anger. Skipped rather than
    # triggered when the model is absent: a diagnostic should not start a
    # 400 MB download behind the user's back.
    try:
        from .depth_engine import DEFAULT_MODEL, DepthEngine
        from .settings import AppSettings

        if not DepthEngine.is_downloaded(DEFAULT_MODEL):
            _line("depth inference  : skipped - model not downloaded yet")
        else:
            import numpy as np

            settings = AppSettings()
            engine = DepthEngine()
            engine.load(DEFAULT_MODEL)
            probe = (np.random.default_rng(0).random((256, 256, 3)) * 255).astype("uint8")
            depth = engine.infer(probe, input_size=settings.depth.input_size)
            _line(
                f"depth inference  : ok {depth.shape} "
                f"range [{float(depth.min()):.3f}, {float(depth.max()):.3f}] "
                f"on {engine.device}"
            )
    except Exception as error:  # noqa: BLE001
        _line(f"depth inference  : FAILED - {type(error).__name__}: {error}")
        failures += 1

    # HDR input and output both go through Windows' own imaging codecs, which
    # are not something this app installs. A round trip through the encoder and
    # back is the only way to know they are really there on this machine.
    try:
        import numpy as np

        from . import wic

        if not wic.available():
            _line("jpeg xr          : unavailable (Windows imaging codecs)")
        else:
            probe_path = paths.scratch_dir() / "selftest_hdr.jxr"
            sample = np.full((8, 8, 3), 0.25, np.float32)
            sample[:, 4:, 0] = 8.0  # a highlight well above diffuse white
            wic.write(probe_path, sample)
            peak = float(wic.read(probe_path).max())
            ok = abs(peak - 8.0) < 0.05
            _line(f"jpeg xr          : {'ok' if ok else 'FAILED'} (peak {peak:.2f}, expected 8.00)")
            if not ok:
                failures += 1
    except Exception as error:  # noqa: BLE001
        _line(f"jpeg xr          : FAILED - {error}")
        failures += 1

    # The window itself, off screen. Every other check here exercises the
    # pipeline, and v0.1.15 shipped with three dead buttons while all of them
    # passed - the break was a missing method on MainWindow, reached only by a
    # click. Constructing the window and running the main-thread half of those
    # clicks costs a second and covers the artifact rather than the source.
    try:
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        import numpy as np
        from PySide6.QtWidgets import QApplication

        from . import pipeline
        from .app import MainWindow

        owned = QApplication.instance() is None
        application = QApplication.instance() or QApplication([])
        window = MainWindow()
        window.prepared = pipeline.Prepared(
            source=np.full((64, 96, 3), 0.5, np.float32),
            inverse_depth=np.zeros((64, 96), np.float32),
            linear=np.full((64, 96, 3), 0.2, np.float32),
        )
        window._begin_progress()
        window._report_progress("DLSS 5 pass 4 of 8")
        swept = window.depth_view._progress
        window._end_progress()
        for view in ("photo", "depth", "result", "difference"):
            window.show_view(view)
        window.deleteLater()
        if owned:
            application.processEvents()
        ok = swept == 0.5
        _line(f"window           : {'ok' if ok else 'FAILED'} (buttons reachable)")
        if not ok:
            failures += 1
    except Exception as error:  # noqa: BLE001
        _line(f"window           : FAILED - {type(error).__name__}: {error}")
        failures += 1

    _line("")

    # --- the DLSS side --------------------------------------------------
    from . import evaluator, runtime

    status = runtime.detect()
    _line(f"harness          : {status.harness or 'MISSING'}")
    _line(f"neural dll       : {status.neural_dll or 'not found'}")
    _line(f"dlss dll         : {status.dlss_dll or 'not found'}")
    _line(f"addon            : {status.addon or 'not found'}")
    _line(f"reshade          : {status.reshade or 'not found'}")
    for problem in status.problems:
        _line(f"  ! {problem}")

    if status.harness is not None and status.ready:
        try:
            runtime.stage_runtime(status)
            _line("")
            _line(evaluator.probe(status.harness))
        except Exception as error:  # noqa: BLE001
            _line(f"probe            : FAILED - {error}")
            failures += 1

        # A real conversion, end to end. The probe only proves the harness
        # starts; this drives the whole FRAME/WRITE protocol, the plane layout
        # and the readback, which is what a user's first click actually does.
        # Two passes because one silently skips the neural pass entirely.
        try:
            import numpy as np

            from . import pipeline
            from .settings import AppSettings

            settings = AppSettings()
            settings.evaluation.frames = 2
            settings.evaluation.max_edge = 512

            scratch = paths.scratch_dir()
            sample = scratch / "selftest_input.png"
            rng = np.random.default_rng(0)
            pipeline.save_image(rng.random((256, 256, 3)).astype("float32"), sample)

            engine = DepthEngine()
            result = pipeline.convert(sample, settings, engine)
            _line(
                f"conversion       : ok {result.enhanced.shape[1]}x"
                f"{result.enhanced.shape[0]} ({result.notes})"
            )

            # The same again in HDR. The codec check above only proves the
            # decoder works; this proves the range makes it all the way through
            # DLSS and back, which is the part with a clip in every stage.
            try:
                from . import wic

                if wic.available():
                    hdr_sample = scratch / "selftest_input.jxr"
                    plate = np.full((256, 256, 3), 0.25, "float32")
                    plate[:, 128:, 0] = 8.0
                    wic.write(hdr_sample, plate)

                    hdr_result = pipeline.convert(hdr_sample, settings, engine)
                    if hdr_result.enhanced_linear is None:
                        raise RuntimeError("the result came back SDR")
                    peak = float(hdr_result.enhanced_linear.max())
                    # The invariant is that the range survived, not what the
                    # neural pass did with it. An SDR round trip clips to 1.0,
                    # so anything comfortably above that proves the point. How
                    # far above is the model's business and moves with the
                    # driver and the card: an earlier version of this line
                    # expected "about 8" from an input peak of 8, and the same
                    # plate on the same machine later measured 4.59. Asserting
                    # the model's output on a synthetic plate is how you build
                    # a self test that cries wolf on someone else's GPU.
                    ok = peak > 1.5
                    _line(
                        f"hdr conversion   : {'ok' if ok else 'FAILED'} "
                        f"(peak {peak:.2f}x diffuse white; SDR would clip to 1.00)"
                    )
                    if not ok:
                        failures += 1
            except Exception as error:  # noqa: BLE001
                _line(f"hdr conversion   : FAILED - {type(error).__name__}: {error}")
                failures += 1
        except Exception as error:  # noqa: BLE001
            _line(f"conversion       : FAILED - {type(error).__name__}: {error}")
            failures += 1

    # --- what the add-on itself said -------------------------------------
    #
    # The probe can only report whether nvngx_dlssnr.dll ended up in the
    # process. When it did not, the reason is always in the add-on's own log and
    # never in ours: it refuses runtimes it does not recognise, and says so.
    # Reprinting its lines here saves a round trip asking for the file.
    if status.harness is not None:
        engine = status.harness.parent
        staged = sorted(
            p.name for p in engine.glob("*") if p.suffix.lower() in {".dll", ".addon64"}
        )
        _line("")
        _line(f"staged beside the harness: {', '.join(staged) or 'nothing'}")

        log = engine / "ReShade.log"
        if log.is_file():
            try:
                lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                lines = []
            addon = [line for line in lines if "DLSS 5 Neural Rendering" in line]
            interesting = [
                line
                for line in addon
                if any(
                    word in line.lower()
                    # "renodx" and "registered" catch the add-on's version
                    # banner. Worth always showing: an outdated add-on is a
                    # confirmed cause of dlssnr_module_loaded: 0 on newer cards,
                    # and the version is the first thing to check.
                    for word in (
                        "sha256", "runtime", "unavailable", "fail", "error",
                        "reject", "renodx", "registered",
                    )
                )
            ]
            if interesting:
                _line("")
                _line("what the add-on reported:")
                for line in interesting[-8:]:
                    # Strip the timestamp and level, keep the message.
                    _line("  " + line.split("| INFO  |")[-1].split("| WARN  |")[-1]
                          .split("| ERROR |")[-1].strip()[:200])
        else:
            _line("(no ReShade.log yet - run a conversion first)")

    _line("=" * 46)
    _line("PASS" if failures == 0 else f"{failures} problem(s)")
    return 1 if failures else 0
