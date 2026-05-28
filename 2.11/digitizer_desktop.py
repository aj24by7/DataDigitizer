from __future__ import annotations

from digitizer_2_11 import configure_runtime_paths, launch_digitizer_gui


if __name__ == "__main__":
    configure_runtime_paths()
    raise SystemExit(launch_digitizer_gui())
