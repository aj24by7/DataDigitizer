from __future__ import annotations

from digitizer_2_11 import configure_runtime_paths


def main() -> int:
    configure_runtime_paths()
    from AccuracyTesterPro import main as accuracy_tester_main

    accuracy_tester_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
