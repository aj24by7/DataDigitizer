from __future__ import annotations

import sys

from digitizer_2_11 import configure_runtime_paths, main


if __name__ == "__main__":
    configure_runtime_paths()
    if len(sys.argv) == 1:
        from digitizer_cli import interactive_main

        raise SystemExit(interactive_main())
    raise SystemExit(main(sys.argv[1:]))
