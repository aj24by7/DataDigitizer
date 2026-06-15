from __future__ import annotations

import sys

from digitizer_2_11 import configure_runtime_paths, main


if __name__ == "__main__":
    configure_runtime_paths()
    args = sys.argv[1:]
    if len(sys.argv) == 1:
        from digitizer_cli import print_function_call_usage

        print_function_call_usage()
        raise SystemExit(0)
    if len(args) == 1 or args[0].startswith("digitizer_cli"):
        from digitizer_cli import is_function_call_syntax, main as cli_main

        if is_function_call_syntax(args):
            raise SystemExit(cli_main(args))
    raise SystemExit(main(args))
