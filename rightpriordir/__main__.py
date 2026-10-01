"""Lightweight command dispatcher; frozen plots do not import registration."""

import argparse
import importlib
import sys


def main():
    commands = {
        "frozen": "frozen",
        "prepare-oasis": "prepare_oasis",
        "prepare-lung": "prepare_lung",
        "masks": "masks",
        "register": "register",
        "evaluate": "evaluate",
        "verify": "verify",
        "experiment": "experiment",
        "summarize": "summarize",
    }
    p = argparse.ArgumentParser(
        description="RightPriorDIR public release; frozen reproduction never runs optimization."
    )
    p.add_argument("command", choices=commands)
    args = p.parse_args(sys.argv[1:2])
    rest = sys.argv[2:]
    importlib.import_module("." + commands[args.command], __package__).main(rest)


if __name__ == "__main__":
    main()
