import argparse
from typing import ClassVar


class Predict:
    name: ClassVar[str] = "predict"
    help: ClassVar[str] = "Classify a single intraoral image."

    def __init__(self, subparsers: argparse._SubParsersAction) -> None:
        subparsers.add_parser(self.name, help=self.help)

    def run(self, args: argparse.Namespace) -> None:
        print("predict: not implemented yet")
