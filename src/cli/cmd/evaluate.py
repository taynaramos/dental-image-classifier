import argparse
from pathlib import Path
from typing import ClassVar

from src.data.loader import DataLoader


class Evaluate:
    name: ClassVar[str] = "evaluate"
    help: ClassVar[str] = "Evaluate a trained model on a test set."

    def __init__(self, subparsers: argparse._SubParsersAction) -> None:
        parser = subparsers.add_parser(self.name, help=self.help)
        parser.add_argument(
            "--train-dir",
            type=Path,
            default=Path("data/training"),
            help="Folder with the train dataset (default: data/training).",
        )
        parser.add_argument(
            "--test-dir",
            type=Path,
            default=Path("data/test"),
            help="Folder with the test dataset (default: data/test).",
        )

    def run(self, args: argparse.Namespace) -> None:
        loader = DataLoader()

        print("Training dataset:")
        for path in sorted(args.train_dir.rglob("*")):
            if path.is_file():
                content = loader.load(path)
                print(f"{path.name}: {len(content)} bytes")

        print("\nTest dataset:")
        for path in sorted(args.test_dir.rglob("*")):
            if path.is_file():
                content = loader.load(path)
                print(f"{path.name}: {len(content)} bytes")
