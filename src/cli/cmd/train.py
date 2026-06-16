import argparse
from pathlib import Path
from typing import ClassVar

from src.data.loader import DataLoader


class Train:
    name: ClassVar[str] = "train"
    help: ClassVar[str] = "Train the dental view classifier."

    def __init__(self, subparsers: argparse._SubParsersAction) -> None:
        parser = subparsers.add_parser(self.name, help=self.help)
        parser.add_argument(
            "--train-dir",
            type=Path,
            default=Path("data/training"),
            help="Folder with the training dataset (default: data/training).",
        )

    def run(self, args: argparse.Namespace) -> None:
        loader = DataLoader()

        print("Training dataset:")
        for path in sorted(args.train_dir.rglob("*")):
            if path.is_file():
                content = loader.load(path)
                print(f"{path.name}: {len(content)} bytes")
