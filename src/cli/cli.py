import argparse
from typing import ClassVar, Protocol

from .cmd.evaluate import Evaluate
from .cmd.predict import Predict
from .cmd.train import Train


class Command(Protocol):
    name: ClassVar[str]
    help: ClassVar[str]

    def __init__(self, subparsers: argparse._SubParsersAction) -> None: ...

    def run(self, args: argparse.Namespace) -> None: ...


class CLI:
    name = "dental-classifier"
    description = (
        "Classify dental intraoral images into 5 views "
        "(frontal, superior, inferior, lateral direita, lateral esquerda)."
    )
    commands: list[type[Command]] = [
        Evaluate,
        Train,
        Predict,
    ]

    def run(self, argv: list[str] | None = None) -> None:
        parser = argparse.ArgumentParser(prog=self.name, description=self.description)
        subparsers = parser.add_subparsers(dest="command", required=True)
        commands = {cls.name: cls(subparsers) for cls in self.commands}
        args = parser.parse_args(argv)
        commands[args.command].run(args)
