import argparse
from typing import ClassVar, Protocol

from .cmd.torch_predict import TorchPredict
from .cmd.torch_train import TorchTrain


class Command(Protocol):
    name: ClassVar[str]
    help: ClassVar[str]

    def __init__(self, subparsers: argparse._SubParsersAction) -> None: ...

    def run(self, args: argparse.Namespace) -> None: ...


class CLI:
    name = "dental-classifier"
    description = (
        "Classifica imagens intraorais odontológicas em 5 vistas "
        "(frontal, superior, inferior, lateral direita, lateral esquerda) "
        "usando uma CNN em PyTorch."
    )
    commands: list[type[Command]] = [
        TorchTrain,
        TorchPredict,
    ]

    def run(self, argv: list[str] | None = None) -> None:
        parser = argparse.ArgumentParser(prog=self.name, description=self.description)
        subparsers = parser.add_subparsers(dest="command", required=True)
        commands = {cls.name: cls(subparsers) for cls in self.commands}
        args = parser.parse_args(argv)
        commands[args.command].run(args)


if __name__ == "__main__":
    CLI().run()
