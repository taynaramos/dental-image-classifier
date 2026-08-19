import argparse
from typing import ClassVar, Protocol

from .cmd.kfold_predict import KfoldPredict
from .cmd.kfold_train import KfoldTrain
from .cmd.pca_predict import PcaPredict
from .cmd.pca_train import PcaTrain
from .cmd.resnet18_predict import Resnet18Predict
from .cmd.resnet18_train import Resnet18Train


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
        "usando um dos módulos de classificação disponíveis "
        "(PCA + SVC, CNN em PyTorch treinada do zero, ou transfer learning com ResNet-18)."
    )
    commands: list[type[Command]] = [
        PcaTrain,
        PcaPredict,
        KfoldTrain,
        KfoldPredict,
        Resnet18Train,
        Resnet18Predict,
    ]

    def run(self, argv: list[str] | None = None) -> None:
        parser = argparse.ArgumentParser(prog=self.name, description=self.description)
        subparsers = parser.add_subparsers(dest="command", required=True)
        commands = {cls.name: cls(subparsers) for cls in self.commands}
        args = parser.parse_args(argv)
        commands[args.command].run(args)
