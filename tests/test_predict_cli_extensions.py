"""Filtro de extensão em ``--image-dir`` dos comandos de inferência (CLI).

``coletar_caminhos`` (em ``src/cli/tools/io.py``) é usada por todos os
quatro comandos de predição (``src/cli/predict/predict.py``,
``resnet18_predict.py``, ``kfold_predict.py``, ``pca_predict.py``) — aceita
``.jpeg``/``.jpg`` (case-insensitive) ao escanear uma pasta com
``--image-dir``, e ignora qualquer outra extensão.

Note que essa lista é mais permissiva que o carregamento do dataset de
**treino**, que só aceita ``.jpeg`` — sem ``.jpg`` (ver
``test_image_format.py``): um arquivo ``.jpg`` solto na pasta bruta de um
sujeito seria ignorado no treino, mas aceito aqui na inferência.

``src/cli/tools/io.py`` não importa PyTorch/scikit-learn no topo do arquivo
(só ``pathlib``/``typing``), e o mesmo vale para os quatro módulos de comando
(ver a explicação do ``TYPE_CHECKING`` em ``resnet18_predict.py``) — então
este teste não depende de nenhum ``requirements-*.txt`` além do próprio
Python padrão.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.cli.predict import kfold_predict, pca_predict, predict, resnet18_predict
from src.cli.tools import io as _shared
from src.cli.tools.io import coletar_caminhos

# Os quatro comandos de predição reexportam coletar_caminhos importando-a de
# src.cli.tools.io — confirma que todos apontam para a mesma função, não cópias.
_MODULOS_COM_COLETAR_CAMINHOS = {
    "predict (unificado)": predict,
    "resnet18_predict": resnet18_predict,
    "kfold_predict": kfold_predict,
    "pca_predict": pca_predict,
}


class SharedModuleWiringTests(unittest.TestCase):
    """Cada comando de predição usa a mesma função compartilhada, não uma cópia própria."""

    def test_all_predict_commands_reference_the_shared_function(self) -> None:
        for nome_modulo, modulo in _MODULOS_COM_COLETAR_CAMINHOS.items():
            with self.subTest(modulo=nome_modulo):
                self.assertIs(modulo.coletar_caminhos, _shared.coletar_caminhos)


class ImageDirExtensionFilterTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def _tocar_arquivos(self, *nomes: str) -> None:
        for nome in nomes:
            (self.root / nome).touch()

    def test_jpeg_and_jpg_are_both_accepted_case_insensitively(self) -> None:
        self._tocar_arquivos("a.jpeg", "b.jpg", "c.JPEG", "d.JPG", "e.png", "f.txt")
        caminhos = coletar_caminhos(None, self.root)
        nomes_encontrados = {p.name for p in caminhos}
        self.assertEqual(nomes_encontrados, {"a.jpeg", "b.jpg", "c.JPEG", "d.JPG"})

    def test_other_extensions_are_ignored(self) -> None:
        self._tocar_arquivos("nota.png", "leiame.txt", "sem_extensao")
        self.assertEqual(coletar_caminhos(None, self.root), [])

    def test_empty_directory_returns_empty_list(self) -> None:
        self.assertEqual(coletar_caminhos(None, self.root), [])

    def test_single_image_mode_bypasses_extension_filter(self) -> None:
        # Quando --image é passado diretamente (não --image-dir), o caminho é
        # usado como está — sem checar extensão, mesmo que não seja .jpeg/.jpg.
        caminho_qualquer = self.root / "raio-x.dcm"
        caminho_qualquer.touch()
        self.assertEqual(coletar_caminhos(caminho_qualquer, None), [caminho_qualquer])


if __name__ == "__main__":
    unittest.main()
