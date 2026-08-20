"""Pacote de testes unitários do projeto.

Ponto único para os testes deste repositório — qualquer teste novo deve ser
adicionado como um módulo ``test_*.py`` dentro de ``tests/`` e importado a
partir daqui (``from tests.test_meu_arquivo import ...`` ou
``from tests import test_meu_arquivo``), reaproveitando as fábricas de dados
sintéticos de :mod:`tests.factories` (imagens e datasets fake) em vez de
duplicá-las.

Como rodar:

.. code-block:: bash

    python -m unittest discover -s tests -t .

ou, se ``pytest`` estiver instalado:

.. code-block:: bash

    pytest tests

Alguns testes dependem de PyTorch/torchvision (módulos ``pytorch_kfold`` e
``pytorch_resnet18_transfer``) ou de scikit-learn (módulo ``pca_svc``) — eles
se pulam automaticamente (``TORCH_AVAILABLE``/``SKLEARN_AVAILABLE`` em
:mod:`tests.factories`) se a dependência não estiver instalada, em vez de
quebrar a coleta do restante do pacote. Instale os ``requirements-*.txt`` dos
módulos que quiser testar (ver README.md na raiz do projeto).

Este ``__init__.py`` propositalmente não importa os módulos ``test_*`` (nem
``factories``) de forma antecipada: ``factories.py`` já depende de Pillow, e
alguns dos ``test_*`` dependem de PyTorch/scikit-learn — importar qualquer um
deles aqui obrigaria até um teste sem essas dependências (ex.:
``test_predict_cli_extensions.py``, que só usa a biblioteca padrão) a falhar
ao ser rodado isoladamente. Cada módulo importa exatamente o que precisa,
quando executado, preservando a possibilidade de instalar (e testar) cada
módulo do projeto de forma isolada.
"""

from __future__ import annotations
