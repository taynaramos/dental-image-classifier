# Dental Image Classifier

Classifica imagens intraorais odontológicas em 5 vistas — **frontal**, **superior**, **inferior**, **lateral direita** e **lateral esquerda**.

O projeto reúne três soluções independentes para o mesmo problema, cada uma como um módulo próprio dentro de `src/`, com sua própria documentação e suas próprias dependências:

| Módulo | Abordagem | Documentação | Dependências |
| --- | --- | --- | --- |
| [`src/pca_svc/`](src/pca_svc/README.md) | PCA + SVC (numpy/scikit-learn, sem PyTorch) | [src/pca_svc/README.md](src/pca_svc/README.md) | [requirements-pca-svc.txt](requirements-pca-svc.txt) |
| [`src/pytorch_kfold/`](src/pytorch_kfold/README.md) | CNN em PyTorch, treinada do zero, com validação cruzada k-fold por sujeito | [src/pytorch_kfold/README.md](src/pytorch_kfold/README.md) | [requirements-pytorch-kfold.txt](requirements-pytorch-kfold.txt) |
| [`src/pytorch_resnet18_transfer/`](src/pytorch_resnet18_transfer/README.md) | Transfer learning: extrator ResNet-18 pré-treinado (ImageNet) + classificador linear | [src/pytorch_resnet18_transfer/README.md](src/pytorch_resnet18_transfer/README.md) | [requirements-pytorch-resnet18-transfer.txt](requirements-pytorch-resnet18-transfer.txt) |

Cada módulo pode ser instalado e usado isoladamente — instalar apenas `requirements-pca-svc.txt`, por exemplo, não requer PyTorch.

---

## Estrutura do projeto

```text
main.py                          # ponto de entrada da CLI

src/
├── cli/
│   ├── cli.py                   # registra os comandos de todos os módulos
│   ├── train/                   # um comando de treino por módulo (pca-train, kfold-train, resnet18-train, train)
│   ├── predict/                 # um comando de inferência por módulo (pca-predict, kfold-predict, resnet18-predict, predict)
│   └── tools/                   # utilitários compartilhados pelos comandos de predição (coletar_caminhos, imprimir_predicao)
├── pca_svc/                      # módulo PCA + SVC (ver seu próprio README)
├── pytorch_kfold/                # módulo CNN em PyTorch, treinada do zero (ver seu próprio README)
└── pytorch_resnet18_transfer/    # módulo transfer learning ResNet-18 (ver seu próprio README)

notebooks/                       # notebooks de cada módulo (Colab e local)
requirements-pca-svc.txt
requirements-pytorch-kfold.txt
requirements-pytorch-resnet18-transfer.txt
```

## Uso via CLI

Depois de instalar as dependências do módulo desejado (veja a documentação de cada um), todos os comandos passam pelo mesmo ponto de entrada:

```bash
python main.py --help
```

```text
{train, predict, pca-train, pca-predict, kfold-train, kfold-predict, resnet18-train, resnet18-predict}
```

### `train` / `predict` — comandos unificados

`train` e `predict` funcionam para os três modelos, escolhidos via `--model {pca,kfold,resnet18}`:

```bash
python main.py train --model kfold --dataset-path data/dataset
```

`predict` tem dois modos, conforme o argumento de entrada:

* `--image` — **single_inference**: uma única imagem, retorna a classe prevista e o vetor de probabilidades em JSON.

  ```bash
  python main.py predict --model kfold --checkpoint artifacts/kfold_model.pth --image caminho/imagem.jpg
  ```

  ```json
  {"class": "frontal", "probabilities": {"frontal": 0.92, "inferior": 0.01, "superior": 0.03, "lateral_direita": 0.02, "lateral_esquerda": 0.02}}
  ```

* `--image-dir` — **test_case**: uma pasta com as imagens de um caso (uma por vista). Retorna, para cada vista, o arquivo classificado com aquele rótulo — ou `"Not found"` se nenhuma imagem da pasta bateu com essa vista. Com `--probabilities`, o vetor de probabilidades de cada imagem também é incluído.

  ```bash
  python main.py predict --model kfold --checkpoint artifacts/kfold_model.pth --image-dir caminho/do/caso --probabilities
  ```

  ```json
  {
    "views": {
      "frontal": "img1.jpg",
      "inferior": "Not found",
      "superior": "img3.jpg",
      "lateral_direita": "img4.jpg",
      "lateral_esquerda": "img5.jpg"
    },
    "probabilities": {
      "img1.jpg": {"frontal": 0.92, "inferior": 0.01, "...": "..."}
    }
  }
  ```

Para os modelos em PyTorch (`kfold`, `resnet18`), o checkpoint salvo por `train` também inclui a loss/acurácia de treino e validação por época (`history`), consultável depois via `load_history(path)` de cada módulo, sem precisar re-treinar.

`train`/`predict` só expõem os parâmetros comuns aos três modelos. Para controle fino de hiperparâmetros específicos, use diretamente `pca-train`/`kfold-train`/`resnet18-train` e seus `*-predict` — consulte o README de cada módulo.
