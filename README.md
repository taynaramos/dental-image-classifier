# Dental Image Classifier

Classifica imagens intraorais odontológicas em 5 vistas — **frontal**, **superior**, **inferior**, **lateral direita** e **lateral esquerda** — usando uma **CNN em PyTorch**, treinada do zero (sem pesos pré-treinados nem transfer learning).

Para treinar com GPU no Google Colab, use o notebook [`experiments/colab_training.ipynb`](experiments/colab_training.ipynb): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/taynaramos/dental-image-classifier/blob/pytorch-kfold/experiments/colab_training.ipynb) — ele clona o repositório, copia o dataset do seu Google Drive e salva o checkpoint de volta no Drive.

---

## Sumário

1. [Arquitetura e fluxo](#1-arquitetura-e-fluxo)
2. [Estrutura do projeto](#2-estrutura-do-projeto)
3. [Módulos da solução](#3-módulos-da-solução)
4. [Instalação](#4-instalação)
5. [Uso via CLI](#5-uso-via-cli)
6. [Validando o projeto](#6-validando-o-projeto)
7. [Persistência e inferência posterior](#7-persistência-e-inferência-posterior)

---

## 1. Arquitetura e fluxo

```text
Imagens (uma pasta por sujeito)
→ preparo automático em layout train/val/test/<classe> (ImageFolder)
→ pré-processamento (Resize, Grayscale opcional, ToTensor, Normalize)
→ CNN (Conv2d → ReLU → MaxPool) x2 → Flatten → Linear → ReLU → Dropout → Linear
→ treinamento (CrossEntropyLoss + Adam)
→ avaliação (loss/acurácia de treino, validação e teste)
→ checkpoint salvo com torch.save (pesos + classes + configurações)
```

O dataset real (`data/dataset/`) tem uma sub-pasta por sujeito (paciente), cada uma com 5 imagens nomeadas por vista (`intraoral-frontal.jpeg`, etc.) — não o layout `classe/imagem` que o `ImageFolder` exige. Por isso, antes do treino, `dataset.py` **materializa automaticamente** esse dataset em `data/dataset_imagefolder/{train,val,test}/<classe>/`, dividindo por **sujeito** (não por imagem), para que fotos do mesmo paciente nunca apareçam em treino e teste ao mesmo tempo. Essa preparação roda uma única vez — execuções seguintes reaproveitam a pasta já gerada. Se o caminho passado em `--dataset-path` já tiver as sub-pastas `train`/`val`/`test`, ele é usado diretamente, sem nenhuma cópia.

A CNN é implementada manualmente e usa `AdaptiveAvgPool2d` antes do classificador, o que permite treinar com qualquer `--image-size` sem alterar o código.

---

## 2. Estrutura do projeto

```text
src/
├── cli/
│   ├── cli.py                  # registra os comandos e o entry-point (python -m src.cli.cli)
│   └── cmd/
│       ├── torch_train.py      # comando torch-train
│       └── torch_predict.py    # comando torch-predict
│
└── pytorch_classifier/
    ├── dataset.py
    ├── model.py
    ├── trainer.py
    ├── predict.py
    └── utils.py

experiments/
├── pytorch_training.ipynb      # treinamento passo a passo (mesma lógica da CLI)
└── colab_training.ipynb        # mesma lógica, pronto para GPU no Google Colab

data/
├── dataset/                    # dataset bruto (uma pasta por sujeito) — não versionado
└── dataset_imagefolder/        # layout train/val/test gerado automaticamente — não versionado

artifacts/
└── torch_model.pth             # checkpoint gerado pelo treino — não versionado

requirements.txt                # dependências do projeto
```

> `data/dataset_imagefolder/` e `artifacts/` são gerados a partir do dataset bruto e do código-fonte, por isso ficam de fora do controle de versão (`.gitignore`) — cada execução do treino os recria.

---

## 3. Módulos da solução

### `dataset.py`

```text
src/pytorch_classifier/dataset.py
```

* `resolve_imagefolder_root` — garante o layout `train/val/test/<classe>` (materializa a partir do dataset por sujeito, se necessário).
* `build_transform` — monta o pipeline `Resize → Grayscale? → ToTensor → Normalize`.
* `build_dataloaders` — cria os `ImageFolder` + `DataLoader` de treino, validação e teste e retorna a lista de classes (obtida automaticamente do dataset).

### `model.py`

```text
src/pytorch_classifier/model.py
```

* `DentalCNN` — a CNN: `(Conv2d → ReLU → MaxPool) x2 → AdaptiveAvgPool → Flatten → Linear → ReLU → Dropout → Linear`.
* `ModelConfig` — hiperparâmetros necessários para reconstruir o modelo e repetir o pré-processamento na inferência (`image_size`, `grayscale`, `hidden_dim`, `dropout`).

### `trainer.py`

```text
src/pytorch_classifier/trainer.py
```

* `Trainer` — treina (`fit`) e avalia (`evaluate`) o modelo com `CrossEntropyLoss` + `Adam`, calculando loss e acurácia e imprimindo as métricas a cada época. Suporta early stopping (`patience`/`min_delta`) e restaura os pesos da melhor época ao final.
* `History` — loss/acurácia de treino e validação, uma entrada por época (usado para os gráficos).

### `predict.py`

```text
src/pytorch_classifier/predict.py
```

* `predict(image_path, model, classes, config, device)` — classifica uma imagem e retorna a classe prevista e as probabilidades por classe (softmax).

### `utils.py`

```text
src/pytorch_classifier/utils.py
```

* `get_device` / `set_seed` — dispositivo (CPU/GPU) e reprodutibilidade.
* `save_checkpoint` / `load_checkpoint` — salvam e restauram `model_state_dict` + `classes` + `config` via `torch.save`/`torch.load`.

### CLI

```text
src/cli/cli.py
```

Disponibiliza os comandos:

```text
torch-train
torch-predict
```

---

## 4. Instalação

### Criar o ambiente virtual

Linux ou macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

### Instalar as dependências

```bash
pip install -r requirements.txt
```

---

## 5. Uso via CLI

### Treinar o modelo

```bash
python -m src.cli.cli torch-train \
    --dataset-path data/dataset \
    --epochs 20 \
    --batch-size 32 \
    --learning-rate 0.001 \
    --image-size 128
```

Ao final de cada época, o comando imprime `Train Loss`, `Train Accuracy`, `Validation Loss` e `Validation Accuracy`; ao término do treino, `Test Loss` e `Test Accuracy` no conjunto de teste, e então salva o checkpoint em `--model-out` (padrão: `artifacts/torch_model.pth`).

Outras opções úteis: `--grayscale` / `--no-grayscale` (padrão: escala de cinza), `--patience`/`--min-delta` (early stopping), `--hidden-dim`, `--dropout`, `--train-ratio`/`--val-ratio`/`--test-ratio`, `--num-workers`, `--seed`, `--model-out`.

### Classificar uma imagem

```bash
python -m src.cli.cli torch-predict \
    --model artifacts/torch_model.pth \
    --image caminho/imagem.jpg
```

### Classificar uma pasta de imagens

```bash
python -m src.cli.cli torch-predict \
    --model artifacts/torch_model.pth \
    --image-dir caminho/para/pasta
```

`torch-predict` imprime a classe prevista e a probabilidade de cada uma das 5 vistas para cada imagem.

> `python -m src.cli.cli --help` lista os comandos disponíveis.

---

## 6. Validando o projeto

O projeto pode ser validado de **duas formas equivalentes**, já que ambas executam exatamente o mesmo código — o notebook não duplica lógica, ele importa diretamente de `src/pytorch_classifier`:

### Opção A — CLI

```bash
python -m src.cli.cli torch-train --dataset-path data/dataset
python -m src.cli.cli torch-predict --model artifacts/torch_model.pth --image caminho/imagem.jpg
```

Mais rápido para reproduzir o pipeline de ponta a ponta ou integrar em scripts/automação.

### Opção B — Notebook

```text
experiments/pytorch_training.ipynb
```

Executa o mesmo pipeline célula a célula, permitindo inspecionar cada etapa (dataset, DataLoaders, arquitetura, curvas de treino, matriz de confusão, predição individual) antes de seguir para a próxima. Ideal para análise exploratória e para revisar visualmente as métricas.

Para rodar, abra o notebook a partir da pasta `experiments/` (ele resolve o `PROJECT_ROOT` a partir de `Path.cwd().parent`) com o kernel do `.venv` do projeto, e execute as células na ordem:

1. Importações
2. Configuração dos caminhos
3. Carregamento do dataset
4. Visualização de algumas imagens
5. Criação do DataLoader
6. Construção da CNN
7. Configuração do treinamento
8. Loop de treinamento
9. Gráfico da loss
10. Gráfico da acurácia
11. Avaliação final (teste)
12. Matriz de confusão (`sklearn.metrics`)
13. Predição em imagens individuais
14. Salvamento do modelo

Como as duas formas compartilham a mesma implementação, os resultados (loss, acurácia, matriz de confusão) devem ser idênticos entre elas para os mesmos parâmetros e seed.

---

## 7. Persistência e inferência posterior

`save_checkpoint` grava um único arquivo `.pth` com:

* `model_state_dict` — pesos da rede;
* `classes` — lista de rótulos, na ordem usada pelos índices de saída;
* `config` — `image_size`, `grayscale`, `hidden_dim` e `dropout`, necessários para reconstruir a `DentalCNN` e repetir exatamente o mesmo pré-processamento usado no treino.

`load_checkpoint(path)` reconstrói o modelo, carrega os pesos e o deixa em modo de avaliação (`eval()`), pronto para chamadas a `predict(...)` — usado tanto pelo `torch-predict` quanto pelo notebook.
