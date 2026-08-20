# Módulo PyTorch CNN (k-fold)

Classifica imagens intraorais odontológicas em 5 vistas — **frontal**, **superior**, **inferior**, **lateral direita** e **lateral esquerda** — usando uma **CNN em PyTorch** (`src/pytorch_kfold/`), treinada do zero (sem pesos pré-treinados nem transfer learning).

> **Sobre o nome do módulo:** `kfold-train`/`kfold-predict` (CLI) e `Trainer.fit` treinam com um único split treino/val/teste (70/15/15 por sujeito) — igual aos outros dois módulos. A **validação cruzada k-fold de verdade** (5 folds por sujeito, um modelo do zero por fold, média ± desvio padrão da acurácia) existe apenas como células manuais na seção 12 do notebook Colab (ver [seção 8](#8-validação-cruzada-k-fold-de-verdade) abaixo) — não é acionável pela CLI.

Para treinar com GPU no Google Colab, use o notebook [`notebooks/pytorch_kfold_colab.ipynb`](../../notebooks/pytorch_kfold_colab.ipynb): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/taynaramos/dental-image-classifier/blob/main/notebooks/pytorch_kfold_colab.ipynb) — ele clona o repositório, copia o dataset do seu Google Drive e salva o checkpoint de volta no Drive.

---

## Sumário

1. [Arquitetura e fluxo](#1-arquitetura-e-fluxo)
2. [Estrutura do projeto](#2-estrutura-do-projeto)
3. [Módulos da solução](#3-módulos-da-solução)
4. [Instalação](#4-instalação)
5. [Uso via CLI](#5-uso-via-cli)
6. [Validando o projeto](#6-validando-o-projeto)
7. [Persistência e inferência posterior](#7-persistência-e-inferência-posterior)
8. [Validação cruzada k-fold de verdade](#8-validação-cruzada-k-fold-de-verdade)

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
│   ├── cli.py                       # registra os comandos de todos os módulos (entry-point: python main.py)
│   ├── train/
│   │   └── kfold_train.py           # comando kfold-train
│   ├── predict/
│   │   └── kfold_predict.py         # comando kfold-predict
│   └── tools/
│       └── io.py                    # coletar_caminhos/imprimir_predicao, compartilhados entre os comandos *-predict
│
└── pytorch_kfold/
    ├── dataset.py
    ├── model.py
    ├── trainer.py
    ├── predict.py
    └── utils.py

notebooks/
├── pytorch_kfold_training.ipynb   # treinamento passo a passo (mesma lógica da CLI)
└── pytorch_kfold_colab.ipynb      # mesma lógica, pronto para GPU no Google Colab

data/
├── dataset/                    # dataset bruto (uma pasta por sujeito) — não versionado
└── dataset_imagefolder/        # layout train/val/test gerado automaticamente — não versionado

artifacts/
└── kfold_model.pth             # checkpoint gerado pelo treino — não versionado

requirements-pytorch-kfold.txt  # dependências deste módulo (na raiz do projeto)
```

> `data/dataset_imagefolder/` e `artifacts/` são gerados a partir do dataset bruto e do código-fonte, por isso ficam de fora do controle de versão (`.gitignore`) — cada execução do treino os recria.

---

## 3. Módulos da solução

### `dataset.py`

```text
src/pytorch_kfold/dataset.py
```

* `resolve_imagefolder_root` — garante o layout `train/val/test/<classe>` (materializa a partir do dataset por sujeito, se necessário).
* `build_transform` — monta o pipeline `Resize → Grayscale? → ToTensor → Normalize`.
* `build_dataloaders` — cria os `ImageFolder` + `DataLoader` de treino, validação e teste e retorna a lista de classes (obtida automaticamente do dataset).

### `model.py`

```text
src/pytorch_kfold/model.py
```

* `DentalCNN` — a CNN: `(Conv2d → ReLU → MaxPool) x2 → AdaptiveAvgPool → Flatten → Linear → ReLU → Dropout → Linear`.
* `ModelConfig` — hiperparâmetros necessários para reconstruir o modelo e repetir o pré-processamento na inferência (`image_size`, `grayscale`, `hidden_dim`, `dropout`).

### `trainer.py`

```text
src/pytorch_kfold/trainer.py
```

* `Trainer` — treina (`fit`) e avalia (`evaluate`) o modelo com `CrossEntropyLoss` + `Adam`, calculando loss e acurácia e imprimindo as métricas a cada época. Suporta early stopping (`patience`/`min_delta`) e restaura os pesos da melhor época ao final.
* `History` — loss/acurácia de treino e validação, uma entrada por época (usado para os gráficos).

### `predict.py`

```text
src/pytorch_kfold/predict.py
```

* `predict(image_path, model, classes, config, device)` — classifica uma imagem e retorna a classe prevista e as probabilidades por classe (softmax).

### `utils.py`

```text
src/pytorch_kfold/utils.py
```

* `get_device` / `set_seed` — dispositivo (CPU/GPU) e reprodutibilidade.
* `save_checkpoint` / `load_checkpoint` — salvam e restauram `model_state_dict` + `classes` + `config` via `torch.save`/`torch.load`.

### CLI

```text
src/cli/cli.py
```

Disponibiliza os comandos:

```text
kfold-train
kfold-predict
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
pip install -r requirements-pytorch-kfold.txt
```

(a partir da raiz do projeto)

---

## 5. Uso via CLI

### Treinar o modelo

```bash
python main.py kfold-train \
    --dataset-path data/dataset \
    --epochs 20 \
    --batch-size 32 \
    --learning-rate 0.001 \
    --image-size 128
```

Ao final de cada época, o comando imprime `Train Loss`, `Train Accuracy`, `Validation Loss` e `Validation Accuracy`; ao término do treino, `Test Loss` e `Test Accuracy` no conjunto de teste, e então salva o checkpoint em `--model-out` (padrão: `artifacts/kfold_model.pth`).

Outras opções úteis: `--grayscale` / `--no-grayscale` (padrão: escala de cinza), `--patience`/`--min-delta` (early stopping), `--hidden-dim`, `--dropout`, `--train-ratio`/`--val-ratio`/`--test-ratio`, `--num-workers`, `--seed`, `--model-out`.

### Classificar uma imagem

```bash
python main.py kfold-predict \
    --model artifacts/kfold_model.pth \
    --image caminho/imagem.jpg
```

### Classificar uma pasta de imagens

```bash
python main.py kfold-predict \
    --model artifacts/kfold_model.pth \
    --image-dir caminho/para/pasta
```

`kfold-predict` imprime a classe prevista e a probabilidade de cada uma das 5 vistas para cada imagem.

> `python main.py --help` lista os comandos disponíveis.

---

## 6. Validando o projeto

O projeto pode ser validado de **duas formas equivalentes**, já que ambas executam exatamente o mesmo código — o notebook não duplica lógica, ele importa diretamente de `src/pytorch_kfold`:

### Opção A — CLI

```bash
python main.py kfold-train --dataset-path data/dataset
python main.py kfold-predict --model artifacts/kfold_model.pth --image caminho/imagem.jpg
```

Mais rápido para reproduzir o pipeline de ponta a ponta ou integrar em scripts/automação.

### Opção B — Notebook

```text
notebooks/pytorch_kfold_training.ipynb
```

Executa o mesmo pipeline célula a célula, permitindo inspecionar cada etapa (dataset, DataLoaders, arquitetura, curvas de treino, matriz de confusão, predição individual) antes de seguir para a próxima. Ideal para análise exploratória e para revisar visualmente as métricas.

Para rodar, abra o notebook a partir da pasta `notebooks/` (ele resolve o `PROJECT_ROOT` a partir de `Path.cwd().parent`) com o kernel do `.venv` do projeto, e execute as células na ordem:

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
* `history` (opcional) — loss/acurácia de treino e validação por época (ver `History` em `trainer.py`), para consultar a curva de treino depois sem re-treinar.

`load_checkpoint(path)` reconstrói o modelo, carrega os pesos e o deixa em modo de avaliação (`eval()`), pronto para chamadas a `predict(...)` — usado tanto pelo `kfold-predict` quanto pelo notebook. `load_history(path)` lê só o `history` salvo (`None` se o checkpoint foi salvo sem ele).

---

## 8. Validação cruzada k-fold de verdade

O que está descrito nas seções anteriores (CLI e notebook de treino) é um **único split** treino/val/teste — o mesmo esquema usado pelos módulos `pca_svc` e `pytorch_resnet18_transfer`. Isso significa que a acurácia de teste é medida sobre só ~45 sujeitos (15% de 300), uma amostra pequena para estimar o desempenho com confiança.

Para uma estimativa mais robusta, a **seção 12** do notebook [`notebooks/pytorch_kfold_colab.ipynb`](../../notebooks/pytorch_kfold_colab.ipynb) implementa validação cruzada k-fold de verdade, à mão (sem `sklearn.model_selection`):

* os 300 sujeitos são embaralhados (seed fixa) e cortados em **5 folds de 60**, sem sobreposição;
* em cada rodada, 1 fold é teste e os outros 4 (240 sujeitos) são treino;
* um **modelo novo (`DentalCNN`) é treinado do zero a cada fold** — nenhum peso é reaproveitado entre rodadas;
* o esquema é **treino/teste clássico, sem validação**: cada fold treina por um número **fixo** de épocas (12, escolhido a partir do early stopping da seção 7), e o fold de teste é avaliado uma única vez ao final — nunca influencia decisões de treinamento;
* o resultado final é a **média ± desvio padrão** da acurácia de teste entre os 5 folds.

Essa validação cruzada serve só para **avaliar a robustez do pipeline** — ela não produz um checkpoint utilizável; o modelo entregável continua sendo o da seção 7 (`kfold-train`/seções 1–11 do notebook). Não há comando de CLI equivalente; para rodar, execute o notebook no Colab (ou localmente) até a seção 7 (as células da seção 12 reaproveitam `DATASET_ROOT`, `IMAGE_SIZE`, `GRAYSCALE`, `BATCH_SIZE`, `SEED` e `device` definidos ali) e depois execute as células da seção 12.
