# Módulo ResNet-18 (transfer learning)

Classifica imagens intraorais odontológicas em 5 vistas — **frontal**, **superior**, **inferior**, **lateral direita** e **lateral esquerda** — por **transfer learning**: um extrator de features **ResNet-18 pré-treinado na ImageNet**, mais um classificador linear treinado sobre essas features.

Diferente do módulo [`pytorch_kfold`](../pytorch_kfold/README.md) (CNN treinada do zero, sem pesos pré-treinados), aqui a extração de features e a classificação são estágios distintos:

```text
Imagens (uma pasta por sujeito)
→ preparo automático em layout train/val/test/<classe> (ImageFolder)
→ pré-processamento (Resize, RandomCrop/CenterCrop, ColorJitter no treino, Normalize com estatísticas da ImageNet)
→ ResNetFeatureExtractor — backbone ResNet-18 pré-treinado, congelado por padrão (extração de features, 512-d)
→ classificador linear sobre as features extraídas
→ Fase 1: treina só o classificador (extrator congelado)
→ Fase 2: libera o extrator e faz fine-tune da rede inteira, com learning rate menor
→ checkpoint salvo com torch.save (pesos + classes + tamanho de imagem)
```

## 1. Por que um extrator de features separado

Um backbone pré-treinado na ImageNet já sabe extrair features visuais genéricas (bordas, texturas, formas). Por isso ele pode ser tratado como um estágio independente — igual ao PCA no módulo [`pca_svc`](../pca_svc/README.md) — em vez de ser aprendido do zero junto com o classificador:

* [`ResNetFeatureExtractor`](feature_extractor.py) (`src/pytorch_resnet18_transfer/feature_extractor.py`) envolve a ResNet-18 pré-treinada, remove sua camada de classificação original e expõe apenas o vetor de features de 512 dimensões. Pode ser usado sozinho (`extractor(x)`) sem passar por nenhum classificador.
* [`DentalResNetTransfer`](model.py) (`src/pytorch_resnet18_transfer/model.py`) combina esse extrator com um `nn.Linear` como classificador — os dois continuam sendo atributos/objetos independentes (`self.extractor`, `self.classifier`), não um bloco só.
* O [`Trainer`](trainer.py) treina em duas fases explícitas: primeiro só o classificador, com o extrator congelado (`extractor.freeze()`); depois libera o extrator inteiro (`extractor.unfreeze()`) para fine-tune com uma taxa de aprendizado menor.

## 2. Estrutura do projeto

```text
src/
├── cli/
│   ├── cli.py                     # registra os comandos de todos os módulos (entry-point: python main.py)
│   └── cmd/
│       ├── resnet18_train.py      # comando resnet18-train
│       └── resnet18_predict.py    # comando resnet18-predict
│
└── pytorch_resnet18_transfer/
    ├── dataset.py                 # dataloader + pré-processamento (transforms com estatísticas da ImageNet)
    ├── feature_extractor.py       # ResNetFeatureExtractor — backbone pré-treinado
    ├── model.py                   # DentalResNetTransfer — extrator + classificador
    ├── trainer.py                 # treino em duas fases (congelado → fine-tune)
    ├── predict.py
    └── utils.py

notebooks/
└── pytorch_resnet18_transfer_colab.ipynb   # mesma lógica, pronto para GPU no Google Colab

data/
├── dataset/                       # dataset bruto (uma pasta por sujeito) — não versionado
└── dataset_imagefolder/           # layout train/val/test gerado automaticamente — não versionado

artifacts/
└── resnet18_transfer_model.pth    # checkpoint gerado pelo treino — não versionado

requirements-pytorch-resnet18-transfer.txt   # dependências deste módulo (na raiz do projeto)
```

## 3. Módulos da solução

### `dataset.py`

```text
src/pytorch_resnet18_transfer/dataset.py
```

* `resolve_imagefolder_root` — garante o layout `train/val/test/<classe>` (materializa a partir do dataset por sujeito, se necessário; mesma lógica usada pelo módulo `pytorch_kfold`).
* `build_train_transform` / `build_eval_transform` — pipelines de pré-processamento com as estatísticas de normalização da ImageNet (`IMAGENET_MEAN`/`IMAGENET_STD`), exigidas pelo backbone pré-treinado. O de treino inclui `RandomCrop` + `ColorJitter` como aumento de dados; o de avaliação usa `CenterCrop`, determinístico.
* `build_dataloaders` — cria os `ImageFolder` + `DataLoader` de treino, validação e teste.

### `feature_extractor.py`

```text
src/pytorch_resnet18_transfer/feature_extractor.py
```

* `ResNetFeatureExtractor` — backbone ResNet-18 pré-treinado na ImageNet, com a camada `fc` original removida (`nn.Identity()`). Expõe `forward(x) -> features` (dimensão 512), `freeze()` e `unfreeze()`.

### `model.py`

```text
src/pytorch_resnet18_transfer/model.py
```

* `DentalResNetTransfer` — combina `ResNetFeatureExtractor` (`self.extractor`) com um `nn.Linear` (`self.classifier`).

### `trainer.py`

```text
src/pytorch_resnet18_transfer/trainer.py
```

* `Trainer` — treina (`fit`) em duas fases (extrator congelado, depois fine-tune) e avalia (`evaluate`). Usa `CrossEntropyLoss` + `Adam` + `ReduceLROnPlateau` em cada fase, mantendo os pesos da melhor época de validação ao final.
* `History` — loss/acurácia de treino e validação por época, com a fase (`"frozen"`/`"finetune"`) de cada uma.

### `predict.py`

```text
src/pytorch_resnet18_transfer/predict.py
```

* `predict(image_path, model, classes, image_size, device)` — classifica uma imagem e retorna a classe prevista e as probabilidades por classe (softmax).

### `utils.py`

```text
src/pytorch_resnet18_transfer/utils.py
```

* `get_device` / `set_seed` — dispositivo (CPU/GPU) e reprodutibilidade.
* `save_checkpoint` / `load_checkpoint` — salvam e restauram `model_state_dict` + `classes` + `image_size` via `torch.save`/`torch.load`.

### CLI

```text
src/cli/cli.py
```

Disponibiliza os comandos:

```text
resnet18-train
resnet18-predict
```

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
pip install -r requirements-pytorch-resnet18-transfer.txt
```

(a partir da raiz do projeto)

## 5. Uso via CLI

### Treinar o modelo

```bash
python main.py resnet18-train \
    --dataset-path data/dataset \
    --frozen-epochs 5 \
    --finetune-epochs 15 \
    --batch-size 32 \
    --image-size 224
```

A Fase 1 treina apenas o classificador (extrator congelado); a Fase 2 libera a rede inteira para fine-tune com uma taxa de aprendizado menor (`--finetune-lr`). Ao final, o comando avalia no conjunto de teste e salva o checkpoint em `--model-out` (padrão: `artifacts/resnet18_transfer_model.pth`).

Outras opções úteis: `--head-lr`/`--finetune-lr`, `--train-ratio`/`--val-ratio`/`--test-ratio`, `--num-workers`, `--seed`, `--model-out`.

### Classificar uma imagem

```bash
python main.py resnet18-predict \
    --model artifacts/resnet18_transfer_model.pth \
    --image caminho/imagem.jpg
```

### Classificar uma pasta de imagens

```bash
python main.py resnet18-predict \
    --model artifacts/resnet18_transfer_model.pth \
    --image-dir caminho/para/pasta
```

`resnet18-predict` imprime a classe prevista e a probabilidade de cada uma das 5 vistas para cada imagem.

> `python main.py --help` lista os comandos disponíveis.

## 6. Validando o projeto

O projeto pode ser validado de **duas formas equivalentes**, já que ambas executam exatamente o mesmo código — o notebook não duplica lógica, ele importa diretamente de `src/pytorch_resnet18_transfer`:

### Opção A — CLI

```bash
python main.py resnet18-train --dataset-path data/dataset
python main.py resnet18-predict --model artifacts/resnet18_transfer_model.pth --image caminho/imagem.jpg
```

### Opção B — Notebook

```text
notebooks/pytorch_resnet18_transfer_colab.ipynb
```

Pronto para rodar no Google Colab com GPU — clona o repositório, baixa o dataset e usa as mesmas classes/funções de `src/pytorch_resnet18_transfer`.

## 7. Persistência e inferência posterior

`save_checkpoint` grava um único arquivo `.pth` com:

* `model_state_dict` — pesos da rede (extrator + classificador);
* `classes` — lista de rótulos, na ordem usada pelos índices de saída;
* `image_size` — necessário para repetir exatamente o mesmo pré-processamento usado no treino.

`load_checkpoint(path)` reconstrói o modelo (com o extrator já **liberado**, pronto para inferência ou para continuar o fine-tune), carrega os pesos e o deixa em modo de avaliação (`eval()`), pronto para chamadas a `predict(...)`.
