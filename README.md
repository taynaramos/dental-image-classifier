# Dental Image Classifier — PCA + SVC

## 1. O que é esta solução

Esta branch implementa o processo de classificação sem o uso do PyTorch.

O fluxo da solução é:

```text
Leitura da imagem
→ pré-processamento
→ extração de características com PCA
→ classificação com SVC
```

As imagens são convertidas para luminância, redimensionadas e transformadas em vetores. Depois, o PCA reduz a quantidade de características e o SVC realiza a classificação.

---

## 2. Módulos da solução

### `DentalDataset`

Arquivo:

```text
src/pca_svc/dataset.py
```

Responsável por preprocessamento das imagens para o treino, validação e teste:

* carregar as imagens;
* dividir os dados em treino, validação e teste;
* converter as imagens para luminância;
* redimensionar as imagens;
* transformar cada imagem em um vetor;
* gerar os rótulos.

### `FeatureExtractor`

Arquivo:

```text
src/pca_svc/feature_extractor.py
```

Responsável por realizar aquilo que as camadas da CNN fazem de forma automatica, isto é, extrair as caracteristicas da imagem que nos perimtam realizar a classificação. Diante disso o modulo faz o seguinte:

* normalizar os dados;
* aplicar PCA;
* reduzir a dimensionalidade das imagens para o espaço dos PC para inferencia.
  

### `DentalClassifier`

Arquivo:

```text
src/pca_svc/model.py
```

Responsável por realizar a parte final do processo, pega as caractericas representadas nos PCs, e realiza a classificação das imagens. Diante disso o modulo faz o seguinte:

* treinar o classificador SVC;
* realizar predições;
* retornar probabilidades por classe;
* avaliar o modelo;
* salvar e carregar o modelo treinado.

### `CLI`

Arquivo:

```text
src/cli/cli.py
```

Disponibiliza os comandos:

```text
pca-train
pca-predict
```

---

## 3. Como usar

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

### Carregar o módulo principal

```python
from src.cli import CLI

CLI().run()
```

Como a branch ainda não possui um `main.py`, a CLI pode ser executada desta forma:

```bash
python -c "from src.cli import CLI; CLI().run()" --help
```

### Treinar o modelo

```bash
python -c "from src.cli import CLI; CLI().run()" \
  pca-train \
  --dataset-path data/dataset \
  --model-out artifacts/pca_svc_model.pkl \
  --image-size 128 \
  --variance-threshold 0.95 \
  --seed 42
```

### Classificar uma imagem

```bash
python -c "from src.cli import CLI; CLI().run()" \
  pca-predict \
  --model artifacts/pca_svc_model.pkl \
  --image caminho/para/imagem.jpeg \
  --image-size 128
```

### Classificar um folder de imagens

```bash
python -c "from src.cli import CLI; CLI().run()" \
  pca-predict \
  --model artifacts/pca_svc_model.pkl \
  --image-dir caminho/para/pasta \
  --image-size 128
```
