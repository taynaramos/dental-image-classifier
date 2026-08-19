# Dental Image Classifier

Classifica imagens intraorais odontológicas em 5 vistas — **frontal**, **superior**, **inferior**, **lateral direita** e **lateral esquerda**.

O projeto reúne duas soluções independentes para o mesmo problema, cada uma como um módulo próprio dentro de `src/`, com sua própria documentação e suas próprias dependências:

| Módulo | Abordagem | Documentação | Dependências |
| --- | --- | --- | --- |
| [`src/pca_svc/`](src/pca_svc/README.md) | PCA + SVC (numpy/scikit-learn, sem PyTorch) | [src/pca_svc/README.md](src/pca_svc/README.md) | [requirements-pca-svc.txt](requirements-pca-svc.txt) |
| [`src/pytorch_kfold/`](src/pytorch_kfold/README.md) | CNN em PyTorch, treinada do zero, com validação cruzada k-fold por sujeito | [src/pytorch_kfold/README.md](src/pytorch_kfold/README.md) | [requirements-pytorch-kfold.txt](requirements-pytorch-kfold.txt) |

Cada módulo pode ser instalado e usado isoladamente — instalar apenas `requirements-pca-svc.txt`, por exemplo, não requer PyTorch.

---

## Estrutura do projeto

```text
main.py                          # ponto de entrada da CLI

src/
├── cli/
│   ├── cli.py                   # registra os comandos de todos os módulos
│   └── cmd/                     # um comando por módulo (pca-*, kfold-*)
├── pca_svc/                      # módulo PCA + SVC (ver seu próprio README)
└── pytorch_kfold/                # módulo CNN em PyTorch (ver seu próprio README)

notebooks/                       # notebooks de cada módulo (Colab e local)
requirements-pca-svc.txt
requirements-pytorch-kfold.txt
```

## Uso via CLI

Depois de instalar as dependências do módulo desejado (veja a documentação de cada um), todos os comandos passam pelo mesmo ponto de entrada:

```bash
python main.py --help
```

```text
{pca-train, pca-predict, kfold-train, kfold-predict}
```

Consulte o README de cada módulo para os parâmetros específicos de treino e inferência.
