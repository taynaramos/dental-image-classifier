# Classificador de Imagens Odontológicas

Projeto integrador da disciplina **Engenharia de Software para IA e Frameworks Profundos**
(Especialização em Deep Learning — CIn/UFPE, Turma III).
Prof. Fernando Maciano de Paula Neto.

Grupo: **UFPE - Classificador de Imagens Odontológicas**

## Tema

Classificação de **imagens odontológicas intraorais**. Dada uma foto intraoral, o sistema
identifica qual das **5 vistas** ela representa:

- frontal
- inferior (oclusal inferior)
- superior (oclusal superior)
- lateral direita
- lateral esquerda

## Dados

Base de dados privada cedida pela empresa de um dos membros do grupo:
aproximadamente **3.000 conjuntos de 5 fotos** (uma foto por vista).

## Entregas

| # | Entrega | Conteúdo |
|---|---------|----------|
| 1 | Funções, modularização e repositório | Escolha do problema, primeiras funções, repositório no GitHub, README, `requirements.txt`. Inclui tipagem e uso de NumPy. |
| 2 | PyTorch — Parte 1 | Conversão para tensores, `Dataset`, `DataLoader`, verificação de shapes/dtypes/device. |
| 3 | PyTorch — Parte 2 | Modelo (`nn.Module`), loss, otimizador, laço de treino/validação, salvar/carregar modelo, inferência. |
| 4 | Testes com `unittest` | Suíte de testes (dados, pré-processamento, tensores, saída do modelo, salvamento). |
| 5 | Requisitos | Documento de requisitos funcionais e não funcionais. |
| 6 | Design, arquitetura e Git | Arquitetura do sistema, fluxo de dados e organização do repositório (branches, commits, colaboração). |
| Final | Apresentação | Demonstração do sistema (máx. 10 min). |
