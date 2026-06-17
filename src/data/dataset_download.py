import os
import zipfile
from typing import Dict, List, Set, Tuple

LABEL_MAP: Dict[str, str] = {
    "intraoral-frontal": "frontal",
    "intraoral-inferior": "inferior",
    "intraoral-superior": "superior",
    "intraoral-lateral-direita": "lateral_direita",
    "intraoral-lateral-esquerda": "lateral_esquerda",
}


def baixar_zip_do_drive(
    file_id: str,
    destino: str = "/content/dataset.zip",
    fuzzy: bool = True,
) -> str:
    """
    Baixa um arquivo .zip do Google Drive a partir do seu file_id,
    usando gdown. Funciona para arquivos compartilhados como
    "qualquer pessoa com o link".

    Args:
        file_id: ID do arquivo no Google Drive (extraído da URL,
            a parte entre '/d/' e '/view').
        destino: caminho local onde o .zip será salvo.
        fuzzy: se True, ajuda o gdown a lidar com o aviso de
            "verificação de vírus" que o Drive dispara em
            arquivos grandes.

    Returns:
        O caminho (destino) onde o arquivo foi salvo.

    Raises:
        RuntimeError: se o download falhar ou o arquivo vier
            visivelmente menor do que o esperado (sinal de que
            baixou uma página de erro em vez do zip).
    """
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError(
            "gdown não está instalado. Rode: !pip install gdown -q"
        ) from exc

    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, destino, quiet=False, fuzzy=fuzzy)

    if not os.path.exists(destino):
        raise RuntimeError(f"Download falhou: arquivo não encontrado em {destino}")

    tamanho_mb = os.path.getsize(destino) / (1024 ** 2)
    if tamanho_mb < 10:
        raise RuntimeError(
            f"Arquivo baixado tem apenas {tamanho_mb:.1f} MB — "
            "provavelmente é uma página de erro/aviso, não o zip real. "
            "Tente novamente com fuzzy=True ou verifique a permissão do link."
        )

    print(f"Download concluído: {destino} ({tamanho_mb:.1f} MB)")
    return destino


def extrair_zip(zip_path: str, destino_dir: str = "/content/dataset") -> str:
    """
    Extrai o .zip do dataset para um diretório local.

    Args:
        zip_path: caminho do arquivo .zip já baixado.
        destino_dir: diretório onde o conteúdo será extraído.

    Returns:
        O caminho raiz onde os dados foram extraídos (destino_dir).
    """
    os.makedirs(destino_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(destino_dir)

    print(f"Extração concluída em: {destino_dir}")
    return destino_dir


def _encontrar_raiz_dataset(destino_dir: str) -> str:
    """
    Detecta automaticamente o nível correto do dataset dentro do diretório
    extraído, descendo um nível se houver apenas uma subpasta intermediária.
    """
    conteudo = [
        item for item in os.listdir(destino_dir)
        if os.path.isdir(os.path.join(destino_dir, item))
    ]

    if len(conteudo) == 1:
        unico_item = os.path.join(destino_dir, conteudo[0])
        sub_conteudo = os.listdir(unico_item)
        subpastas_dentro = [
            s for s in sub_conteudo if os.path.isdir(os.path.join(unico_item, s))
        ]
        if len(subpastas_dentro) > 1:
            return unico_item

    return destino_dir


def validar_estrutura_dataset(
    dataset_path: str,
    label_map: Dict[str, str] = LABEL_MAP,
) -> Tuple[List[str], List[Tuple[str, Set[str], Set[str]]]]:
    """
    Percorre as pastas de casos dentro de dataset_path e verifica se cada uma
    contém exatamente os arquivos esperados (definidos em label_map).

    Args:
        dataset_path: caminho raiz onde estão as pastas de casos.
        label_map: dicionário nome_arquivo_sem_extensao -> rotulo.

    Returns:
        Tupla (pastas_validas, problemas):
            - pastas_validas: lista de nomes de pastas sem nenhum problema.
            - problemas: lista de tuplas (nome_pasta, faltando, extras).
    """
    pastas = sorted(
        item for item in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, item))
    )

    esperados = set(label_map.keys())
    pastas_validas = []
    problemas = []

    for pasta in pastas:
        caminho_pasta = os.path.join(dataset_path, pasta)
        arquivos = os.listdir(caminho_pasta)
        nomes_sem_ext = {os.path.splitext(a)[0] for a in arquivos}

        faltando = esperados - nomes_sem_ext
        extras = nomes_sem_ext - esperados

        if faltando or extras:
            problemas.append((pasta, faltando, extras))
        else:
            pastas_validas.append(pasta)

    print(f"Total de pastas (casos): {len(pastas)}")
    print(f"Pastas válidas: {len(pastas_validas)}")
    print(f"Pastas com problema: {len(problemas)}")

    return pastas_validas, problemas


def preparar_dataset(
    file_id: str,
    zip_destino: str = "/content/dataset.zip",
    extract_dir: str = "/content/dataset",
    forcar_novo_download: bool = False,
) -> str:
    """
    Encadeia download → extração → validação do dataset.
    Pensada para ser chamada uma vez por sessão do Colab.

    Args:
        file_id: ID do arquivo no Google Drive.
        zip_destino: caminho local para salvar o .zip baixado.
        extract_dir: diretório onde os dados serão extraídos.
        forcar_novo_download: se True, baixa de novo mesmo que o
            .zip já exista. Se False, reaproveita o arquivo existente.

    Returns:
        Caminho raiz validado do dataset extraído.

    Raises:
        RuntimeError: se a validação encontrar pastas com problema.
    """
    if forcar_novo_download or not os.path.exists(zip_destino):
        baixar_zip_do_drive(file_id, destino=zip_destino)
    else:
        print(f"Reaproveitando zip já existente em {zip_destino}")

    extrair_zip(zip_destino, destino_dir=extract_dir)

    dataset_root = _encontrar_raiz_dataset(extract_dir)
    if dataset_root != extract_dir:
        print(f"Subpasta intermediária detectada, usando: {dataset_root}")

    pastas_validas, problemas = validar_estrutura_dataset(dataset_root)

    if problemas:
        exemplos = problemas[:5]
        raise RuntimeError(
            f"{len(problemas)} pasta(s) com inconsistência encontradas. "
            f"Exemplos: {exemplos}. "
            "Use validar_estrutura_dataset() diretamente para inspecionar "
            "sem interromper o fluxo."
        )

    print(f"Dataset pronto em: {dataset_root} ({len(pastas_validas)} casos válidos)")
    return dataset_root
