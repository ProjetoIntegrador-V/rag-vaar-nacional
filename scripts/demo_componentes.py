"""Demonstra a chamada dos dois componentes, lado a lado.

    python scripts/demo_componentes.py                 # so BM25, nao baixa nada
    python scripts/demo_componentes.py --com-embedding # baixa ~1.2 GB e roda o Qwen

Serve para a equipe ver o formato de entrada e de saida de cada peca antes de
decidir onde os vetores e os textos vao morar.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.esparso import BM25Retriever

DOCS = [
    ("vaar-conceito",
     "O VAAR (Valor Aluno Ano por Resultados) e a parcela da complementacao da "
     "Uniao ao FUNDEB condicionada ao cumprimento de condicionalidades."),
    ("vaar-habilitacao",
     "Para se habilitar ao VAAR, o ente deve comprovar as condicionalidades do "
     "art.14 da Lei 14.113/2020, entre elas criterios tecnicos para provimento "
     "da direcao escolar."),
    ("icms-educacional",
     "O ICMS Educacional de Minas Gerais distribui recursos aos municipios com "
     "base no Indice de Qualidade da Educacao calculado pela Fundacao Joao Pinheiro."),
    ("codigo-municipio",
     "O codigo IBGE 3106200 identifica o municipio de Belo Horizonte nos "
     "repasses do FUNDEB."),
]

PERGUNTAS = [
    "quais criterios habilitam o municipio ao VAAR?",   # semantica, BM25 sofre
    "art.14 da lei 14.113",                              # ancora exata, BM25 brilha
    "codigo 3106200",                                    # ancora exata
]

ids = [doc_id for doc_id, _ in DOCS]
textos = [texto for _, texto in DOCS]


def demo_bm25() -> None:
    print("=" * 70)
    print("BM25 (lexical)")
    print("=" * 70)

    bm25 = BM25Retriever().fit(textos, ids)

    for pergunta in PERGUNTAS:
        print(f"\n  ? {pergunta}")
        resultados = bm25.score(pergunta, top_k=3)
        if not resultados:
            print("    (nenhum termo em comum com o corpus)")
            continue
        for doc_id, score in resultados:
            print(f"    {score:7.3f}  {doc_id}")


def demo_embedding() -> None:
    import numpy as np

    from src.embedding import QwenEmbedder

    print("\n" + "=" * 70)
    print("Qwen3-Embedding-0.6B (denso, via HuggingFace)")
    print("=" * 70)

    embedder = QwenEmbedder()
    print(f"\n  modelo: {embedder.model_name}")

    vetores_doc = embedder.encode_documents(textos)
    print(f"  encode_documents -> matriz {vetores_doc.shape} (sem prefixo)")
    print(f"  norma do primeiro vetor: {np.linalg.norm(vetores_doc[0]):.4f}")

    for pergunta in PERGUNTAS:
        v_query = embedder.encode_queries([pergunta])[0]
        # Vetores normalizados: produto interno == cosseno.
        similaridades = vetores_doc @ v_query
        ordem = np.argsort(-similaridades)[:3]
        print(f"\n  ? {pergunta}")
        for i in ordem:
            print(f"    {similaridades[i]:7.4f}  {ids[i]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--com-embedding", action="store_true",
                        help="roda tambem o Qwen (baixa ~1.2 GB na primeira vez)")
    args = parser.parse_args()

    demo_bm25()
    if args.com_embedding:
        demo_embedding()
    else:
        print("\n(use --com-embedding para rodar tambem o Qwen3-Embedding-0.6B)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
