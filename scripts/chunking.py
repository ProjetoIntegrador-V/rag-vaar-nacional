"""
Chunking / Segmentação para pipeline RAG - FATEC Cotia

Estratégia:
- Extração de texto por página usando PyMuPDF.
- Preservação da página como metadado.
- Segmentação preferencial por parágrafos.
- Agrupamento até um limite de tokens.
- Overlap entre chunks para preservar contexto.
- Saída em JSONL, facilitando reprodução e integração com embeddings/vector DB.

Uso:
    python src/chunking.py

Coloque os PDFs em:
    data/pdfs/

Saída:
    output/chunks.jsonl
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

import fitz  # PyMuPDF
import tiktoken


# Parâmetros principais do experimento
CHUNK_SIZE = 600       # valor inicial dentro da faixa proposta de 500–800 tokens
OVERLAP = 90           # aproximadamente 15%
MIN_CHUNK_TOKENS = 40  # evita chunks muito pequenos


BASE_DIR = Path(__file__).resolve().parents[1]
PDF_DIR = BASE_DIR / "data" / "pdfs"
OUTPUT_FILE = BASE_DIR / "output" / "chunks.jsonl"

# Tokenizador utilizado apenas para controlar o tamanho dos chunks.
# Não é necessário utilizar um LLM ou API.
ENCODER = tiktoken.get_encoding("cl100k_base")


def normalize_text(text: str) -> str:
    """Limpa espaços e quebras de linha sem alterar o conteúdo essencial."""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_into_paragraphs(text: str) -> List[str]:
    """
    Divide o texto preferencialmente por parágrafos.
    Quando o PDF não preserva bem os parágrafos, usa linhas como fallback.
    """
    text = normalize_text(text)

    paragraphs = re.split(r"\n\s*\n", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    if len(paragraphs) <= 1:
        paragraphs = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

    return paragraphs


def count_tokens(text: str) -> int:
    return len(ENCODER.encode(text))


def make_chunks(paragraphs: List[str]) -> List[str]:
    """
    Agrupa parágrafos até CHUNK_SIZE tokens.

    Se um único parágrafo ultrapassar o limite, ele é dividido
    diretamente em blocos de tokens.
    """
    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0

    for paragraph in paragraphs:
        paragraph_tokens = count_tokens(paragraph)

        # Parágrafo muito grande: primeiro salva o chunk atual
        if paragraph_tokens > CHUNK_SIZE:
            if current:
                chunks.append("\n\n".join(current))
                current = []
                current_tokens = 0

            tokens = ENCODER.encode(paragraph)

            for start in range(0, len(tokens), CHUNK_SIZE - OVERLAP):
                piece = ENCODER.decode(tokens[start:start + CHUNK_SIZE])
                if count_tokens(piece) >= MIN_CHUNK_TOKENS:
                    chunks.append(piece.strip())
            continue

        # Adiciona o parágrafo ao chunk atual se couber
        if current_tokens + paragraph_tokens <= CHUNK_SIZE:
            current.append(paragraph)
            current_tokens += paragraph_tokens
        else:
            if current:
                chunks.append("\n\n".join(current))

            # Overlap baseado nos últimos parágrafos do chunk anterior
            overlap_parts: List[str] = []
            overlap_tokens = 0

            for previous in reversed(current):
                previous_tokens = count_tokens(previous)
                if overlap_tokens + previous_tokens > OVERLAP:
                    break
                overlap_parts.insert(0, previous)
                overlap_tokens += previous_tokens

            current = overlap_parts + [paragraph]
            current_tokens = overlap_tokens + paragraph_tokens

    if current:
        chunks.append("\n\n".join(current))

    return [chunk.strip() for chunk in chunks if count_tokens(chunk) >= MIN_CHUNK_TOKENS]


def extract_pdf(pdf_path: Path) -> List[Dict]:
    """Extrai texto página a página, preservando número da página."""
    pages = []

    with fitz.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text")
            text = normalize_text(text)

            if text:
                pages.append({
                    "page": page_number,
                    "text": text
                })

    return pages


def process_pdf(pdf_path: Path) -> List[Dict]:
    """Processa um PDF e retorna seus chunks com metadados."""
    pages = extract_pdf(pdf_path)

    all_chunks = []

    for page_data in pages:
        paragraphs = split_into_paragraphs(page_data["text"])
        chunks = make_chunks(paragraphs)

        for chunk_index, chunk in enumerate(chunks):
            all_chunks.append({
                "document_id": pdf_path.stem,
                "title": pdf_path.stem,
                "source_file": pdf_path.name,
                "page": page_data["page"],
                "section": None,
                "subsection": None,
                "chunk_id": f"{pdf_path.stem}_p{page_data['page']}_c{chunk_index}",
                "chunk_index": chunk_index,
                "token_count": count_tokens(chunk),
                "text": chunk
            })

    return all_chunks


def main() -> None:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"Nenhum PDF encontrado em: {PDF_DIR}")
        print("Adicione os PDFs na pasta data/pdfs/ e execute novamente.")
        return

    total_chunks = 0

    with OUTPUT_FILE.open("w", encoding="utf-8") as output:
        for pdf_path in pdf_files:
            try:
                chunks = process_pdf(pdf_path)

                for chunk in chunks:
                    output.write(
                        json.dumps(chunk, ensure_ascii=False) + "\n"
                    )

                total_chunks += len(chunks)
                print(f"[OK] {pdf_path.name}: {len(chunks)} chunks")

            except Exception as error:
                print(f"[ERRO] {pdf_path.name}: {error}")

    print("\nProcessamento concluído.")
    print(f"Documentos processados: {len(pdf_files)}")
    print(f"Total de chunks: {total_chunks}")
    print(f"Arquivo de saída: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
