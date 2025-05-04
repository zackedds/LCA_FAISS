#!/usr/bin/env python3
"""
parse_pdf_simple.py  ─ lean version (no OCR / CV / NLTK)

• Accepts one PDF file or a directory containing PDFs
• Extracts plain text page‑by‑page with PyPDF
• Splits into chunks:
   – New chunk at numbered headings (“1.”, “1.1”, “A.”, “B.”) or blank line
   – Hard cap 1000 characters
• Writes <pdfname>.json next to each PDF, compatible with encode_json.py
"""

import sys, json, uuid, regex as re
from pathlib import Path
from pypdf import PdfReader
from typing import List

MAX_CHARS = 1000
HEAD_RE   = re.compile(r"^([0-9]+(\.[0-9]+)*|[A-Z])\s")  # 1.2  or  A.

def pdf_to_chunks(pdf: Path):
    reader = PdfReader(str(pdf))
    chunks, buf, cur_title = [], [], "Untitled"

    def flush():
        if buf:
            chunks.append({
                "id": str(uuid.uuid4()),
                "title": cur_title,
                "text": " ".join(buf).strip(),
                "source": pdf.name,
            })
            buf.clear()

    for page in reader.pages:
        for line in page.extract_text().splitlines():
            line = line.strip()
            if not line:
                continue                              # skip extra blank lines

            # heading ➜ start new chunk
            if HEAD_RE.match(line):
                flush()
                cur_title = line
                continue

            # add line, flush if too big
            if sum(len(x) for x in buf) + len(line) > MAX_CHARS:
                flush()
            buf.append(line)

    flush()
    return chunks

def main():
    """
    Collect chunks from ONE PDF or EVERY PDF in a folder and dump *all*
    chunks into a single JSON file called ``auto_pdf_doc.json`` in the
    current working directory (unless the caller provides a custom
    output path as the second CLI argument).

    Examples
    --------
    # Parse every PDF inside pdf_repo/ ➜ auto_pdf_doc.json
    python parse_pdf_to_json.py pdf_repo

    # Parse one PDF ➜ custom_output.json
    python parse_pdf_to_json.py LCA.pdf my_custom.json
    """
    if len(sys.argv) < 2:
        sys.exit("Usage: python parse_pdf_to_json.py <PDF | directory> [output.json]")

    target = Path(sys.argv[1]).expanduser().resolve()

    # Build the list of PDF files
    if target.is_dir():
        pdf_files: List[Path] = sorted(target.glob("*.pdf"))
        if not pdf_files:
            sys.exit(f"No PDFs found in {target}")
    elif target.is_file() and target.suffix.lower() == ".pdf":
        pdf_files = [target]
    else:
        sys.exit("First argument must be a PDF or directory containing PDFs")

    # Where to write the aggregated JSON
    out_path = (
        Path(sys.argv[2]).expanduser().resolve()
        if len(sys.argv) > 2
        else Path.cwd() / "auto_pdf_doc.json"
    )

    all_chunks = []
    for pdf in pdf_files:
        print(f"Parsing {pdf.name} …", end="", flush=True)
        chunks = pdf_to_chunks(pdf)
        all_chunks.extend(chunks)
        print(f"  +{len(chunks)} chunks")

    out_path.write_text(json.dumps(all_chunks, indent=2))
    print(f"\nWrote {len(all_chunks)} total chunks → {out_path}")

if __name__ == "__main__":
    main()