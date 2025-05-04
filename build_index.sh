#!/usr/bin/env bash

# Example: build_index.sh
# This script parses all PDFs in 'pdf_repo/' and encodes them into FAISS

echo "📄 Parsing PDFs in pdf_repo..."
python parse_pdf_to_json.py pdf_repo

echo "⚙️ Encoding JSON into FAISS index..."
python encode_json.py

echo "✅ Done! You now have auto_pdf_doc.json and faiss_index.pkl"

echo "🔍 Launching FAISS query interface..."
python faiss_query.py