# # --- FILE WRITER ---


import os
from datetime import datetime

from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker

import chromadb
import uuid
from typing import List, Dict, Any
import wikipedia


import arxiv

from itertools import groupby


class DoclingVectorStore:
    def __init__(
        self, db_path: str = "./local_vector_db", collection_name: str = "docs"
    ):
        """
        Initialize the Vector Store.

        Args:
            db_path: Folder path where ChromaDB will store files.
            collection_name: Name of the collection inside ChromaDB.
        """
        print(f"Initializing Vector DB at '{db_path}'...")
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.grouped_by_header = dict()

        # Initialize Docling once (loads models)
        print("Initializing Docling Converter...")
        self.converter = DocumentConverter()

    def ingest_pdf(self, pdf_path: str, max_tokens: int = 500):
        """
        Reads a PDF, chunks it via HybridChunker, and saves to ChromaDB.
        """
        print(f"--- Processing: {pdf_path} ---")

        # 1. Convert PDF
        result = self.converter.convert(pdf_path)
        doc = result.document

        return self.ingest_doc(doc, pdf_path, max_tokens)

    def ingest_arxiv(self, query: str, max_results: int = 1, max_tokens: int = 500):
        """
        Searches Arxiv for a query, fetches the top paper's PDF, and ingests it.
        """
        print(f"--- Searching Arxiv for: '{query}' ---")

        # 1. Search Arxiv
        client = arxiv.Client()
        search = arxiv.Search(
            query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance
        )

        results = list(client.results(search))

        if not results:
            print("No Arxiv papers found.")
            return {}

        # 2. Process the top result
        paper = results[0]
        pdf_url = paper.pdf_url
        title = f"Arxiv: {paper.title}"

        print(f"Found Paper: {paper.title}")
        print(f"PDF URL: {pdf_url}")
        print("Downloading and processing with Docling...")

        # 3. Convert via URL
        # Docling can download and parse the PDF directly from the link
        try:
            result = self.converter.convert(pdf_url)
            return self.ingest_doc(
                result.document, source_name=title, max_tokens=max_tokens
            )
        except Exception as e:
            print(f"Error processing Arxiv PDF: {e}")
            return {}

    def ingest_wikipedia(self, query: str, max_tokens: int = 500, lang: str = "en"):
        """
        Resolves a Wikipedia query to a URL, fetches it via Docling, and ingests it.
        """
        wikipedia.set_lang(lang)

        try:
            # 1. Resolve Query to Page/URL
            search_results = wikipedia.search(query, results=1)
            wiki_page = wikipedia.page(search_results[0], auto_suggest=True)
            url = wiki_page.url
            title = f"Wiki: {wiki_page.title}"

            print(f"--- Processing Wikipedia: {title} ({url}) ---")

            # 2. Convert URL using Docling
            # Docling handles HTML parsing, preserving headers for the chunker
            result = self.converter.convert(url)

            # 3. Ingest using shared logic
            return self.ingest_doc(
                result.document, source_name=title, max_tokens=max_tokens
            )

        except wikipedia.exceptions.DisambiguationError as e:
            print(f"Error: Ambiguous query. Options: {e.options[:5]}")
            return {}
        except wikipedia.exceptions.PageError:
            print(f"Error: Page '{query}' not found.")
            return {}
        except Exception as e:
            print(f"Error: {e}")
            return {}

    def ingest_doc(self, doc, source_name, max_tokens=500):

        # 2. Chunking
        chunker = HybridChunker(
            tokenizer="sentence-transformers/all-MiniLM-L6-v2", max_tokens=max_tokens
        )
        chunks = list(chunker.chunk(doc))
        print(f"Generated {len(chunks)} chunks. Uploading to DB...")

        # 3. Prepare Data for Chroma
        ids = []
        documents = []
        metadatas = []
        grouped_by_header = self.grouped_by_header  ## copy over prev vals

        for chunk in chunks:
            # Generate a unique ID (or use chunk.id if stable)
            ids.append(str(uuid.uuid4()))

            # Content
            documents.append(chunk.text)

            # Metadata Flattening (Vector DBs usually prefer flat strings/ints)
            # Handle page numbers safely
            page_no = 0
            if chunk.meta.doc_items and chunk.meta.doc_items[0].prov:
                page_no = chunk.meta.doc_items[0].prov[0].page_no

            metadatas.append(
                {
                    "filename": source_name,
                    "headers": (
                        " > ".join(chunk.meta.headings)
                        if chunk.meta.headings
                        else "Root"
                    ),
                    "page_number": page_no,
                }
            )

            # Group by headers
            if metadatas[-1]["headers"] not in grouped_by_header:
                grouped_by_header[metadatas[-1]["headers"]] = []
            grouped_by_header[metadatas[-1]["headers"]].append(
                {"id": ids[-1], "content": documents[-1], "page": page_no}
            )

        # 4. Upsert to DB
        self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

        self.grouped_by_header = grouped_by_header  ## assign new dict when complete
        print("Ingestion Complete.")

        return grouped_by_header

    def query(self, query_text: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """
        Searches the database for context.
        """
        results = self.collection.query(query_texts=[query_text], n_results=n_results)

        # Format the raw Chroma results into a cleaner list of dictionaries
        structured_results = []
        if results["ids"]:
            for i in range(len(results["ids"][0])):
                structured_results.append(
                    {
                        "id": results["ids"][0][i],
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": (
                            results["distances"][0][i]
                            if "distances" in results
                            else None
                        ),
                    }
                )

        return structured_results

    def query_n_merge(
        self, query_text: str, n_results: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Searches the database and merges context by source and page number.
        """
        results = self.collection.query(query_texts=[query_text], n_results=n_results)

        # 1. Format raw results into a list of dicts
        structured_results = []
        if results["ids"]:
            for i in range(len(results["ids"][0])):
                structured_results.append(
                    {
                        "id": results["ids"][0][i],
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": (
                            results["distances"][0][i]
                            if "distances" in results
                            else None
                        ),
                    }
                )

        # 2. Sort by filename (source) then page_number
        # This is required for groupby to work correctly and ensures logical reading order
        structured_results.sort(
            key=lambda x: (
                x["metadata"].get("filename", ""),
                x["metadata"].get("page_number", 0),
            )
        )

        # 3. Group and Merge
        merged_results = []

        # We group by a tuple of (filename, page_number)
        key_func = lambda x: (
            x["metadata"].get("filename"),
            x["metadata"].get("page_number"),
        )

        for (filename, page_num), group in groupby(structured_results, key=key_func):
            group_list = list(group)

            # Concatenate text from all chunks on this specific page/source
            # We use "\n\n" to clearly separate the original chunks
            merged_text = "\n\n".join([item["text"] for item in group_list])

            # We take the metadata and ID from the first item in the group
            # For distance, we keep the minimum (best) score found in the group
            best_distance = min(
                (
                    item["distance"]
                    for item in group_list
                    if item["distance"] is not None
                ),
                default=None,
            )

            merged_results.append(
                {
                    "id": group_list[0]["id"],  # Representative ID
                    "text": merged_text,
                    "metadata": group_list[0]["metadata"],
                    "distance": best_distance,
                }
            )

        return merged_results


def save_blog(title, chapters_html, author="Auto Distill Agent", outputs_dir="."):
    # Join chapters with a semantic section divider, not just a generic HR
    full_body = "\n".join(chapters_html)

    # Generate a clean filename
    filename = f"{title.replace(' ', '_').replace(':', '').lower()}_distill.html"
    filename = os.path.join(outputs_dir, filename)

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{title}</title>
        
        <script src="https://d3js.org/d3.v7.min.js"></script>
        
        <script>
        window.MathJax = {{
          tex: {{
            inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
            displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
          }},
          svg: {{
            fontCache: 'global'
          }}
        }};
        </script>
        <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

        <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&family=Merriweather:ital,wght@0,300;0,400;0,700;1,300&display=swap" rel="stylesheet">
        
        <style>
            :root {{
                --font-sans: 'Roboto', -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
                --font-serif: 'Merriweather', Georgia, 'Times New Roman', serif;
                --color-text: #333;
                --color-bg: #fff;
                --color-accent: #000; /* Distill uses minimal color, mostly black/white */
                --color-caption: #666;
                --width-text: 700px;
                --width-wide: 1000px;
            }}

            /* --- BASE LAYOUT --- */
            body {{
                font-family: var(--font-serif);
                line-height: 1.6;
                color: var(--color-text);
                background: var(--color-bg);
                margin: 0;
                padding: 0;
                font-size: 19px; /* Distill uses slightly larger text for readability */
            }}

            /* Center the main content column */
            article {{
                max-width: var(--width-text);
                margin: 0 auto;
                padding: 2rem 1.5rem;
            }}

            /* --- TYPOGRAPHY --- */
            h1, h2, h3, h4, .front-matter {{
                font-family: var(--font-sans);
            }}

            h1 {{
                font-size: 3rem;
                font-weight: 700;
                line-height: 1.1;
                margin-top: 3rem;
                margin-bottom: 1rem;
                letter-spacing: -0.02em;
            }}

            h2 {{
                font-size: 1.75rem;
                font-weight: 500;
                margin-top: 3rem;
                margin-bottom: 1rem;
                border-bottom: 1px solid rgba(0,0,0,0.1);
                padding-bottom: 0.5rem;
            }}

            h3 {{
                font-size: 1.25rem;
                font-weight: 600;
                margin-top: 2rem;
                margin-bottom: 0.5rem;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                color: #555;
            }}

            p {{
                margin-bottom: 1.5em;
                font-weight: 300;
            }}

            a {{
                color: #0044cc;
                text-decoration: none;
                border-bottom: 1px solid transparent;
                transition: border 0.2s;
            }}
            
            a:hover {{
                border-bottom: 1px solid #0044cc;
            }}

            /* --- CODE BLOCKS --- */
            pre {{
                background: #f7f7f7;
                padding: 15px;
                border-radius: 5px;
                overflow-x: auto;
                font-size: 0.85em;
                border: 1px solid #eee;
            }}
            
            code {{
                background: rgba(0,0,0,0.05);
                padding: 2px 4px;
                border-radius: 3px;
                font-family: 'Menlo', 'Consolas', monospace;
                font-size: 0.9em;
            }}

            /* --- FIGURES & VISUALIZATIONS --- */
            /* Figures allow breaking out of the text column if needed */
            figure {{
                margin: 2.5rem 0;
                text-align: center;
            }}

            img, svg {{
                max-width: 100%;
                height: auto;
            }}

            figcaption {{
                font-family: var(--font-sans);
                font-size: 0.85rem;
                color: var(--color-caption);
                margin-top: 10px;
                line-height: 1.4;
                text-align: left; /* Distill captions are often left-aligned even if img is centered */
            }}

            .vis-wrapper {{ 
                margin: 40px 0; 
                padding: 20px; 
                background: white; 
                border: 1px solid #eee; 
                border-radius: 8px; 
            }}

            /* --- FRONT MATTER (Title Block) --- */
            .front-matter {{
                margin-bottom: 4rem;
                text-align: left;
                border-bottom: 1px solid #eee;
                padding-bottom: 2rem;
            }}

            .authors {{
                display: flex;
                flex-wrap: wrap;
                gap: 20px;
                font-size: 1rem;
                color: #444;
                margin-top: 1rem;
            }}

            .author-name {{
                font-weight: 500;
                color: #000;
            }}

            .metadata {{
                margin-top: 1rem;
                font-size: 0.85rem;
                color: #777;
            }}

        </style>
    </head>
    <body>

        <article>
            <div class="front-matter">
                <h1>{title}</h1>
                <div class="authors">
                    <div>
                        <span class="author-name">{author}</span><br>
                        <span style="font-size: 0.9em;">AI Research Assistant</span>
                    </div>
                </div>
                <div class="metadata">
                    Published on {datetime.now().strftime("%B %d, %Y")} &bull; Generated by Agentic Workflow
                </div>
            </div>

            {full_body}
            
            <hr style="margin: 4rem 0; border: 0; border-top: 1px solid #eee;">
            
            <div style="font-family: var(--font-sans); font-size: 0.8rem; color: #999; text-align: center;">
                End of Article
            </div>
        </article>

    </body>
    </html>
    """

    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n✅ Blog saved to '{filename}'")
    return filename
