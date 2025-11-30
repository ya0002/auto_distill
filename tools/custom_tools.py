import os
import sys

# Ensure project root is on sys.path so `utils` can be imported even when running from `tools/`
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import glob
import subprocess
import uuid
import chromadb
import wikipedia
import arxiv
import pandas as pd
import json
from itertools import groupby
from typing import List, Dict, Any, Optional

from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from langchain_core.tools import tool
from langchain_experimental.tools import PythonAstREPLTool

from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker import HybridChunker

from utils import DoclingVectorStore


# --- TOOLS ---

python_repl_tool = PythonAstREPLTool()


@tool
def query_vector_db(query: str, db_path: str) -> str:
    """
    Queries the vector database for semantic context and knowledge base.
    Useful for finding definitions of terms.
    Params:
    query: what is the context needed for
    db_path: path to the vector database
    """
    vector_db = DoclingVectorStore(db_path=db_path)
    results = vector_db.query_n_merge(query, n_results=10)

    # 4. Display
    val = []
    for res in results:
        val.append(f"## {res['metadata']['headers']}\n{res['text']}\n---")

    if val:
        val_string = "\n".join(val)
        return f"# Context\n{val_string}"

    return "No specific definition found in VectorDB, rely on internal knowledge."


def fetch_wikipedia_content(query: str, max_chars: int = 8000, lang: str = "en") -> str:
    """
    Searches Wikipedia for a query and fetches the content of the most relevant page.

    This tool is designed for AI agents. It handles the search, retrieves the
    top matching page, and manages disambiguation errors by returning
    alternative options if the query is unclear.

    Args:
        query (str): The search topic (e.g., "Python programming", "Isaac Newton").
        max_chars (int, optional): The maximum number of characters to return
                                   to save context tokens. Defaults to 8000.
        lang (str, optional): The language code (e.g., 'en', 'es'). Defaults to 'en'.

    Returns:
        str: The full text of the article (truncated), a list of disambiguation
             options, or an error message.
    """
    wikipedia.set_lang(lang)

    try:
        # Step 1: Search to get the most specific title
        # We limit results to 1 to try and get the best match immediately
        search_results = wikipedia.search(query, results=1)

        if not search_results:
            return f"No Wikipedia results found for query: '{query}'"

        # Step 2: Fetch the page using the specific title found
        # auto_suggest=False prevents the library from guessing wrong on typos
        page_title = search_results[0]
        page = wikipedia.page(page_title, auto_suggest=False)

        # Step 3: Clean and Truncate Content
        content = page.content
        if len(content) > max_chars:
            content = (
                content[:max_chars]
                + f"\n... [Content truncated. Original length: {len(page.content)} chars]"
            )

        return f"Title: {page.title}\n" f"URL: {page.url}\n" f"Content:\n{content}"

    except wikipedia.exceptions.DisambiguationError as e:
        # The API found multiple pages. Return the list so the Agent can choose.
        options = e.options[:10]  # Limit options to first 10
        return f"Ambiguous query '{query}'. Did you mean one of these?: {', '.join(options)}"

    except wikipedia.exceptions.PageError:
        return f"PageError: The page for '{query}' could not be accessed."

    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"


@tool
def search_wikipedia_tool(query: str) -> str:
    """
    Useful for when you need to answer questions about history, science,
    people, or definitions. Input should be a specific search query.
    """
    return fetch_wikipedia_content(query)


def search_arxiv_papers(query: str, max_results: int = 20) -> str:
    """
    Searches Arxiv for research papers and returns their titles, authors, URLs, and summaries.

    Use this tool when you need to find scientific papers, check the latest research
    on a topic, or find summaries of specific technical concepts.

    Args:
        query (str): The search topic (e.g., "Attention mechanisms", "Quantum computing").
        max_results (int): Max papers to return. Defaults to 20.

    Returns:
        str: A formatted string containing the details of the found papers.
    """
    print(f"--- Searching Arxiv for: '{query}' ---")

    # 1. Initialize Client
    client = arxiv.Client()

    # 2. Configure Search
    # SortCriterion.Relevance ensures we get the best matches, not just the newest
    search = arxiv.Search(
        query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance
    )

    results = []

    try:
        # 3. Fetch and Format Results
        for result in client.results(search):
            # Clean up the summary (remove newlines to make it a single block of text)
            clean_summary = result.summary.replace("\n", " ")

            # Format the authors list
            authors = ", ".join([author.name for author in result.authors])

            paper_info = (
                f"Title: {result.title}\n"
                f"Authors: {authors}\n"
                f"Published: {result.published.strftime('%Y-%m-%d')}\n"
                f"URL: {result.pdf_url}\n"
                f"Summary: {clean_summary}\n"
                f"---"
            )
            results.append(paper_info)

        if not results:
            return f"No papers found for query: {query}"

        return "\n".join(results)

    except Exception as e:
        return f"An error occurred while searching Arxiv: {str(e)}"


@tool
def arxiv_search_tool(query: str) -> str:
    """
    Useful for finding scientific papers and summaries on a specific topic.
    Input should be a search query like 'Large Language Models' or 'Photosynthesis'.
    Returns titles, authors, links, and abstracts.
    """
    return search_arxiv_papers(query)


class LibraryDocsDB:
    def __init__(
        self,
        db_path="./chroma_db_native",
        source_root="./my_docs_source",
        auto_ingest=True,
    ):
        self.source_root = source_root
        self.db_path = db_path

        # 1. Initialize Native ChromaDB Client
        self.client = chromadb.PersistentClient(path=self.db_path)

        # Get or create the collection
        # We use cosine distance for semantic similarity
        self.collection = self.client.get_or_create_collection(
            name="library_docs", metadata={"hnsw:space": "cosine"}
        )

        # 2. Initialize Embedding Model (MiniLM is fast and good for code/docs)
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        print(f"Loading embedding model: {self.model_name}...")
        self.embedder = SentenceTransformer(self.model_name)

        # populate db with docs if not available
        if (
            auto_ingest
            and len(
                self.query("Explain scaleLinear", library_filter="d3")["documents"][0]
            )
            == 0
        ):
            print("Ingesting library documentation...")
            self.ingest()

    def _ensure_repos(self):
        """Clones D3 and ThreeJS repositories if they don't exist."""
        repos = {
            "threejs": ("https://github.com/mrdoob/three.js.git", "docs"),
            "d3": ("https://github.com/d3/d3.git", "."),
        }

        if not os.path.exists(self.source_root):
            os.makedirs(self.source_root)

        for lib_name, (url, _) in repos.items():
            lib_path = os.path.join(self.source_root, lib_name)
            if not os.path.exists(lib_path):
                print(f"[{lib_name}] Cloning repo...")
                subprocess.run(
                    ["git", "clone", "--depth", "1", url, lib_path], check=True
                )
            else:
                print(f"[{lib_name}] Repo exists.")

    def _get_files(self) -> List[tuple]:
        """Finds all HTML/MD files and tags them with their library name."""
        files = []
        # We only care about these extensions
        extensions = ["**/*.html", "**/*.md"]

        for lib_name in ["threejs", "d3"]:
            lib_path = os.path.join(self.source_root, lib_name)
            if not os.path.isdir(lib_path):
                continue

            for ext in extensions:
                # Recursive search
                found = glob.glob(os.path.join(lib_path, ext), recursive=True)
                for f in found:
                    files.append((f, lib_name))
        return files

    def ingest(self):
        """Parses files with Docling, chunks them, embeds them, and saves to Chroma."""
        self._ensure_repos()

        # Docling Setup
        converter = DocumentConverter()
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        chunker = HybridChunker(tokenizer=tokenizer, max_tokens=512, merge_peers=True)

        files = self._get_files()
        print(f"Found {len(files)} files to ingest.")

        for i, (file_path, lib_name) in enumerate(files):
            try:
                # A. Parse (Docling)
                conv_result = converter.convert(file_path)
                doc = conv_result.document

                # B. Chunk (Hybrid)
                chunk_iter = chunker.chunk(doc)

                # Prepare batch data for this file
                ids = []
                documents = []
                metadatas = []

                for chunk in chunk_iter:
                    text_content = chunk.text
                    if not text_content.strip():
                        continue

                    # Generate a unique ID for Chroma
                    ids.append(str(uuid.uuid4()))
                    documents.append(text_content)
                    metadatas.append(
                        {
                            "source": file_path,
                            "library": lib_name,
                            "type": "docling_hybrid",
                        }
                    )

                if not documents:
                    continue

                # C. Embed (SentenceTransformers)
                # We embed the list of strings in one go for speed
                embeddings = self.embedder.encode(documents).tolist()

                # D. Store (Native Chroma)
                self.collection.add(
                    ids=ids,
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas,
                )

                if (i + 1) % 10 == 0:
                    print(f"Processed {i+1}/{len(files)} files...")

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

        print("Ingestion complete.")

    def query(
        self, question: str, library_filter: Optional[str] = None, n_results: int = 5
    ):
        """
        Embeds the question and searches ChromaDB.
        """
        # print(f"\n--- Question: '{question}' [Filter: {library_filter}] ---")

        # 1. Embed the query
        query_embedding = self.embedder.encode([question]).tolist()

        # 2. Build Filter
        # Chroma native filter syntax: where={"field": "value"}
        where_clause = {"library": library_filter} if library_filter else None

        # 3. Search
        results = self.collection.query(
            query_embeddings=query_embedding, n_results=n_results, where=where_clause
        )

        # 4. Parse Results
        # Chroma returns lists of lists (because you can query multiple embeddings at once)
        if not results["documents"][0]:
            print("No results found.")
            return results

        return results


@tool
def d3js_documentation_reference(query: str) -> str:
    """
    Useful for answering questions about the D3.js data visualization library.
    Use this to look up specific D3 functions, scales, or usage examples.
    """
    db = LibraryDocsDB()
    results = db.query(query, library_filter="d3")["documents"][0]
    return f"QUERY : {query}\n---\n{'---\n---'.join(results)}"


@tool
def threejs_documentation_reference(query: str) -> str:
    """
    Useful for answering questions about the Three.js 3D library.
    Use this to find information on geometries, materials, scenes, or WebGL rendering.
    """
    db = LibraryDocsDB()
    results = db.query(query, library_filter="threejs")["documents"][0]
    return f"QUERY : {query}\n---\n{'---\n---'.join(results)}"


if __name__ == "__main__":
    db = LibraryDocsDB()
    print(db.query("Explain scaleLinear", library_filter="d3"))
