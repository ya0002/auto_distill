from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker

import chromadb
import uuid
from typing import List, Dict, Any


import wikipedia
import arxiv

import pandas as pd

from typing import List, Dict, Any
from itertools import groupby

import wikipedia


import arxiv

import json

from langchain_core.tools import tool
from langchain_experimental.tools import PythonAstREPLTool

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



