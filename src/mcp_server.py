"""
Minimal MCP-style server exposing tools used by the agent.

This is a simplified server that registers tool functions and can be
invoked via a local client wrapper. For Hugging Face Spaces, you can
run this server process separately (stdio/websocket) and point the
client to it. Here we keep it in-process for simplicity.
"""

from typing import Dict, Any
import os
import uuid

from .agent_pipeline import DoclingVectorStore
import arxiv
import wikipedia


class ToolError(Exception):
    pass


class MCPServer:
    def __init__(self):
        self.registry = {}
        self.register("query_vector_db", self.query_vector_db)
        self.register("ingest_pdf", self.ingest_pdf)
        self.register("search_wikipedia", self.search_wikipedia)
        self.register("search_arxiv", self.search_arxiv)

    def register(self, name: str, func):
        self.registry[name] = func

    # --- Tools ---
    def query_vector_db(self, params: Dict[str, Any]) -> Dict[str, Any]:
        query = params.get("query", "")
        db_path = params.get("db_path")
        if not db_path:
            raise ToolError("db_path is required")
        store = DoclingVectorStore(db_path=db_path)
        results = store.query_n_merge(query, n_results=10)
        return {"results": results}

    def ingest_pdf(self, params: Dict[str, Any]) -> Dict[str, Any]:
        pdf_path = params.get("pdf_path")
        db_path = params.get("db_path") or os.path.join("outputs", f"my_rag_data_{uuid.uuid4()}")
        if not pdf_path:
            raise ToolError("pdf_path is required")
        store = DoclingVectorStore(db_path=db_path)
        grouped = store.ingest_pdf(pdf_path)
        return {"db_path": db_path, "grouped": grouped}

    def search_wikipedia(self, params: Dict[str, Any]) -> Dict[str, Any]:
        query = params.get("query", "")
        wikipedia.set_lang(params.get("lang", "en"))
        results = wikipedia.search(query, results=3)
        return {"results": results}

    def search_arxiv(self, params: Dict[str, Any]) -> Dict[str, Any]:
        query = params.get("query", "")
        client = arxiv.Client()
        search = arxiv.Search(query=query, max_results=5, sort_by=arxiv.SortCriterion.Relevance)
        items = []
        for r in client.results(search):
            items.append({"title": r.title, "pdf_url": r.pdf_url})
        return {"results": items}


SERVER = MCPServer()


class MCPClient:
    """In-process client that calls the MCPServer registry.

    Replace this with a real MCP client (stdio/websocket) when deploying
    an external server.
    """

    def __init__(self, server: MCPServer):
        self.server = server

    def invoke(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if tool_name not in self.server.registry:
            raise ToolError(f"Unknown tool {tool_name}")
        return self.server.registry[tool_name](params)
