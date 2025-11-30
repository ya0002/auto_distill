import os
import uuid
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Sequence, TypedDict

# Core libs from distilled.py
import chromadb
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
import arxiv
import wikipedia

# LangChain / LangGraph
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_experimental.tools import PythonAstREPLTool
from langchain_core.tools import tool
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langgraph.graph import StateGraph, END

# LLMs
from langchain_google_genai import ChatGoogleGenerativeAI


# ---------- Vector Store (Docling + Chroma) ----------
class DoclingVectorStore:
    def __init__(self, db_path: str = "./local_vector_db", collection_name: str = "docs"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.grouped_by_header = dict()
        self.converter = DocumentConverter()

    def ingest_pdf(self, pdf_path: str, max_tokens: int = 500):
        result = self.converter.convert(pdf_path)
        return self.ingest_doc(result.document, pdf_path, max_tokens)

    def ingest_arxiv(self, query: str, max_results: int = 1, max_tokens: int = 500):
        client = arxiv.Client()
        search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
        results = list(client.results(search))
        if not results:
            return {}
        paper = results[0]
        title = f"Arxiv: {paper.title}"
        try:
            result = self.converter.convert(paper.pdf_url)
            return self.ingest_doc(result.document, source_name=title, max_tokens=max_tokens)
        except Exception:
            return {}

    def ingest_wikipedia(self, query: str, max_tokens: int = 500, lang: str = "en"):
        wikipedia.set_lang(lang)
        try:
            search_results = wikipedia.search(query, results=1)
            page = wikipedia.page(search_results[0], auto_suggest=True)
            result = self.converter.convert(page.url)
            return self.ingest_doc(result.document, source_name=f"Wiki: {page.title}", max_tokens=max_tokens)
        except Exception:
            return {}

    def ingest_doc(self, doc, source_name, max_tokens=500):
        chunker = HybridChunker(tokenizer="sentence-transformers/all-MiniLM-L6-v2", max_tokens=max_tokens)
        chunks = list(chunker.chunk(doc))

        ids, documents, metadatas = [], [], []
        grouped_by_header = self.grouped_by_header

        for chunk in chunks:
            ids.append(str(uuid.uuid4()))
            documents.append(chunk.text)
            page_no = 0
            if chunk.meta.doc_items and chunk.meta.doc_items[0].prov:
                page_no = chunk.meta.doc_items[0].prov[0].page_no
            meta = {
                "filename": source_name,
                "headers": " > ".join(chunk.meta.headings) if chunk.meta.headings else "Root",
                "page_number": page_no,
            }
            metadatas.append(meta)
            grouped_by_header.setdefault(meta["headers"], []).append({"id": ids[-1], "content": documents[-1], "page": page_no})

        self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        self.grouped_by_header = grouped_by_header
        return grouped_by_header

    def query_n_merge(self, query_text: str, n_results: int = 3) -> List[Dict[str, Any]]:
        results = self.collection.query(query_texts=[query_text], n_results=n_results)
        structured = []
        if results.get("ids"):
            for i in range(len(results["ids"][0])):
                structured.append({
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results.get("distances", [[None]])[0][i],
                })
        structured.sort(key=lambda x: (x["metadata"].get("filename", ""), x["metadata"].get("page_number", 0)))
        merged = []
        from itertools import groupby
        key_func = lambda x: (x["metadata"].get("filename"), x["metadata"].get("page_number"))
        for (filename, page_num), group in groupby(structured, key=key_func):
            group_list = list(group)
            merged.append({
                "id": group_list[0]["id"],
                "text": "\n\n".join([g["text"] for g in group_list]),
                "metadata": group_list[0]["metadata"],
                "distance": min((g["distance"] for g in group_list if g["distance"] is not None), default=None),
            })
        return merged


# ---------- LLM Setup ----------
def _get_llm():
    api_key = os.getenv("GEMINI_KEY")
    if not api_key:
        # Return a dummy LLM-like object with simple content to avoid runtime failure
        class Dummy:
            def invoke(self, prompt):
                class R:
                    content = (
                        "<p>LLM unavailable. Set GEMINI_KEY to enable generation.</p>"
                    )
                return R()
        return Dummy()
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2, google_api_key=api_key)


python_repl_tool = PythonAstREPLTool()

# MCP client (in-process). For external MCP servers, swap this client.
from .mcp_server import MCPClient, SERVER as MCP_SERVER
MCP = MCPClient(MCP_SERVER)


@tool
def query_vector_db(query: str, db_path: str) -> str:
    """Query the Chroma vector database at `db_path` for semantic matches to `query`.

    Returns a markdown-formatted string combining retrieved chunk texts grouped by headers.
    If no matches are found, returns a fallback notice.
    """
    vector_db = DoclingVectorStore(db_path=db_path)
    results = vector_db.query_n_merge(query, n_results=10)
    val = []
    for res in results:
        val.append(f"## {res['metadata']['headers']}\n{res['text']}\n---")
    return f"# Context\n" + "\n".join(val) if val else "No specific definition found in VectorDB."


class ChapterPlan(TypedDict):
    chapter_id: int
    title: str
    goal: str
    data_requirements: str
    visual_requirements: str


class AgentState(TypedDict):
    raw_sections: Dict[str, Any]
    user_query: Optional[str]
    db_path: str
    story_title: str
    story_arc: List[ChapterPlan]
    current_chapter_index: int
    current_chapter_data: Dict[str, Any]
    current_chapter_vis: str
    finished_chapters: List[str]
    messages: Sequence[BaseMessage]
    critic_feedback: Optional[str]
    coder_attempts: int
    error: Optional[str]


def know_it_all_node(state: AgentState):
    user_query = state.get("user_query")
    if not user_query:
        return {"error": "No user query provided."}
    llm = _get_llm()
    # Use MCP tools for discovery and ingestion
    # 1) Try wikipedia and arxiv searches (discovery)
    _ = MCP.invoke("search_wikipedia", {"query": user_query})
    _ = MCP.invoke("search_arxiv", {"query": user_query})
    # 2) Build DB path and ingest via MCP
    db_path = state.get("db_path")
    if not db_path:
        db_path = os.path.join("outputs", f"my_rag_data_{uuid.uuid4()}")
    # no direct download here; rely on wikipedia/arxiv ingestion via DoclingVectorStore where applicable
    # For now, populate raw sections by running vector store wiki/arxiv helpers to seed content
    store = DoclingVectorStore(db_path=db_path)
    wiki = store.ingest_wikipedia(user_query)
    arx = store.ingest_arxiv(user_query)
    raw = {}
    raw.update(wiki or {})
    raw.update(arx or {})
    if not raw:
        return {"error": "Ingestion failed for provided query."}
    return {"raw_sections": raw, "db_path": db_path}


def planner_node(state: AgentState):
    llm = _get_llm()
    raw_headers = f"ALL HEADINGS: {list(state['raw_sections'].keys())}"
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are the Editor-in-Chief of Distill.pub. Create a JSON plan with blog_title and 5-8 chapters."""),
        ("user", "{raw_headers}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(llm, [query_vector_db], prompt)
    executor = AgentExecutor(agent=agent, tools=[query_vector_db], verbose=False)
    result = executor.invoke({"raw_headers": raw_headers, "db_path": state["db_path"], "user_query": state.get("user_query", "")})
    content = str(result.get("output", "")).replace("```json", "").replace("```", "")
    try:
        plan = json.loads(content)
    except Exception:
        plan = {"blog_title": state.get("user_query", "Distill Blog"), "chapters": [
            {"chapter_id": i+1, "title": f"Section {i+1}", "goal": "Explain key idea", "data_requirements": "None", "visual_requirements": "None"}
            for i in range(6)
        ]}
    return {
        "story_title": plan.get("blog_title", "Distill Blog"),
        "story_arc": plan.get("chapters", []),
        "current_chapter_index": 0,
        "finished_chapters": [],
        "coder_attempts": 0,
        "critic_feedback": None,
    }


def miner_node(state: AgentState):
    idx = state["current_chapter_index"]
    chapter = state["story_arc"][idx]
    # Call MCP tool for vector query
    resp = MCP.invoke("query_vector_db", {"query": f"{chapter['title']}: {chapter['goal']}", "db_path": state["db_path"]})
    return {"current_chapter_data": {"extracted": json.dumps(resp.get("results", []))}}


def coder_node(state: AgentState):
    llm = _get_llm()
    idx = state["current_chapter_index"]
    chapter = state["story_arc"][idx]
    if chapter.get("visual_requirements", "None") == "None":
        return {"current_chapter_vis": ""}
    resp = llm.invoke("Create a small D3.js interactive in a div with id 'vis_chapter_%d'" % idx)
    code = resp.content.replace("```html", "").replace("```", "")
    return {"current_chapter_vis": code}


def critic_node(state: AgentState):
    return {"critic_feedback": None}


def writer_node(state: AgentState):
    idx = state["current_chapter_index"]
    chapter = state["story_arc"][idx]
    data = state["current_chapter_data"].get("extracted", "")
    vis = state.get("current_chapter_vis", "")
    html = f"<section id='chapter-{idx}'><h2>{chapter['title']}</h2><p>{chapter['goal']}</p><div class='vis-wrapper'>{vis}</div><pre>{data}</pre></section>"
    finished = state.get("finished_chapters", [])
    finished.append(html)
    return {"finished_chapters": finished, "current_chapter_index": idx + 1, "coder_attempts": 0, "critic_feedback": None}


def router_node(state: AgentState):
    if state.get("error"):
        return "finish"
    return "continue" if state["current_chapter_index"] < len(state["story_arc"]) else "finish"


def build_workflow():
    workflow = StateGraph(AgentState)
    workflow.add_node("know_it_all", know_it_all_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("miner", miner_node)
    workflow.add_node("coder", coder_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("writer", writer_node)
    workflow.set_entry_point("know_it_all")
    workflow.add_edge("know_it_all", "planner")
    workflow.add_edge("planner", "miner")
    workflow.add_edge("miner", "coder")
    workflow.add_edge("coder", "critic")
    workflow.add_conditional_edges("critic", lambda s: "approve", {"approve": "writer"})
    workflow.add_conditional_edges("writer", router_node, {"continue": "miner", "finish": END})
    return workflow.compile()


def save_blog(title: str, chapters_html: List[str], out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    full_body = "\n".join(chapters_html)
    filename = f"{title.replace(' ', '_').replace(':', '').lower()}_distill.html"
    path = os.path.join(out_dir, filename)
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{title}</title>
        <script src="https://d3js.org/d3.v7.min.js"></script>
        <script>
        window.MathJax = {{ tex: {{ inlineMath: [['$', '$'], ['\\(', '\\)']] }} }};
        </script>
        <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
        <style>body{{font-family:serif;max-width:840px;margin:0 auto;padding:32px}}.vis-wrapper{{margin:24px 0;padding:16px;border:1px solid #eee;border-radius:8px}}</style>
    </head>
    <body>
        <h1>{title}</h1>
        <div class="metadata">Generated {datetime.now().strftime('%Y-%m-%d')}</div>
        {full_body}
    </body>
    </html>
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


def run_agent(user_query: str, outputs_dir: str, db_path: Optional[str] = None) -> str:
    db_path = db_path or os.path.join(outputs_dir, f"my_rag_data_{uuid.uuid4()}")
    initial_state: AgentState = {
        "raw_sections": {},
        "user_query": user_query,
        "story_title": "",
        "story_arc": [],
        "current_chapter_index": 0,
        "current_chapter_data": {},
        "current_chapter_vis": "",
        "finished_chapters": [],
        "messages": [],
        "error": None,
        "critic_feedback": None,
        "coder_attempts": 0,
        "db_path": db_path,
    }
    app = build_workflow()
    final_state = app.invoke(initial_state, config={"recursion_limit": 100})
    title = final_state.get("story_title", user_query)
    chapters = final_state.get("finished_chapters", [])
    return save_blog(title, chapters, outputs_dir)


def run_agent_with_pdf(pdf_path: str, outputs_dir: str) -> str:
    db_path = os.path.join(outputs_dir, f"my_rag_data_{uuid.uuid4()}")
    # Ingest via MCP server tool
    MCP.invoke("ingest_pdf", {"pdf_path": pdf_path, "db_path": db_path})
    # Use filename stem as query topic
    topic = os.path.splitext(os.path.basename(pdf_path))[0]
    return run_agent(topic, outputs_dir, db_path=db_path)
