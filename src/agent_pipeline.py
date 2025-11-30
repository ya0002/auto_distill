import os
import shutil
import operator
import uuid
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Sequence, TypedDict
from typing import Annotated, Sequence, TypedDict, Dict, Any, List
import urllib.request
import asyncio


# LangChain / LangGraph
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain.agents import create_agent
from langgraph.graph import StateGraph, END

# LLMs
from langchain_google_genai import ChatGoogleGenerativeAI

from tools.custom_tools import (
    arxiv_search_tool,
    search_wikipedia_tool,
    query_vector_db,
    python_repl_tool,
    d3js_documentation_reference,
    threejs_documentation_reference
)

from tools.mcp_tools import video_client

from utils import save_blog, DoclingVectorStore


GEMINI_API = os.getenv("GEMINI_KEY")
flash_model_name = ["gemini-2.0-flash", "gemini-2.5-flash"]
llm_flash = ChatGoogleGenerativeAI(
    model=flash_model_name[1], temperature=0.2, google_api_key=GEMINI_API
)

# llm_flash = ChatAnthropic(model="claude-haiku-4-5",
#                           temperature=0,
#                           api_key = ANTHROPIC_API_KEY
#                           )

creative_model_name = ["gemini-2.0-flash", "gemini-2.5-flash"]
llm_creative = ChatGoogleGenerativeAI(
    model=creative_model_name[1], temperature=0.7, google_api_key=GEMINI_API
)

# --- NODES (AGENTS) ---


class ChapterPlan(TypedDict):
    """Defines the blueprint for a single section of the blog."""

    chapter_id: int
    title: str
    goal: str  # What is the storytelling goal of this section?
    data_requirements: str  # What data needs to be mined?
    visual_requirements: str  # Description of the interactive needed (if any)


class AgentState(TypedDict):
    """The shared memory of the system."""

    # Global Inputs
    raw_sections: Dict[str, Any]
    user_query: Optional[
        str
    ]  ## something specifies by the user, would be passed to planner
    db_path: str
    outputs_dir: str

    # The Master Plan
    story_title: str
    story_arc: List[ChapterPlan]

    # Loop State (Processing one chapter at a time)
    current_chapter_index: int
    current_chapter_data: Dict[str, Any]  # Data mined for specific chapter
    current_chapter_vis: str  # HTML/JS for specific chapter
    current_chapter_video: Optional[str]  # Video URL if any

    # Outputs
    finished_chapters: List[str]  # List of HTML strings (the body text)
    messages: Annotated[Sequence[BaseMessage], operator.add]

    # CRITIC STATE
    critic_feedback: Optional[str]  # Feedback from the critic
    coder_attempts: int  # Count retries to prevent infinite loops

    # Error Handling
    error: Optional[str]  # If set, stops execution flow


def know_it_all_node(state: AgentState):
    """
    The Research Architect.

    Workflow:
    1. SEARCH: Uses Arxiv/Wiki SEARCH tools to find the exact paper titles/definitions.
    2. PLAN: Outputs a JSON identifying the best targets.
    3. INGEST: Triggers the VectorDB ingestion using the precise targets.
    """

    # 1. Check if data exists (Short-circuit)
    raw_sections = state.get("raw_sections", {})
    if raw_sections and len(raw_sections) > 0:
        print("---KNOW-IT-ALL: DATA DETECTED. SKIPPING.---")
        return {}

    user_query = state.get("user_query")
    if not user_query:
        return {"error": "No raw sections and no user query."}

    print(f"---KNOW-IT-ALL: RESEARCHING '{user_query}'---")

    # --- PHASE 1: THE DISCOVERY AGENT ---
    # This agent uses tools to READ, not to ingest.

    search_tools = [
        arxiv_search_tool, 
        search_wikipedia_tool]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a Senior Data Curator.
        
        GOAL: The user wants to write a blog about: "{user_query}".
        You need to find the specific documents we should add to our library.
        
        PROCESS:
        1. **Explore**: Use `arxiv_search_tool` and `search_wikipedia_tool` to find relevant material.
           - Example: If user asks for "Mamba", search Arxiv to find the full paper title "Mamba: Linear-Time Sequence Modeling...".
           - Example: If user asks for "CRISPR", search Wiki to verify the best page title.
        2. **Select**: Choose ONE foundational paper and numerous comprehensive wiki page.
        3. **Finalize**: Output a JSON object with the exact search terms to be used for ingestion.
        
        OUTPUT FORMAT (JSON ONLY):
        {{
            "reasoning": "I found paper X which covers the math, and Wiki page Y for history.",
            "arxiv_target": "The Exact Paper Title Found in Search",
            "wiki_target": ["The Exact Wiki Page Title", ...]
        }}
        
        If no Arxiv paper is relevant (e.g., for purely historical topics), set "arxiv_target" to "None".
        """,
            ),
            ("user", "{user_query}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    agent = create_tool_calling_agent(llm_flash, search_tools, prompt)
    executor = AgentExecutor(agent=agent, tools=search_tools, verbose=True)

    # Run the research loop
    try:
        response = executor.invoke({"user_query": user_query})
        raw_output = response["output"]
        content = ""

        # Check if output is a list (Gemini/Vertex often returns a list of blocks)
        if isinstance(raw_output, list):
            for block in raw_output:
                # Handle dictionary blocks (e.g. {'type': 'text', 'text': '...'})
                if isinstance(block, dict) and "text" in block:
                    content += block["text"]
                # Handle direct strings in list
                elif isinstance(block, str):
                    content += block
        else:
            # Standard string output
            content = str(raw_output)

        # Parse the JSON from the text response
        # (Handling potential markdown wrapping)
        clean_json = content.replace("```json", "").replace("```", "").strip()
        plan = json.loads(clean_json)

        print(f"--- RESEARCH COMPLETE ---")
        print(f" > Plan: {plan.get('reasoning')}")
        print(f" > Target Arxiv: {plan.get('arxiv_target')}")
        print(f" > Target Wiki:  {plan.get('wiki_target')}")

    except Exception as e:
        print(f"Research Agent failed: {e}")
        return {"error": f"Failed to plan research: {str(e)}"}

    # --- PHASE 2: THE INGESTION ENGINE ---
    # Now we strictly follow the plan using the internal DB methods

    vector_db = DoclingVectorStore(db_path=state.get("db_path"))

    # 1. Ingest Arxiv (if planned)
    target_paper = plan.get("arxiv_target")
    if target_paper and target_paper != "None":
        print(f" > Ingesting Arxiv: '{target_paper}'...")
        # Note: We use max_results=1 because the agent should have given us a specific title
        all_grouped_by_header = vector_db.ingest_arxiv(
            query=target_paper, max_results=1
        )

    # 2. Ingest Wikipedia (if planned)
    target_wikis = plan.get("wiki_target")
    for target_wiki in target_wikis or []:
        if target_wiki and target_wiki != "None":
            print(f" > Ingesting Wiki: '{target_wiki}'...")
            all_grouped_by_header = vector_db.ingest_wikipedia(query=target_wiki)

    if not all_grouped_by_header:
        return {
            "error": f"Ingestion failed. Plan was generated ({target_paper}), but no data was loaded."
        }

    print(f"---KNOW-IT-ALL: FINISHED. {len(all_grouped_by_header)} SECTIONS LOADED---")

    # Return the data to populate the state
    return {"raw_sections": all_grouped_by_header}


def planner_node(state: AgentState):
    """
    The Editor-in-Chief.
    Reads the raw data and creates a 'Story Arc' (Table of Contents).
    """

    if state.get("error"):
        print(f"\n!!! SYSTEM HALT DUE TO ERROR: {state['error']} !!!")
        return "finish"

    print("---PLANNER: CREATING STORY ARC---")

    # Flatten inputs for analysis
    raw_headers = f"ALL HEADINGS: {list(state["raw_sections"].keys())}"

    raw_sections_headings_with_stringed = {
        k: str(v) for k, v in state["raw_sections"].items()
    }
    raw_sections_headings_with_context = {
        k: v[: len(v) // 10] for k, v in raw_sections_headings_with_stringed.items()
    }
    raw_preview_str = str(raw_sections_headings_with_context)
    raw_preview = raw_preview_str  # f"{raw_preview_str[:5000]} ... {raw_preview_str[5000:]}" # Truncate to avoid context limit if huge

    user_query = state["user_query"]

    if user_query:
        user_query = f"Focus on: '{user_query}'"
        print(user_query)
    else:
        user_query = ""

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are the Editor-in-Chief of Distill.pub.

        YOUR GOAL:
        Analyze the provided raw data dictionary and create a "Story Arc" for a blog post.
        The blog must explain the concepts clearly, using storytelling techniques.
        {user_query}

        db_path: {db_path}

        INPUT STRUCTURE:
        The input is a dictionary containing text sections, tables, and abstract data.


        YOUR OUTPUT:
        Generate a JSON list of "Chapters". Each chapter must have:
        1. 'title': Catchy title.
        2. 'goal': The narrative goal.
        3. 'data_requirements': Specific keys or topics to look for in the raw data. If none, put "None".
        4. 'visual_requirements': A descriptions of an interactive visualization to build. If none, put "None".

        CRITICAL:
        - The story must flow: Intro -> Core Concept -> Deep Dive/Data -> Conclusion.
        - Plan for at least 5-15 chapters.
        - Ensure at least one chapter focuses heavily on the DATA.
        - **VISUALS:** We want a highly visual blog. Plan for numerous visuals in *EVERY* chapter. If no data exists for a chapter, request a "Conceptual Diagram" .

        Output format: JSON ONLY.
        {{
            "blog_title": "The Overall Title",
            "chapters": [
                {{ "chapter_id": 1, "title": "...", "goal": "...", "data_requirements": "...", "visual_requirements": "..." }},
                ...
            ]
        }}
        """,
            ),
            (
                "user",
                "ALL Heading keys: {raw_headers}\nRaw Data Preview: {raw_preview}",
            ),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm_flash, [query_vector_db], prompt)
    executor = AgentExecutor(agent=agent, tools=[query_vector_db], verbose=True)

    # chain = prompt | llm_flash
    result = executor.invoke(
        {
            "user_query": user_query,
            "raw_headers": raw_headers,
            "raw_preview": raw_preview,
            "db_path": state["db_path"],
        }
    )

    # Parsing logic to handle potential markdown wrapping

    raw_output = result["output"]
    content = ""

    # Check if output is a list (Gemini/Vertex often returns a list of blocks)
    if isinstance(raw_output, list):
        for block in raw_output:
            # Handle dictionary blocks (e.g. {'type': 'text', 'text': '...'})
            if isinstance(block, dict) and "text" in block:
                content += block["text"]
            # Handle direct strings in list
            elif isinstance(block, str):
                content += block
    else:
        # Standard string output
        content = str(raw_output)

    content = content.replace("```json", "").replace("```", "")

    try:
        plan = json.loads(content)
        print(f"\n\n----\nPLAN :\n{plan}\n-----\n\n")
        return {
            "story_title": plan.get("blog_title", "Distill Blog"),
            "story_arc": plan.get("chapters", []),
            "current_chapter_index": 0,
            "finished_chapters": [],
            "coder_attempts": 0,  # Reset attempts
            "critic_feedback": None,
        }
    except Exception as e:
        print(f"Error in Planner: {e}")
        return {
            "error": f"Planner failed to generate arc: {str(e)}",
            "story_arc": [],
            "current_chapter_index": 0,
            "finished_chapters": [],
            "coder_attempts": 0,
        }


def miner_node(state: AgentState):
    """
    The Researcher.
    Extracts data ONLY for the current chapter's requirements.
    """

    if state.get("error"):
        return {}

    try:
        current_idx = state["current_chapter_index"]
        chapter = state["story_arc"][current_idx]

        print(f"---MINER: PROCESSING CHAPTER {current_idx + 1}: {chapter['title']}---")

        if chapter["data_requirements"] == "None":
            return {"current_chapter_data": {}}

        # Contextual flattening
        data_context = (
            query_vector_db.invoke(
                {
                    "query": f"{chapter['title']}: {chapter['goal']}",
                    "db_path": state["db_path"],
                }
            )
            .replace("{", "{{")
            .replace("}", "}}")
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a Data Researcher.

          CHAPTER : {chapter}
          CURRENT CHAPTER GOAL: {goal}
          DATA REQUIREMENTS: {requirements}

          Your task is to scan the content and extract the specific data needed for this chapter.
          If the requirement asks for experimental results or tables, use the Python Tool to parse them via Regex.

          Output the extracted data as a clean string or JSON structure.
          """,
                ),
                ("user", data_context),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        agent = create_tool_calling_agent(llm_flash, [python_repl_tool], prompt)
        executor = AgentExecutor(agent=agent, tools=[python_repl_tool], verbose=True)

        result = executor.invoke(
            {
                "chapter": chapter["title"],
                "goal": chapter["goal"],
                "requirements": chapter["data_requirements"],
            }
        )

        return {"current_chapter_data": {"extracted": result["output"]}}

    except Exception as e:
        print(f"Error in Miner: {e}")
        return {"error": f"Miner failed to extract data: {str(e)}"}


def coder_node(state: AgentState):
    """
    The Visualization Engineer.
    Uses an AgentExecutor to reason about docs before coding.
    """
    if state.get("error"):
        return {}

    current_idx = state["current_chapter_index"]
    chapter = state["story_arc"][current_idx]
    attempts = state.get("coder_attempts", 0)
    feedback = state.get("critic_feedback", None)

    # 1. Check if we need to do anything
    if (
        chapter["visual_requirements"] == "None"
        or "None" in chapter["visual_requirements"]
    ):
        return {"current_chapter_vis": "", "coder_attempts": 0, "critic_feedback": None}

    print(
        f"---CODER (Agent): VISUALIZING CHAPTER {current_idx + 1} (Attempt {attempts + 1})---"
    )

    # 2. Define the Agent Prompt
    # The 'agent_scratchpad' is where the tool input/outputs are automatically stored
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
        You are a Distill.pub Frontend Engineer.
        
        Your Goal: Write a visualization for a specific chapter of a story.
        
        RULES:
        1. Consult the attached tools (D3.js or Three.js docs) if you are unsure about syntax.
        2. If the concept is 2D/Charts -> Use D3.js.
        3. If the concept is 3D/Spatial -> Use Three.js.
        4. Output HTML/JS only. It must be self-contained in <div id='vis_chapter_{current_idx}'>.
        5. DO NOT output markdown text (like "Here is the code"). Just the code block.
        """,
            ),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    # 3. Create the Agent and Executor
    coder_tools = [d3js_documentation_reference, threejs_documentation_reference]
    # This automatically binds tools and handles the ReAct loop
    agent = create_tool_calling_agent(llm_flash, coder_tools, prompt)

    executor = AgentExecutor(
        agent=agent,
        tools=coder_tools,
        verbose=True,  # Useful to see it thinking/calling tools in logs
        max_iterations=5,  # Prevent infinite loops
        handle_parsing_errors=True,  # Auto-recover if the LLM messes up tool syntax
    )

    # 4. Prepare the Input String
    extracted_data = state["current_chapter_data"].get("extracted", "No data")

    instruction_prefix = ""
    if feedback:
        instruction_prefix = f"""
        !!! CRITICAL FIX REQUIRED !!!
        Previous Attempt Rejected.
        CRITIC FEEDBACK: "{feedback}"
        Fix these specific errors.
        """

    user_input = f"""
    TARGET DIV ID: vis_chapter_{current_idx}
    CHAPTER GOAL: {chapter['goal']}
    VISUALIZATION IDEA: {chapter['visual_requirements']}
    DATA AVAILABLE: {extracted_data}
    
    {instruction_prefix}
    
    Task: Write the code.
    """

    # 5. Invoke the Agent
    try:
        result = executor.invoke({"input": user_input, "current_idx": current_idx})

        # AgentExecutor returns a dict usually containing 'input' and 'output'
        raw_output = result["output"]
        content = ""

        # Check if output is a list (Gemini/Vertex often returns a list of blocks)
        if isinstance(raw_output, list):
            for block in raw_output:
                # Handle dictionary blocks (e.g. {'type': 'text', 'text': '...'})
                if isinstance(block, dict) and "text" in block:
                    content += block["text"]
                # Handle direct strings in list
                elif isinstance(block, str):
                    content += block
        else:
            # Standard string output
            content = str(raw_output)

        # Cleanup markdown formatting if the agent added it
        clean_code = (
            content.replace("```html", "")
            .replace("```javascript", "")
            .replace("```", "")
        )

        return {"current_chapter_vis": clean_code, "coder_attempts": attempts + 1}

    except Exception as e:
        print(f"Agent Execution Failed: {e}")
        return {"error": str(e)}


def critic_node(state: AgentState):
    """
    The Critic (QA).
    Simulates execution and checks for syntax/logic errors.
    """
    if state.get("error"):
        return {}

    vis_code = state.get("current_chapter_vis", "")
    current_idx = state["current_chapter_index"]

    # If no code was generated (not required), auto-approve
    if not vis_code or len(vis_code) < 10:
        return {"critic_feedback": None}

    print(f"---CRITIC: REVIEWING CODE FOR CHAPTER {current_idx + 1}---")

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a Senior QA Engineer and Code Critic.
        Your job is to statically analyze HTML/JavaScript (D3.js/Three.js) code.

        You must "mentally execute" the code and look for:
        1. **Selector Errors**: Does it select the correct ID? (Expected: #vis_chapter_{current_idx})
        2. **Syntax Errors**: Unclosed brackets, missing semicolons, invalid D3 chaining.
        3. **Logic Errors**: Trying to access undefined variables.
        4. **Emptiness**: Does the code actually draw nothing?

        Response Format: JSON ONLY
        {{
            "status": "APPROVE" or "REJECT",
            "feedback": "Short explanation of what is wrong (if REJECT). Otherwise empty string."
        }}
        """,
            ),
            (
                "user",
                "Target ID: #vis_chapter_{current_idx}\n\nCODE TO REVIEW:\n{vis_code}",
            ),
        ]
    )

    chain = prompt | llm_flash
    result = chain.invoke({"current_idx": current_idx, "vis_code": vis_code})

    try:
        content = result.content.replace("```json", "").replace("```", "")
        review = json.loads(content)

        if review["status"] == "APPROVE":
            print("   ✅ Critic Approved")
            return {"critic_feedback": None}  # None implies success
        else:
            print(f"   ❌ Critic Rejected: {review['feedback']}")
            return {"critic_feedback": review["feedback"]}

    except Exception as e:
        print(f"Critic parsing error: {e}")
        # If critic fails to parse, we usually let it pass to avoid blocking,
        # or force a retry. Here we let it pass.
        return {"critic_feedback": None}


async def video_agent_node(state: AgentState):
    """
    The Videographer.
    Finds and downloads a relevant video for the current chapter using the custom agent.
    """
    if state.get("error"):
        return {}

    try:
        current_idx = state["current_chapter_index"]
        chapter = state["story_arc"][current_idx]

        print(f"---VIDEO AGENT: LOOKING FOR CLIPS FOR '{chapter['title']}'---")

        # 1. Initialize the custom agent
        video_tools = await video_client.get_tools()
        agent = create_agent(
            model=llm_flash,
            tools=video_tools,
            system_prompt="""You are a scientific video creation assistant. 
                             Create a video according to the user query.
                             Only make videos if the CONCEPT is scientific other wise return 'None'.""",
        )

        # 2. Formulate the query
        query = f"CONCEPT: {chapter['title']} - {chapter['goal']}"

        # 3. Invoke the agent (using ainvoke as per your snippet, but we must await it)
        response = await agent.ainvoke(
            {"messages": [{"role": "user", "content": query}]}
        )

        video_filename = f"video_chapter_{current_idx}.mp4"
        video_path = None
        os.makedirs(os.path.join(state["output_dir"], "videos"), exist_ok=True)

        try:
            # 4. Extract URL using the specific logic from your snippet

            # Locate the message containing the tool output (Video Search Result)

            last_msg = response["messages"][2]
            content_to_parse = last_msg.content

            if isinstance(content_to_parse, str):
                if content_to_parse in ["None", "'None'"]:
                    return {"current_chapter_video": None}

                # Sometimes the model wraps it in markdown blocks
                clean_content = (
                    content_to_parse.replace("```json", "").replace("```", "").strip()
                )

                video_url = eval(json.loads(clean_content)[0])["video"]["url"]

                print(f" > Found Video URL: {video_url}")

                # 5. Download
                video_path = os.path.join(
                    os.path.join(state["output_dir"], "videos"), video_filename
                )
                try:
                    # 3. Download directly
                    urllib.request.urlretrieve(video_url, video_path)
                    print("Download complete!")
                except Exception as e:
                    print(f"Error: {e}")

                print(f" > Download complete: {video_path}")

        except Exception as e:
            print(f" > Video extraction/download failed: {e}")
            video_path = None

        return {"current_chapter_video": video_path}

    except Exception as e:
        print(f"Error in Video Agent: {e}")
        return {"current_chapter_video": None}


def writer_node(state: AgentState):
    """
    The Storyteller.
    Writes the specific chapter, weaving in the data and visual.
    """
    if state.get("error"):
        return {}

    try:
        current_idx = state["current_chapter_index"]
        chapter = state["story_arc"][current_idx]

        print(f"---WRITER: DRAFTING CHAPTER {current_idx + 1}---")

        data = state["current_chapter_data"].get("extracted", "")
        vis = state["current_chapter_vis"]
        video_path = state.get("current_chapter_video")

        # Determine if visual exists to instruct the writer properly
        visual_instruction = "NO visual available for this chapter."
        if vis and len(vis) > 50:
            visual_instruction = "An interactive visualization IS available. You MUST insert the placeholder `{{INSERT_VISUAL_HERE}}` in the text where it fits best."

        video_instruction = "NO video available."
        if video_path:
            video_instruction = f"A video file has been downloaded to '{video_path}'. You MUST insert the placeholder `{{INSERT_VIDEO_HERE}}` where a video demonstration would be helpful."

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a Science Writer.

          Write the content for ONE chapter of a blog post.

          Title: {title}
          Goal: {goal}

          Instructions:
          1. Write in clear, engaging HTML (<p>, <h3>, <ul>).
          2. Explain the concepts simply (Feynman style).
          3. If there is data, reference it specifically.
          4. If there is a visualization code provided, INSERT the placeholder `{{INSERT_VISUAL_HERE}}` exactly where it should appear in the flow.
          5. Do not write the whole blog, JUST this chapter.
          6. Use simple language and talk like you are telling a story.
          7. VISUAL STATUS: {visual_instruction}.
          8. Use LaTeX formatting for math (e.g., $d_model$, $N=6$)
          9. **CRITICAL FORMATTING RULE:** Do NOT use Markdown for bolding or italics (like **text** or *text*). Browsers will not render this. YOU MUST USE HTML TAGS: <b>bold</b>, <i>italics</i>, <strong>strong</strong>.
        10. VIDEO STATUS: {video_instruction}
          """,
                ),
                ("user", "Data Context: {data}"),
            ]
        )

        chain = prompt | llm_creative
        result = chain.invoke(
            {
                "title": chapter["title"],
                "goal": chapter["goal"],
                "data": str(data),
                "visual_instruction": visual_instruction,
                "video_instruction": video_instruction,
            }
        )

        # Inject the visual code immediately
        chapter_content = result.content
        if vis and len(vis) > 50:
            if "{{INSERT_VISUAL_HERE}}" in chapter_content:
                chapter_content = chapter_content.replace(
                    "{{INSERT_VISUAL_HERE}}", f"<div class='vis-wrapper'>{vis}</div>"
                )
            else:
                chapter_content += f"\n<div class='vis-wrapper'>{vis}</div>"

        # 2. Inject Video Tag (NEW)
        if video_path:
          # Use relative path for HTML portability
          video_filename = os.path.basename(video_path)
          relative_video_path = f"videos/{video_filename}"
          
          video_html = f"""
          <figure>
              <video width="100%" controls>
                  <source src="{relative_video_path}" type="video/mp4">
                  Your browser does not support the video tag.
              </video>
              <figcaption>Video resource for {chapter['title']}</figcaption>
          </figure>
          """
          if "{{INSERT_VIDEO_HERE}}" in chapter_content:
              chapter_content = chapter_content.replace("{{INSERT_VIDEO_HERE}}", video_html)
          else:
              # If LLM forgot to place it, append to bottom
              chapter_content += video_html

        # Wrap in a section tag
        full_chapter_html = f"<section id='chapter-{current_idx}'><h2>{chapter['title']}</h2>{chapter_content}</section>"

        # Append to finished chapters
        current_finished = state.get("finished_chapters", [])
        current_finished.append(full_chapter_html)

        return {
            "finished_chapters": current_finished,
            "current_chapter_index": current_idx + 1,  # Increment for next loop
            # Reset critic/coder state for the NEXT chapter
            "coder_attempts": 0,
            "critic_feedback": None,
        }

    except Exception as e:
        print(f"Error in Writer: {e}")
        return {"error": f"Writer failed to write chapter: {str(e)}"}


def router_node(state: AgentState):
    """
    The Traffic Controller.
    Checks if we have processed all chapters in the arc.
    """
    # IMMEDIATE STOP if error is present
    if state.get("error"):
        print(f"\n!!! SYSTEM HALT DUE TO ERROR: {state['error']} !!!")
        return "finish"

    ## write out the current blog progress
    title = state.get("story_title")
    chapters = state.get("finished_chapters", [])
    filename = save_blog(title, chapters, outputs_dir=state["outputs_dir"])
    print(f"---BLOG PROGRESS SAVED: {len(chapters)} chapters done. SAVED IN {filename}---")

    current_idx = state["current_chapter_index"]
    total_chapters = len(state["story_arc"])

    if current_idx < total_chapters:
        return "continue"
    else:
        return "finish"


def critic_router(state: AgentState):
    """
    Decides if we retry coding or move to writing.
    """
    feedback = state.get("critic_feedback")
    attempts = state.get("coder_attempts", 0)

    # If no feedback, it was approved
    if not feedback:
        return "approve"

    # If too many attempts, force move on (to prevent infinite loops)
    if attempts >= 3:
        print("---CRITIC: TOO MANY RETRIES, SKIPPING VISUAL---")
        # We wipe the visual so the writer doesn't include broken code
        state["current_chapter_vis"] = ""
        return "approve"  # Move to writer, but without the visual

    return "reject"  # Go back to coder


# --- GRAPH CONSTRUCTION ---
def build_workflow():

    workflow = StateGraph(AgentState)

    workflow.add_node("know_it_all", know_it_all_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("miner", miner_node)
    workflow.add_node("coder", coder_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("video_agent", video_agent_node)
    workflow.add_node("writer", writer_node)

    # Entry
    workflow.set_entry_point("know_it_all")

    # Logic
    workflow.add_edge("know_it_all", "planner")
    workflow.add_edge("planner", "miner")  # Start the loop
    workflow.add_edge("miner", "video_agent") 
    workflow.add_edge("video_agent", "coder")
    workflow.add_edge("coder", "critic")  # Coder sends to Critic

    # Conditional Edge for Critic
    workflow.add_conditional_edges(
        "critic", critic_router, {"approve": "writer", "reject": "coder"}
    )

    # Conditional Loop
    workflow.add_conditional_edges(
        "writer",
        router_node,
        {"continue": "miner", "finish": END},  # Loop back for next chapter  # Done
    )

    return workflow.compile()


async def run_agent(user_query: str, outputs_dir: str, db_path: Optional[str] = None, raw_sections: Dict[str, Any] = None) -> str:
    db_path = db_path or os.path.join(outputs_dir, f"my_rag_data_{uuid.uuid4()}")
    initial_state: AgentState = {
        "raw_sections": raw_sections,
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
        "outputs_dir": outputs_dir,
        "current_chapter_video": None,  # Initialize with no video URL
    }
    app = build_workflow()
    final_state = await app.ainvoke(initial_state, config={"recursion_limit": 100})

    title = final_state.get("story_title", user_query)
    chapters = final_state.get("finished_chapters", [])
    filename =save_blog(title, chapters, outputs_dir=outputs_dir)

    # deltete the vector db folder to save space
    if os.path.exists(db_path):
        shutil.rmtree(db_path)

    return filename


async def run_agent_with_pdf(pdf_path: str, outputs_dir: str) -> str:
    db_path = os.path.join(outputs_dir, f"my_rag_data_{uuid.uuid4()}")
    # Ingest via MCP server tool
    store = DoclingVectorStore(db_path=db_path)
    grouped = store.ingest_pdf(pdf_path)

    # Use filename stem as query topic
    return await run_agent("", outputs_dir, db_path=db_path, raw_sections=grouped)
