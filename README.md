# Auto Distill

Auto Distill is an AI-powered agentic pipeline designed to generate high-quality, "Distill-style" interactive articles. Whether starting from a simple user query or an uploaded PDF, Auto Distill orchestrates a team of AI agents to research, plan, write, and visualize complex scientific and technical concepts.

[DEMO VIDEO](https://youtu.be/j7Va_BvqJSA?si=psUXyB5kExpwSsaH)

[LIVE DEMO](https://huggingface.co/spaces/MCP-1st-Birthday/auto-distill)

![Demo GIF](auto_distill.gif)

## Features

- **Agentic Workflow**: Powered by [LangGraph](https://langchain-ai.github.io/langgraph/), the system employs specialized agents:
  - **Know-It-All**: Researches topics using Arxiv and Wikipedia.
  - **Planner**: Creates a structured "Story Arc" for the article.
  - **Miner**: Extracts specific data and insights for each chapter.
  - **Coder**: Generates interactive visualizations using D3.js or Three.js. It references a local vector database of D3.js and Three.js documentation to ensure accurate code generation.
  - **Critic**: Reviews and validates the generated code.
  - **Video Agent**: Finds or generates relevant videos using MCP tools.
  - **Writer**: Drafts engaging, educational content in HTML.
- **Interactive Visualizations**: Automatically generates custom D3.js or Three.js visualizations to explain concepts.
- **PDF Ingestion**: Upload your own research papers or documents. The system uses **Docling** with hybrid chunking to intelligently parse and index the content.
- **MCP Integration**: Utilizes the Model Context Protocol (MCP) to connect with external tools like video generators.
- **Gradio UI**: A user-friendly web interface to interact with the system and view results.

## Installation

1.  **Clone the repository:**

    ```bash
    git clone <repository-url>
    cd auto_distill
    ```

2.  **Install dependencies:**

    It is recommended to use a virtual environment.

    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up Environment Variables:**

    You need to set your Google Gemini API key. You can do this by setting an environment variable `GEMINI_KEY`.

    ```bash
    # Linux/macOS
    export GEMINI_KEY="your_gemini_api_key"

    # Windows (PowerShell)
    $env:GEMINI_KEY="your_gemini_api_key"
    ```

## Usage

1.  **Run the application:**

    ```bash
    python app.py
    ```

2.  **Access the UI:**

    Open your web browser and navigate to the URL provided in the terminal (usually `http://127.0.0.1:7860`).

3.  **Generate Articles:**

    *   **Run from Query:** Go to the "Run from Query" tab, enter a topic (e.g., "Graph Neural Networks"), and click "Run Agent".
    *   **Run from PDF:** Go to the "Run from PDF" tab, upload a PDF file, and click "Ingest + Generate".

4.  **View Results:**

    The generated HTML articles will be saved in the `outputs/` directory and can be previewed directly in the "Browse Outputs" tab.

## Project Structure

*   `app.py`: The main entry point for the application, containing the Gradio UI logic.
*   `src/agent_pipeline.py`: Defines the LangGraph workflow, agent nodes, and state management.
*   `tools/`: Contains custom tools and MCP client configurations.
    *   `mcp_tools.py`: Configuration for the Multi-Server MCP Client.
    *   `custom_tools.py`: Custom tools for search, vector DB queries, etc.
*   `utils.py`: Utility functions for file handling and vector store operations.
*   `requirements.txt`: List of Python dependencies.
*   `outputs/`: Directory where generated HTML files and assets are stored.
*   `chroma_db_native/` & `data/`: Directories for local data storage and vector databases.

## Technologies Used

*   **LangChain & LangGraph**: For building the agentic workflow.
*   **Google Gemini**: As the primary LLM for reasoning and content generation.
*   **Gradio**: For the web interface.
*   **ChromaDB**: For vector storage and retrieval.
*   **Docling**: For document parsing and ingestion.
*   **Model Context Protocol (MCP)**: For extensible tool integration.

## License

[MIT License](LICENSE)
