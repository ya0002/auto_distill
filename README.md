# Auto Distill Gradio App

A Gradio app that:
- Ingests PDFs using Docling + Chroma,
- Runs an agentic pipeline to plan, mine, code, and write a Distill-style article,
- Saves HTML outputs and allows browsing/preview.

## Project Structure
- `app.py`: Gradio UI (entrypoint for Hugging Face Spaces)
- `src/agent_pipeline.py`: Agent workflow, vector store, and blog saver
- `outputs/`: Generated HTML files
- `data/`: Uploaded PDFs

## Environment
- Requires `GEMINI_KEY` env var if you want Gemini-based generation.
  Without it, the app will still run with a minimal fallback but richer output requires the key.

## Install & Run (Locally)
```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:GEMINI_KEY="YOUR_KEY_HERE"
python app.py
```
Open the Gradio URL printed in the terminal.

## Hugging Face Spaces
- Set Space type to "Gradio".
- Use `app.py` as the entry file.
- Add a Secret: `GEMINI_KEY`.
- The app listens on `PORT` env var (Spaces provides it).

## Notes
- Vector DBs are created under `outputs/` per run.
- HTMLs are placed in `outputs/` and listed in the UI.
- PDF ingestion uses the file name stem as the topic query.
