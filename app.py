import os
import shutil
import glob
import gradio as gr
import urllib.parse
import time

# Assuming these imports exist in your project structure
from src.agent_pipeline import run_agent, run_agent_with_pdf


# Get absolute paths to ensure Gradio's file server works correctly
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
DATA_DIR = os.path.join(BASE_DIR, "data")

os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


def list_html_files():
    files = sorted(glob.glob(os.path.join(OUTPUTS_DIR, "*.html")))
    return [os.path.basename(f) for f in files]


def render_iframe(filename):
    if not filename:
        return None

    # Get the absolute path
    file_path = os.path.abspath(os.path.join(OUTPUTS_DIR, filename))

    if not os.path.exists(file_path):
        return f"<p>File not found: {filename}</p>"

    print(f"Serving file: {file_path}")

    # FIX:
    # 1. Use the absolute path.
    # 2. Prepend '/file=' (with the leading slash).
    # 3. Add a timestamp query param (?t=...) to prevent browser caching when you re-run a query.
    iframe_src = f"gradio_api/file/{file_path}?t={str(time.time())}"

    return f"""
    <iframe
        src="{iframe_src}"
        width="100%"
        height="900px"
        style="border:1px solid #eee;border-radius:8px;">
    </iframe>
    """


async def handle_run_query(query):
    if not query or len(query.strip()) == 0:
        return gr.update(choices=list_html_files()), "<p>Please enter a query.</p>", ""

    out_path = await run_agent(query.strip(), OUTPUTS_DIR)
    newest = os.path.basename(out_path)

    # Ensure file exists before rendering (sometimes agents fail silently)
    if not os.path.exists(out_path):
        with open(out_path, "w") as f:
            f.write("<h1>Generated Content</h1>")

    return (
        gr.update(choices=list_html_files(), value=newest),
        render_iframe(newest),
        f"Done. Generated: {newest}",
    )


async def handle_upload_pdf(pdf_file_path):
    if pdf_file_path is None:
        return gr.update(choices=list_html_files()), "<p>Please upload a PDF.</p>", ""

    filename = os.path.basename(pdf_file_path)
    saved_path = os.path.join(DATA_DIR, filename)
    shutil.copyfile(pdf_file_path, saved_path)

    out_path = await run_agent_with_pdf(saved_path, OUTPUTS_DIR)
    newest = os.path.basename(out_path)

    return (
        gr.update(choices=list_html_files(), value=newest),
        render_iframe(newest),
        f"Done. Generated: {newest}",
    )


def handle_select_html(selected):
    if not selected:
        return "<p>Select an HTML file to preview.</p>", None

    preview_html = render_iframe(selected)
    file_path = os.path.join(OUTPUTS_DIR, selected)

    if not os.path.exists(file_path):
        file_path = None

    return preview_html, file_path


def build_ui():
    with gr.Blocks(title="Auto Distill Agent") as demo:
        gr.Markdown(
            """
        # Auto Distill Agent
        - Upload a PDF to ingest and generate a Distill-style article.
        - Or enter a topic query to run the agentic pipeline.
        - Select any generated HTML to preview.
        """
        )

        with gr.Tab("Run from Query"):
            query = gr.Textbox(
                label="Topic Query", placeholder="e.g., Graph Neural Networks"
            )
            run_btn = gr.Button("Run Agent")
            html_list = gr.Dropdown(
                choices=list_html_files(),
                label="Generated HTML Files",
                interactive=True,
            )
            preview = gr.HTML()
            status_q = gr.Markdown(visible=True)

            run_btn.click(
                fn=handle_run_query,
                inputs=[query],
                outputs=[html_list, preview, status_q],
            )
            html_list.change(
                fn=handle_select_html, inputs=[html_list], outputs=[preview]
            )

        with gr.Tab("Run from PDF"):
            pdf = gr.File(label="Upload PDF", file_types=[".pdf"], type="filepath")
            ingest_btn = gr.Button("Ingest + Generate")
            html_list2 = gr.Dropdown(
                choices=list_html_files(),
                label="Generated HTML Files",
                interactive=True,
            )
            preview2 = gr.HTML()
            status_p = gr.Markdown(visible=True)

            ingest_btn.click(
                fn=handle_upload_pdf,
                inputs=[pdf],
                outputs=[html_list2, preview2, status_p],
            )
            html_list2.change(
                fn=handle_select_html, inputs=[html_list2], outputs=[preview2]
            )

        with gr.Tab("Browse Outputs"):
            html_list3 = gr.Dropdown(
                choices=list_html_files(),
                label="Generated HTML Files",
                interactive=True,
            )
            preview3 = gr.HTML()
            download3 = gr.DownloadButton(label="Download HTML", value=None)
            refresh = gr.Button("Refresh List")

            refresh.click(
                lambda: gr.update(choices=list_html_files()), outputs=[html_list3]
            )
            html_list3.change(
                fn=handle_select_html,
                inputs=[html_list3],
                outputs=[preview3, download3],
            )

    return demo


if __name__ == "__main__":
    ui = build_ui()

    # CRITICAL: allowed_paths is required for Gradio to serve the local HTML files
    # We add BASE_DIR to allowed_paths to support serving files via relative paths like /file/outputs/...
    print(f"Allowed paths: {[BASE_DIR, OUTPUTS_DIR, DATA_DIR]}")
    ui.launch(
        server_port=int(os.getenv("PORT", "7860")),
        share=False,
        allowed_paths=[BASE_DIR, OUTPUTS_DIR, DATA_DIR],
    )
