import os
import glob
import gradio as gr
from src.agent_pipeline import run_agent, run_agent_with_pdf

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


def list_html_files():
    files = sorted(glob.glob(os.path.join(OUTPUTS_DIR, "*.html")))
    return [os.path.basename(f) for f in files]


def read_html(filename):
    path = os.path.join(OUTPUTS_DIR, filename)
    if not os.path.exists(path):
        return f"<p>File not found: {filename}</p>"
    with open(path, "r", encoding="utf-8") as f:
        html = f.read()
    return html


def render_iframe(filename):
    html = read_html(filename)
    html_safe = html.replace('"', '&quot;')
    return f"""
    <iframe srcdoc="{html_safe}" width="100%" height="900px" style="border:1px solid #eee;border-radius:8px;"></iframe>
    """


def handle_run_query(query):
    if not query or len(query.strip()) == 0:
        return gr.update(choices=list_html_files()), "<p>Please enter a query.</p>", ""
    status = "Running agent… This may take a minute."
    out_path = run_agent(query.strip(), OUTPUTS_DIR)
    newest = os.path.basename(out_path)
    return gr.update(choices=list_html_files(), value=newest), render_iframe(newest), f"Done. Generated: {newest}"


def handle_upload_pdf(pdf_file_path):
    if pdf_file_path is None:
        return gr.update(choices=list_html_files()), "<p>Please upload a PDF.</p>", ""
    # pdf_file_path is a path string when using type="filepath"
    filename = os.path.basename(pdf_file_path)
    saved_path = os.path.join(DATA_DIR, filename)
    import shutil
    shutil.copyfile(pdf_file_path, saved_path)
    out_path = run_agent_with_pdf(saved_path, OUTPUTS_DIR)
    newest = os.path.basename(out_path)
    return gr.update(choices=list_html_files(), value=newest), render_iframe(newest), f"Done. Generated: {newest}"


def handle_select_html(selected):
    if not selected:
        return "<p>Select an HTML file to preview.</p>"
    return render_iframe(selected)


def build_ui():
    with gr.Blocks(title="Auto Distill Agent") as demo:
        gr.Markdown("""
        # Auto Distill Agent
        - Upload a PDF to ingest and generate a Distill-style article.
        - Or enter a topic query to run the agentic pipeline.
        - Select any generated HTML to preview.
        """)

        with gr.Tab("Run from Query"):
            query = gr.Textbox(label="Topic Query", placeholder="e.g., Graph Neural Networks")
            run_btn = gr.Button("Run Agent")
            html_list = gr.Dropdown(choices=list_html_files(), label="Generated HTML Files", interactive=True)
            preview = gr.HTML()
            status_q = gr.Markdown(visible=True)

            run_btn.click(fn=handle_run_query, inputs=[query], outputs=[html_list, preview, status_q])
            html_list.change(fn=handle_select_html, inputs=[html_list], outputs=[preview])

        with gr.Tab("Run from PDF"):
            # Use type="filepath" for new Gradio versions
            pdf = gr.File(label="Upload PDF", file_types=[".pdf"], type="filepath")
            ingest_btn = gr.Button("Ingest + Generate")
            html_list2 = gr.Dropdown(choices=list_html_files(), label="Generated HTML Files", interactive=True)
            preview2 = gr.HTML()
            status_p = gr.Markdown(visible=True)

            ingest_btn.click(fn=handle_upload_pdf, inputs=[pdf], outputs=[html_list2, preview2, status_p])
            html_list2.change(fn=handle_select_html, inputs=[html_list2], outputs=[preview2])

        with gr.Tab("Browse Outputs"):
            html_list3 = gr.Dropdown(choices=list_html_files(), label="Generated HTML Files", interactive=True)
            preview3 = gr.HTML()
            refresh = gr.Button("Refresh List")
            refresh.click(lambda: gr.update(choices=list_html_files()), outputs=[html_list3])
            html_list3.change(fn=handle_select_html, inputs=[html_list3], outputs=[preview3])

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(server_port=int(os.getenv("PORT", "7860")), share=False)
