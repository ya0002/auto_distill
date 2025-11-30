import sys
import os

# Ensure project root is on sys.path so `utils` can be imported even when running from `tools/`
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from langchain_mcp_adapters.client import MultiServerMCPClient 

video_client = MultiServerMCPClient(
    {
        "math_animator": {
            "transport": "streamable_http",  # HTTP-based remote server
            # Ensure you start your weather server on port 8000
            "url": "https://mcp-1st-birthday-anim-lab-ai.hf.space/gradio_api/mcp/",
        }
    }
)

# video_tools = await video_client.get_tools()
