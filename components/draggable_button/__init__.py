import streamlit.components.v1 as components
from pathlib import Path

# Get the directory of this file
_RELEASE = True
COMPONENT_DIR = Path(__file__).parent

# Load the HTML
def draggable_button():
    """
    Render a draggable floating button that toggles the chatbot.
    Returns True when clicked, None otherwise.
    """
    html_path = COMPONENT_DIR / "index.html"
    with open(html_path, "r") as f:
        html_content = f.read()
    
    # Use components.html to embed the custom HTML/CSS/JS
    result = components.html(html_content, height=0, width=70)
    
    return result == "clicked"
