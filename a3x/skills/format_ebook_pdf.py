import json
import logging
import argparse
import io
import re
from pathlib import Path
from typing import Dict, Optional, List
from a3x.core.skills import skill
from a3x.core.context import Context

import markdown2
from xhtml2pdf import pisa

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
OUTPUT_DIR_NAME = "exports/ebooks"

# --- Helper Functions ---
def create_slug(title):
    """Creates a filesystem-safe slug from a title."""
    # Remove special characters, replace spaces with hyphens, lowercase
    slug = re.sub(r'[^\w\s-]', '', title).strip().lower()
    slug = re.sub(r'[\s_]+', '-', slug)
    # Limit length to avoid overly long filenames
    return slug[:80]

def markdown_to_pdf(markdown_content: str, title: str, author: str, output_path: Path):
    """Converts Markdown content to a styled PDF file."""
    logger.info(f"Converting Markdown to PDF: {output_path}")

    # Convert Markdown to HTML
    html_body = markdown2.markdown(markdown_content, extras=["fenced-code-blocks", "tables", "smarty-pants"])

    # Basic HTML structure with minimal CSS for styling
    # Note: CSS support in xhtml2pdf is limited. Complex layouts might require more advanced techniques.
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>{title}</title>
        <style>
            @page {{
                size: a4 portrait;
                @frame content_frame {{
                    left: 50pt; width: 500pt; top: 50pt; height: 700pt;
                }}
            }}
            body {{
                font-family: sans-serif;
                line-height: 1.6;
                color: #333;
            }}
            h1, h2, h3 {{
                color: #000;
                margin-bottom: 0.5em;
                line-height: 1.2;
            }}
            h1 {{
                font-size: 24pt;
                text-align: center;
                margin-top: 100pt; /* Push title down */
                margin-bottom: 10pt;
            }}
            .author {{
                font-size: 14pt;
                text-align: center;
                margin-bottom: 150pt; /* Space after author */
                font-style: italic;
            }}
             h2 {{
                font-size: 18pt;
                margin-top: 2em;
                border-bottom: 1px solid #eee;
                padding-bottom: 0.3em;
            }}
            h3 {{
                font-size: 14pt;
                margin-top: 1.5em;
            }}
            p {{
                margin-bottom: 1em;
            }}
            code {{
                font-family: monospace;
                background-color: #f5f5f5;
                padding: 0.2em 0.4em;
                border-radius: 3px;
            }}
            pre {{
                background-color: #f5f5f5;
                border: 1px solid #ddd;
                padding: 1em;
                overflow-x: auto;
                border-radius: 4px;
            }}
            pre code {{
                 background-color: transparent;
                 padding: 0;
                 border: none;
            }}
             /* Add page break before H2 (chapters) for better structure */
            h2 {{
                 page-break-before: always;
             }}
             /* Avoid page break right after a heading */
            h2, h3 {{
                page-break-after: avoid;
            }}
        </style>
    </head>
    <body>
        <h1>{title}</h1>
        <p class="author">By {author}</p>
        {html_body}
    </body>
    </html>
    """

    # Convert HTML to PDF
    try:
        with open(output_path, "wb") as pdf_file:
            # Encode HTML template to bytes
            source_html = io.BytesIO(html_template.encode('utf-8'))
            
            # Use pisa to generate PDF
            pisa_status = pisa.CreatePDF(
                src=source_html,
                dest=pdf_file,
                encoding='utf-8'
            )
        
        if pisa_status.err:
            logger.error(f"PDF generation failed with errors: {pisa_status.err}")
            # Attempt to clean up potentially corrupted file
            if output_path.exists():
                 output_path.unlink()
            return False
        else:
            logger.info("PDF generated successfully.")
            return True
            
    except Exception as e:
        logger.error(f"Exception during PDF generation: {e}", exc_info=True)
        # Attempt cleanup on generic exception
        if output_path.exists():
             output_path.unlink()
        return False


# --- Skill Function ---
@skill(
    name="format_ebook_pdf",
    description="Formats the generated markdown content into a styled PDF ebook.",
    parameters={
        "context": {"type": Context, "description": "Execution context (not typically used here)."},
        "ebook_markdown": {"type": str, "description": "The complete markdown content of the ebook."},
        "title": {"type": str, "description": "The title of the ebook."},
        "author": {"type": str, "description": "The author of the ebook."},
        "chapters": {"type": List[str], "description": "List of chapter titles extracted from the markdown."},
        "cover_image_path": {"type": Optional[str], "default": None, "description": "Optional path to a cover image file."}
    }
)
def format_ebook_pdf(
    context: Context,
    ebook_markdown: str,
    title: str,
    author: str,
    chapters: List[str],
    cover_image_path: Optional[str] = None
) -> dict:
    """Formats the markdown e-book content into a PDF file."""
    logger.info(f"Formatting PDF for: '{title}' by {author}")

    # Determine project root (assuming this script is in a3x/skills/)
    project_root = Path(__file__).parent.parent.parent 
    output_dir = project_root / OUTPUT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Ensured output directory exists: {output_dir}")

    # Create filename and path
    pdf_slug = create_slug(title)
    pdf_filename = f"{pdf_slug}.pdf"
    output_pdf_path = output_dir / pdf_filename

    # Generate the PDF
    success = markdown_to_pdf(ebook_markdown, title, author, output_pdf_path)

    if not success:
        # Handle PDF generation failure
        logger.error("PDF generation failed. Cannot provide file path.")
        # Returning an error structure might be better in a real implementation
        return {"error": "PDF generation failed", "file_path": None}

    # Return the absolute path
    absolute_path = str(output_pdf_path.resolve())
    output = {
        "file_path": absolute_path
    }

    logger.info(f"PDF formatting completed. Output path: {absolute_path}")
    return output

# --- Main Execution for Standalone Testing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Markdown e-book to formatted PDF.")
    # Input arguments can be file paths or direct strings
    parser.add_argument("--markdown-input", type=str, required=True, help="Markdown content string or path to a .md file")
    parser.add_argument("--title", type=str, required=True, help="E-book title")
    parser.add_argument("--author", type=str, required=True, help="Author name")
    parser.add_argument("--output-json-file", type=str, default=None, help="Optional path to save the JSON output containing the PDF path")

    args = parser.parse_args()

    # Read markdown from file if path is provided
    markdown_input_path = Path(args.markdown_input)
    if markdown_input_path.is_file():
        logger.info(f"Reading Markdown content from file: {markdown_input_path}")
        try:
            markdown_text = markdown_input_path.read_text(encoding='utf-8')
        except Exception as e:
            logger.error(f"Failed to read markdown file {markdown_input_path}: {e}")
            exit(1)
    else:
        logger.info("Using provided string as Markdown content.")
        markdown_text = args.markdown_input

    result = format_ebook_pdf(markdown_text, args.title, args.author)

    output_json = json.dumps(result, indent=2)
    print(output_json)

    if args.output_json_file:
        try:
            output_json_path = Path(args.output_json_file)
            output_json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_json_path, 'w', encoding='utf-8') as f:
                f.write(output_json)
            logger.info(f"Output JSON saved to {output_json_path}")
        except Exception as e:
            logger.error(f"Failed to save output JSON to {args.output_json_file}: {e}") 