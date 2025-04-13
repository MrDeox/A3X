import json
import random
import logging
import argparse
from pathlib import Path
from typing import Optional, Dict
from a3x.core.skills import skill
from a3x.core.context import Context

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Placeholder Data (Simulated LLM Output) ---
FAKE_AUTHORS = ["Alex Thornton", "Sarah Chen", "Marcus Bellwether", "Jasmine Kaur", "Leo Maxwell", "Digital Nomad Press", "NicheMaster Guides"]

# --- Skill Function ---
@skill(
    name="generate_ebook_from_niche",
    description="Generates the complete markdown content for an ebook based on a given niche and optional outline.",
    parameters={
        "context": {"type": Context, "description": "The execution context provided by the agent."},
        "niche_topic": {"type": str, "description": "The central topic or niche for the ebook."},
        "target_audience": {"type": str, "description": "The intended audience for the ebook."},
        "num_chapters": {"type": int, "default": 5, "description": "The desired number of chapters (default: 5)."},
        "chapters_outline": {"type": Optional[Dict[str, str]], "default": None, "description": "Optional pre-defined outline as a dictionary {chapter_title: chapter_description}."},
        "writing_style": {"type": Optional[str], "default": "informative and engaging", "description": "Desired writing style (e.g., 'conversational', 'academic')."}
    }
)
def generate_ebook_from_niche(
    context: Context,
    niche_topic: str,
    target_audience: str,
    num_chapters: int = 5,
    chapters_outline: Optional[Dict[str, str]] = None,
    writing_style: Optional[str] = "informative and engaging"
) -> dict:
    """Generates placeholder e-book content based on niche information."""
    logger.info(f"Generating e-book content. Niche: '{niche_topic[:50]}...', Audience: '{target_audience[:50]}...', Chapters: {num_chapters}")

    # --- LLM Simulation - Replace with actual LLM calls --- 
    # TODO: Integrate with a real Language Model (e.g., OpenAI API, local model via API) 
    # The following is placeholder generation.

    # Generate Placeholder Title
    # Simple approach: Extract a key concept from the niche summary
    # A real LLM would do much better based on context.
    placeholder_topic = "Your Niche Topic" # Default topic
    try:
        # Attempt a very basic keyword extraction (replace with proper NLP/LLM call)
        keywords_in_summary = [kw for kw in ["Notion", "Etsy", "Imposter Syndrome", "Stoic", "Midjourney", "Obsidian", "No-Code"] if kw.lower() in niche_topic.lower()]
        if keywords_in_summary:
            placeholder_topic = keywords_in_summary[0] 
        else: 
            # Very crude topic guess
             summary_words = niche_topic.split()
             if len(summary_words) > 5:
                 placeholder_topic = " ".join(summary_words[2:5]).replace("for", "").strip().title()
    except Exception as e:
        logger.warning(f"Could not extract simple topic, using default. Error: {e}")
    
    generated_title = f"The Ultimate Guide to {placeholder_topic}"
    logger.info(f"Generated placeholder title: {generated_title}")

    # Generate Placeholder Author
    generated_author = random.choice(FAKE_AUTHORS)
    logger.info(f"Generated placeholder author: {generated_author}")

    # Generate Placeholder Markdown Content
    markdown_content = f"# {generated_title}\n\n" 
    markdown_content += f"*By {generated_author}*\n\n"
    markdown_content += f"## Introduction\n\nWelcome to this guide on {placeholder_topic}. This book is specifically designed for {target_audience}. In the following chapters, we will explore practical steps and insights to help you master this area.\n\n[Placeholder: Add more engaging introduction content here, outlining the book's promise and structure.]\n\n"

    for i in range(1, num_chapters + 1):
        markdown_content += f"## Chapter {i}: [Placeholder Chapter Title {i}]\n\n"
        markdown_content += f"[Placeholder: Detailed content for chapter {i} focusing on a specific aspect of {placeholder_topic}, tailored for {target_audience}. Include practical tips, examples, and actionable advice.]\n\n"
    
    markdown_content += f"## Conclusion\n\nWe've covered the essential aspects of {placeholder_topic}. By applying the strategies discussed, you ({target_audience}) can achieve significant results. Remember to stay consistent and keep learning!\n\n[Placeholder: Add a summary of key takeaways, final encouragement, and potential next steps for the reader.]\n\n"
    logger.info(f"Generated {len(markdown_content.splitlines())} lines of placeholder Markdown content.")

    # Generate Placeholder Description
    generated_description = f"Unlock the secrets of {placeholder_topic}! This practical, no-fluff guide is tailored for {target_audience}. Get actionable insights and start seeing results today. Download your copy now!"
    logger.info(f"Generated placeholder description: {generated_description}")

    # --- End LLM Simulation ---

    # Construct the output JSON
    output = {
        "title": generated_title,
        "author": generated_author,
        "markdown": markdown_content,
        "description": generated_description
    }

    logger.info("E-book content generation (simulated) completed.")
    return output

# --- Main Execution for Standalone Testing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate e-book content (simulated) from a niche.")
    parser.add_argument("--niche-topic", type=str, required=True, help="The central topic or niche for the ebook.")
    parser.add_argument("--target-audience", type=str, required=True, help="The intended audience for the ebook.")
    parser.add_argument("--niche-summary-context", type=str, required=True, help="Summary context of the niche")
    parser.add_argument("--target-audience", type=str, required=True, help="Description of the target audience")
    parser.add_argument("--chapters", type=int, default=8, help="Number of chapters for the e-book")
    parser.add_argument("--output-file", type=str, default=None, help="Optional path to save the JSON output")

    args = parser.parse_args()

    # Example of reading from a file if needed (e.g., if niche context is large)
    # niche_summary = Path(args.niche_summary_context).read_text() if Path(args.niche_summary_context).is_file() else args.niche_summary_context
    # target_audience_text = Path(args.target_audience).read_text() if Path(args.target_audience).is_file() else args.target_audience

    result = generate_ebook(args.niche_summary_context, args.target_audience, args.chapters)

    output_json = json.dumps(result, indent=2, ensure_ascii=False)
    print(output_json)

    if args.output_file:
        try:
            output_path = Path(args.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(output_json)
            logger.info(f"Output saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save output to {args.output_file}: {e}") 