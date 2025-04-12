import json
import random
import logging
import argparse
from pathlib import Path

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Potential Niche Database (Simulated) ---
# In a real scenario, this would involve web scraping, trend analysis, API calls etc.
POTENTIAL_NICHES = [
    {
        "id": "notion_templates_freelancers",
        "base_title": "A Minimalist Guide to Notion Templates for Freelancers",
        "description": "Leveraging Notion's flexibility with pre-built templates to streamline freelance workflows, project management, and client communication.",
        "target_audience_segment": "Freelancers, Solopreneurs, Notion users, Productivity enthusiasts",
        "problem_solved": "Disorganization, inefficient workflows, difficulty managing multiple clients/projects.",
        "interest_score": 0.85,
        "gap_opportunity": 0.65,
        "keywords": ["notion templates", "freelance productivity", "project management", "solopreneur tools", "digital organization"],
        "why_it_works_base": "High demand for Notion solutions, addresses specific pain points of a large and growing freelance market."
    },
    {
        "id": "digital_assets_etsy",
        "base_title": "Side Hustle Blueprint: Selling Digital Assets on Etsy",
        "description": "A step-by-step guide to creating and selling profitable digital products (printables, templates, graphics) on Etsy, requiring minimal upfront investment.",
        "target_audience_segment": "Aspiring entrepreneurs, side hustlers, creatives, Etsy sellers, passive income seekers",
        "problem_solved": "Need for low-cost business ideas, desire for passive income streams, uncertainty about selling digital products.",
        "interest_score": 0.90,
        "gap_opportunity": 0.70,
        "keywords": ["etsy digital products", "side hustle ideas", "passive income online", "sell printables", "digital assets"],
        "why_it_works_base": "Taps into the huge 'side hustle' and 'passive income' trends, Etsy is a popular platform, digital products have high margins."
    },
    {
        "id": "imposter_syndrome_creatives",
        "base_title": "Overcoming Imposter Syndrome for Creative Entrepreneurs",
        "description": "Practical strategies and mindset shifts to combat self-doubt and build confidence for artists, writers, designers, and other creative professionals running their own businesses.",
        "target_audience_segment": "Creative entrepreneurs, artists, writers, designers, freelancers facing self-doubt",
        "problem_solved": "Imposter syndrome hindering business growth, lack of confidence, fear of failure.",
        "interest_score": 0.80,
        "gap_opportunity": 0.75,
        "keywords": ["imposter syndrome", "creative entrepreneurship", "mindset for artists", "build confidence", "overcome self-doubt"],
        "why_it_works_base": "Addresses a deep, common emotional pain point for a specific, passionate audience. High potential for connection."
    },
    {
        "id": "stoic_digital_declutter",
        "base_title": "The Stoic's Guide to Digital Decluttering",
        "description": "Applying ancient Stoic principles to modern digital life to reduce distractions, manage information overload, and cultivate focus and tranquility.",
        "target_audience_segment": "Individuals feeling overwhelmed by technology, productivity seekers, mindfulness practitioners, Stoicism enthusiasts",
        "problem_solved": "Digital distraction, information overload, lack of focus, tech-induced anxiety.",
        "interest_score": 0.75,
        "gap_opportunity": 0.80,
        "keywords": ["digital minimalism", "stoicism productivity", "information overload", "reduce screen time", "focus techniques"],
        "why_it_works_base": "Combines the timeless appeal of Stoicism with the modern problem of digital overload. Unique angle."
    },
    {
        "id": "midjourney_prompting",
        "base_title": "Mastering Prompt Engineering for Midjourney Art",
        "description": "Advanced techniques and strategies for crafting effective Midjourney prompts to generate stunning and specific AI art.",
        "target_audience_segment": "AI art enthusiasts, Midjourney users, graphic designers, digital artists",
        "problem_solved": "Difficulty getting desired results from Midjourney, need for advanced prompting skills, desire to create unique AI art.",
        "interest_score": 0.95, # High interest due to AI hype
        "gap_opportunity": 0.55, # Competition likely increasing
        "keywords": ["midjourney prompts", "ai art generation", "prompt engineering", "advanced midjourney", "ai artist tips"],
        "why_it_works_base": "Capitalizes on the massive AI art trend. Provides practical, sought-after skills for a specific tool."
    },
    {
        "id": "obsidian_second_brain",
        "base_title": "Building a Second Brain with Obsidian: A Practical Guide",
        "description": "A step-by-step implementation guide to setting up and utilizing Obsidian.md as a powerful 'second brain' for knowledge management, learning, and creative thinking.",
        "target_audience_segment": "Knowledge workers, students, writers, researchers, Obsidian users, PKM enthusiasts",
        "problem_solved": "Information overload, forgetting ideas, disorganized notes, inefficient learning.",
        "interest_score": 0.88,
        "gap_opportunity": 0.60,
        "keywords": ["obsidian md guide", "building a second brain", "personal knowledge management", "zettelkasten obsidian", "pkm tools"],
        "why_it_works_base": "Addresses the growing need for effective knowledge management. Obsidian has a dedicated, growing user base."
    },
    {
        "id": "nocode_solopreneur",
        "base_title": "The No-Code Solopreneur: Building SaaS without Code",
        "description": "Exploring the landscape of no-code tools and strategies to build, launch, and scale a Software-as-a-Service business as a solo, non-technical founder.",
        "target_audience_segment": "Aspiring SaaS founders, non-technical entrepreneurs, solopreneurs, no-code enthusiasts",
        "problem_solved": "Lack of coding skills as a barrier to starting SaaS, high cost of developers, desire for lean startup methods.",
        "interest_score": 0.82,
        "gap_opportunity": 0.72,
        "keywords": ["no-code saas", "build saas without code", "solopreneur business", "non-technical founder", "no-code tools"],
        "why_it_works_base": "Empowers non-technical individuals to build software businesses, riding the strong no-code movement."
    }
]

# --- Skill Function ---
def scan_niches(language: str, min_interest_score: float, min_gap_opportunity: float, audience_context: str) -> dict:
    """Scans for profitable ultra-niche e-book topics based on input criteria."""
    logger.info(f"Scanning for niches. Language: {language}, Min Interest: {min_interest_score}, Min Gap: {min_gap_opportunity}, Audience Context: '{audience_context}'")

    if language.lower() != 'en':
        logger.warning("This skill is primarily designed for English (en) niches. Results may be limited for other languages.")
        # In a real skill, we might load different niche databases per language

    # Simulate filtering based on scores
    eligible_niches = [
        n for n in POTENTIAL_NICHES
        if n["interest_score"] >= min_interest_score and n["gap_opportunity"] >= min_gap_opportunity
    ]

    if not eligible_niches:
        logger.warning("No niches found matching the criteria after filtering. Returning a default/fallback.")
        # Fallback strategy: maybe return the highest scoring one regardless of gap, or a predefined safe bet.
        # For MVP, let's just pick one randomly from the original list if filtering fails.
        if POTENTIAL_NICHES:
             selected_niche = random.choice(POTENTIAL_NICHES)
        else:
            # Handle empty POTENTIAL_NICHES case
            logger.error("Potential niches database is empty!")
            return { # Return a dummy error structure
                "error": "No potential niches defined.",
                "niche_summary": None,
                "title_suggestion": None,
                "audience": audience_context, # Echo back input audience
                "why_it_works": None,
                "keywords": []
            }
    else:
        # Select the 'best' from the eligible list (e.g., highest combined score)
        # For MVP, let's just pick one randomly from the eligible list
        selected_niche = random.choice(eligible_niches)
        logger.info(f"Selected niche: {selected_niche['id']}")

    # Construct the output JSON
    output = {
        "niche_summary": selected_niche["description"],
        "title_suggestion": selected_niche["base_title"],
        # Refine audience description slightly based on the niche
        "audience": f"{audience_context} (Specifically: {selected_niche['target_audience_segment']})",
        "why_it_works": f"{selected_niche['why_it_works_base']} Solves: {selected_niche['problem_solved']}",
        "keywords": selected_niche["keywords"]
    }

    logger.info("Niche scan completed.")
    return output

# --- Main Execution for Standalone Testing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scan for profitable ultra-niche e-book topics.")
    parser.add_argument("--language", type=str, default="en", help="Target language (e.g., 'en', 'pt-BR')")
    parser.add_argument("--min-interest-score", type=float, default=0.8, help="Minimum interest score (0.0-1.0)")
    parser.add_argument("--min-gap-opportunity", type=float, default=0.6, help="Minimum gap/opportunity score (0.0-1.0)")
    parser.add_argument("--audience", type=str, default="international buyers interested in digital solutions, productivity, side income, personal growth or niche obsessions", help="Description of the target audience")
    parser.add_argument("--output-file", type=str, default=None, help="Optional path to save the JSON output")

    args = parser.parse_args()

    result = scan_niches(args.language, args.min_interest_score, args.min_gap_opportunity, args.audience)

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