import json
import logging
import argparse
import os
import requests
from pathlib import Path
from html import escape # For basic description HTML escaping
from typing import Optional, List, Dict, Any # Added Optional and List

# Import skill context and decorator
from a3x.core.skills import skill
from a3x.core.context import Context
from a3x.core.config import GUMROAD_API_BASE_URL, GUMROAD_API_KEY

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
GUMROAD_API_ENDPOINT = "https://api.gumroad.com/v2/products"

# --- Skill Function ---
@skill(
    name="publish_to_gumroad",
    description="Publica um produto digital (e-book) na plataforma Gumroad.",
    parameters={
        "name": {"type": str, "description": "O nome do produto (e-book)."},
        "price_cents": {"type": int, "description": "O preço em centavos (ex: 1000 para $10.00)."},
        "description_html": {"type": str, "description": "A descrição do produto em HTML."},
        "file_path": {"type": str, "description": "O caminho para o arquivo do e-book a ser carregado."},
        "cover_image_path": {"type": Optional[str], "description": "O caminho para a imagem de capa (opcional).", "default": None},
        "tags": {"type": Optional[List[str]], "description": "Lista de tags para o produto (opcional).", "default": None},
        "custom_permalink": {"type": Optional[str], "description": "Um permalink customizado (opcional).", "default": None}
        # Adicionar outros parâmetros relevantes da API do Gumroad conforme necessário
    }
)
async def publish_to_gumroad(
    ctx: Context, # Ou SkillContext se preferir
    name: str,
    price_cents: int,
    description_html: str,
    file_path: str,
    cover_image_path: Optional[str] = None,
    tags: Optional[List[str]] = None,
    custom_permalink: Optional[str] = None
) -> Dict[str, Any]:
    """Uses the Gumroad API to create and publish a digital product."""
    logger.info(f"Executando skill publish_to_gumroad: '{name}'")

    # --- 1. Get Access Token ---
    # Use provided api_key or fallback to environment variable
    access_token = GUMROAD_API_KEY or os.environ.get('GUMROAD_ACCESS_TOKEN')
    if not access_token:
        logger.error("API Key/GUMROAD_ACCESS_TOKEN não fornecido ou encontrado.")
        return {"status": "error", "message": "Token de acesso Gumroad não configurado (via parâmetro api_key ou GUMROAD_ACCESS_TOKEN).", "product_url": None}

    # --- 2. Validate Input Files ---
    pdf_path = Path(file_path)
    cover_path = Path(cover_image_path) if cover_image_path else None

    if not pdf_path.is_file():
        logger.error(f"Arquivo principal não encontrado em: {pdf_path}")
        return {"status": "error", "message": f"Arquivo do produto não encontrado: {file_path}", "product_url": None}
    if cover_path and not cover_path.is_file():
        # Allow fallback/optional cover?
        # For now, require it as per input spec.
        logger.error(f"Arquivo de capa não encontrado em: {cover_path}")
        return {"status": "error", "message": f"Imagem de capa não encontrada: {cover_image_path}", "product_url": None}

    # --- 3. Prepare API Data --- 
    price_in_cents = price_cents
    is_published = True  # Assuming the product is always published
    # Basic HTML formatting for description - Gumroad requires this field as HTML.
    # Escape user-provided description to prevent injection issues.
    # Simple paragraph wrapping.
    formatted_description = f"<p>{escape(description_html)}</p>"

    payload = {
        'name': name,
        'description': formatted_description,
        'price': price_in_cents,
        'tags': ",".join(tags) if tags else None, # API expects comma-separated string
        'published': str(is_published).lower(), # API expects 'true' or 'false' as string
        'custom_permalink': custom_permalink,
        # Add other relevant fields if needed (e.g., summary, custom permalink suggestion)
        # 'custom_summary': description[:140], # Example
    }

    headers = {
        'Authorization': f'Bearer {access_token}',
        'Accept': 'application/json', # Expect JSON response
    }

    # --- 4. Prepare Files for Upload ---
    # Use a dictionary for the 'files' argument in requests.post
    files_to_upload = {}
    try:
        files_to_upload['content'] = (pdf_path.name, open(pdf_path, 'rb'), 'application/pdf')
        if cover_path:
            files_to_upload['preview'] = (cover_path.name, open(cover_path, 'rb'), 'image/jpeg') # Assume JPEG, adjust if needed
    except IOError as e:
        logger.error(f"Error opening files for upload: {e}")
        # Ensure files are closed if partially opened (context manager does this)
        return {"status": "error", "message": f"Error opening files: {e}", "product_url": None}

    # --- 5. Make API Request ---
    logger.info(f"Enviando requisição para API Gumroad para produto: {name}")
    product_url = None
    status = "error"
    message = "API request failed."

    try:
        response = requests.post(
            GUMROAD_API_ENDPOINT,
            headers=headers,
            data=payload,
            files=files_to_upload
        )
        response.raise_for_status() # Raise exception for 4xx/5xx errors

        # --- 6. Process Response --- 
        if response.status_code == 201: # 201 Created is the expected success code
            response_data = response.json()
            if response_data.get('success') and 'product' in response_data:
                product_info = response_data['product']
                short_permalink = product_info.get('short_permalink')
                seller_id = product_info.get('seller_id') # Needed to construct full URL potentially
                
                # Gumroad URL format is typically https://<username>.gumroad.com/l/<short_permalink>
                # We don't know the username here, so we return the permalink identifier.
                # The calling process might need to construct the full URL.
                # Or just return the short permalink directly.
                # Let's return the short URL structure Gumroad uses.
                product_url = f"https://gum.co/l/{short_permalink}" if short_permalink else "URL not available in response"
                
                status = "success"
                message = f"Produto '{name}' publicado com sucesso."
                logger.info(message)
                logger.info(f"Product URL (short): {product_url}")
            else:
                message = "API reported success but response format was unexpected."
                logger.warning(f"{message} Response: {response.text[:200]}...")
        else:
             # This case shouldn't be reached due to raise_for_status, but included for robustness
             message = f"Unexpected status code: {response.status_code}. Response: {response.text[:200]}..."
             logger.error(message)

    except requests.exceptions.RequestException as e:
        message = f"Gumroad API request failed: {e}"
        logger.error(message, exc_info=True)
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response status: {e.response.status_code}")
            logger.error(f"Response body: {e.response.text[:500]}...") # Log partial response body
            message += f" (Status: {e.response.status_code})"
        status = "error"
        product_url = None
    except Exception as e:
        message = f"An unexpected error occurred during publishing: {e}"
        logger.error(message, exc_info=True)
        status = "error"
        product_url = None
    finally:
        # Ensure files are closed
        for f_tuple in files_to_upload.values():
            if hasattr(f_tuple[1], 'close'):
                f_tuple[1].close()

    # --- 7. Return Result --- 
    return {
        "status": status,
        "message": message, # Include message for debugging
        "product_url": product_url
    }

# --- Main Execution for Standalone Testing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Publish an e-book to Gumroad.")
    parser.add_argument("--title", type=str, required=True, help="E-book title")
    parser.add_argument("--description", type=str, required=True, help="Marketing description")
    parser.add_argument("--price", type=float, required=True, help="Price in USD (e.g., 7.00)")
    parser.add_argument("--file-path", type=str, required=True, help="Absolute path to the PDF file")
    parser.add_argument("--cover-image-path", type=str, required=True, help="Path to the cover image (e.g., jpg, png)")
    parser.add_argument("--tags", type=str, required=True, help="Comma-separated list of tags")
    parser.add_argument("--visibility", type=str, default="public", choices=['public', 'private'], help="Product visibility")
    parser.add_argument("--output-json-file", type=str, default=None, help="Optional path to save the JSON output")

    args = parser.parse_args()

    # Standalone run still needs the env var check if api_key arg isn't added
    # Or you could add an --api-key argument to the parser
    gumroad_token = os.environ.get('GUMROAD_ACCESS_TOKEN')
    if not gumroad_token:
        print("ERROR: GUMROAD_ACCESS_TOKEN environment variable is not set. Required for standalone test.")
        exit(1)

    tags_list = [tag.strip() for tag in args.tags.split(',') if tag.strip()]

    result = publish_to_gumroad(
        ctx=None, # Provide dummy context for standalone run
        name=args.title,
        price_cents=int(args.price * 100),
        description_html=args.description,
        file_path=args.file_path,
        cover_image_path=args.cover_image_path,
        tags=tags_list,
        custom_permalink=args.visibility
    )

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