# skills/sales_fetcher.py
import logging
import random
from datetime import datetime
from typing import List, Dict, Any

# from core.skill_registry import register_skill
from a3x.core.tools import skill
from a3x.core.skills_utils import create_skill_response

logger = logging.getLogger(__name__)

# Define o nome da skill de forma consistente
SKILL_NAME = "sales_fetcher"


class SalesFetcherSkill:
    """
    Skill to retrieve sales data for published digital products.
    Currently simulates fetching data, mimicking an API call to a platform like Gumroad.
    """

    def __init__(self):
        # Placeholder for potential future initializations (e.g., API keys, platform connections)
        pass

    @skill(
        name=SKILL_NAME,
        description="Fetches sales data (simulated) for published products, typically from platforms like Gumroad.",
        parameters={"platform": (str, "gumroad"), "product_id": (str | None, None)},
    )
    def execute(self, platform: str = "gumroad", product_id: str | None = None) -> dict:
        """
        Simulates fetching sales data for products.
        THIS is the main entry point for the skill.

        If product_id is provided, simulates data for that specific product.
        Otherwise, returns aggregated simulated data for a few predefined products.

        Args:
            platform (str, optional): The platform name (ignored in simulation). Defaults to "gumroad".
            product_id (str | None, optional): Specific product ID (ignored in simulation). Defaults to None.

        Returns:
            dict: A dictionary containing the status and a list of simulated sales data records.
                  Example Record: {"title": "Prompt Pack", "sales": 4, "revenue": 8.00, "fetch_timestamp": "..."}
        """
        logger.info(
            f"Executing {SKILL_NAME} skill: execute (simulation) for platform '{platform}'"
        )

        # --- Simulation Logic ---
        # Simulate data for a few products if no specific ID is given
        simulated_data = []
        if product_id:
            # Simulate data for a specific (fictional) product ID
            logger.info(
                f"Simulating sales data fetch for specific product ID: {product_id}"
            )
            # Generate somewhat random but plausible data
            title = f"Product {product_id[:8]}..."  # Generic title based on ID
            sales = random.randint(0, 50)
            price = round(random.uniform(1.99, 19.99), 2)
            revenue = round(
                sales * price * random.uniform(0.85, 0.95), 2
            )  # Simulate platform fees
            simulated_data.append(
                {
                    "product_id": product_id,
                    "title": title,
                    "sales": sales,
                    "revenue": revenue,
                    "fetch_timestamp": datetime.now().isoformat(),
                }
            )
        else:
            # Simulate aggregated data for a predefined list of products
            logger.info(
                "Simulating aggregated sales data fetch for predefined products."
            )
            predefined_products = [
                ("Checklist de automação com IA local", 2.99),
                ("Prompt Pack para criadores de conteúdo", 4.99),
                ("Template Notion para organização pessoal", 9.99),
                ("Guia de configuração de LLM local", 7.50),
                ("Ebook: Primeiros passos com Agentes Autônomos", 12.00),
            ]
            for title, price in predefined_products:
                sales = random.randint(
                    0, 25
                )  # Lower sales range for individual items in aggregate view
                revenue = round(
                    sales * price * random.uniform(0.8, 0.95), 2
                )  # Simulate fees
                simulated_data.append(
                    {
                        "product_id": f"sim_{title[:10].replace(' ', '_').lower()}_{random.randint(100, 999)}",  # Generate fake ID
                        "title": title,
                        "sales": sales,
                        "revenue": revenue,
                        "fetch_timestamp": datetime.now().isoformat(),
                    }
                )

        logger.debug(f"Generated simulated sales data: {simulated_data}")

        return create_skill_response(
            status="success",
            action=f"{SKILL_NAME}_simulation_completed",
            data={"sales_records": simulated_data},
            message=f"Simulated sales data fetched successfully for {len(simulated_data)} product(s).",
        )


# Example Usage (if run directly)
# if __name__ == '__main__':
#     logging.basicConfig(level=logging.INFO)
#     fetcher = SalesFetcherSkill()
#
#     print("\n--- Fetching Aggregated Data ---")
#     result_agg = fetcher.execute()
#     import json
#     print(json.dumps(result_agg, indent=2))
#
#     print("\n--- Fetching Specific Product Data ---")
#     result_spec = fetcher.execute(product_id="prod_123xyz")
#     print(json.dumps(result_spec, indent=2))
