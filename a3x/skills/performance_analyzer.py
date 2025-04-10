# skills/performance_analyzer.py
import logging
from typing import List, Dict, Any

# Use absolute imports
# from a3x.core.skill_registry import register_skill # Keep commented if not used
from a3x.core.tools import skill
from a3x.core.skills_utils import create_skill_response

logger = logging.getLogger(__name__)

# Define o nome da skill de forma consistente
SKILL_NAME = "performance_analyzer"


class PerformanceAnalyzerSkill:
    """
    Skill to analyze the performance of published products based on sales data.
    Provides simple directives based on identifying top and bottom performers.
    """

    def __init__(self, revenue_threshold_high=15.0, sales_threshold_low=2):
        """
        Initializes the analyzer with thresholds for identifying high/low performers.

        Args:
            revenue_threshold_high (float): Minimum revenue to be considered a top performer.
            sales_threshold_low (int): Maximum sales count to be considered a low performer (if revenue is also low).
        """
        self.revenue_threshold_high = revenue_threshold_high
        self.sales_threshold_low = sales_threshold_low
        logger.info(
            f"Initialized PerformanceAnalyzerSkill with thresholds: High Revenue >= ${revenue_threshold_high}, Low Sales <= {sales_threshold_low}"
        )

    @skill(
        name=SKILL_NAME,
        description="Analyzes sales data to provide directives for future content strategy.",
        parameters={
            "sales_data": (
                List[Dict[str, Any]],
                "A list of sales data records, typically from SalesFetcherSkill.",
            )
        },
    )
    def execute(self, sales_data: List[Dict[str, Any]]) -> dict:
        """
        Analyzes a list of sales data records to determine future content strategy.
        THIS is the main entry point for the skill.

        Identifies the best performing product (by revenue) and the worst
        performing product (by lowest sales, potentially filtered by low revenue).

        Args:
            sales_data (List[Dict[str, Any]]): The list of sales records.
                Expected format per record: {"title": str, "sales": int, "revenue": float, ...}

        Returns:
            dict: A dictionary containing the analysis status, directives, and summary.
                  Example Directives: ["Create more content like 'Top Product'", "Avoid content like 'Worst Product'"]
        """
        logger.info(f"Executing {SKILL_NAME} skill: execute")

        if not isinstance(sales_data, list) or not sales_data:
            logger.warning("execute called with empty or invalid sales data.")
            return create_skill_response(
                status="error",
                action=f"{SKILL_NAME}_failed_no_data",
                error_details="Input sales_data was empty or not a list.",
                message="Cannot analyze performance without valid sales data.",
            )

        # --- Basic Analysis Logic ---
        # Validate expected keys in the first record (assuming consistent structure)
        required_keys = {"title", "sales", "revenue"}
        if not required_keys.issubset(sales_data[0].keys()):
            logger.error(
                f"Sales data records are missing required keys: {required_keys - set(sales_data[0].keys())}"
            )
            return create_skill_response(
                status="error",
                action=f"{SKILL_NAME}_failed_invalid_format",
                error_details=f"Sales data records missing required keys: {required_keys}",
                message="Invalid format for sales data records.",
            )

        # Sort by revenue (descending) to find the top performer
        try:
            # Filter out any potential non-numeric revenues before sorting
            valid_sales_data = [
                record
                for record in sales_data
                if isinstance(record.get("revenue"), (int, float))
                and isinstance(record.get("sales"), int)
            ]
            if not valid_sales_data:
                raise ValueError(
                    "No valid numeric sales/revenue data found after filtering."
                )

            sorted_by_revenue = sorted(
                valid_sales_data, key=lambda x: x.get("revenue", 0.0), reverse=True
            )
            top_performer = sorted_by_revenue[0]

            # Sort by sales (ascending) to find potential low performers
            sorted_by_sales = sorted(valid_sales_data, key=lambda x: x.get("sales", 0))
            # Find the lowest seller that also meets the low threshold criteria
            worst_performer = None
            for record in sorted_by_sales:
                # Consider it "worst" if sales are low AND revenue isn't accidentally high (e.g., one sale of high price item)
                if (
                    record.get("sales", 0) <= self.sales_threshold_low
                    and record.get("revenue", 0) < self.revenue_threshold_high
                ):
                    worst_performer = record
                    break
            # If no clear "worst" found by threshold, pick the absolute lowest seller
            if worst_performer is None and sorted_by_sales:
                worst_performer = sorted_by_sales[0]

        except KeyError as e:
            logger.error(
                f"Missing key '{e}' in sales data during analysis.", exc_info=True
            )
            return create_skill_response(
                status="error",
                action=f"{SKILL_NAME}_failed_processing",
                error_details=f"Missing key: {e}",
                message="Error processing sales data records.",
            )
        except ValueError as e:
            logger.error(f"Error during data validation or sorting: {e}", exc_info=True)
            return create_skill_response(
                status="error",
                action=f"{SKILL_NAME}_failed_processing",
                error_details=str(e),
                message=f"Error processing sales data: {e}",
            )
        except Exception as e:
            logger.error(
                f"Unexpected error during performance analysis: {e}", exc_info=True
            )
            return create_skill_response(
                status="error",
                action=f"{SKILL_NAME}_failed_internal_error",
                error_details=str(e),
                message=f"Internal error during analysis: {e}",
            )

        # --- Generate Directives ---
        directives = []
        analysis_summary = ""

        if top_performer:
            top_title = top_performer.get("title", "Unknown Title")
            top_revenue = top_performer.get("revenue", 0.0)
            analysis_summary += (
                f"Top performer: '{top_title}' (Revenue: ${top_revenue:.2f}). "
            )
            # Add directive only if revenue meets a certain threshold? Or always praise the top?
            if top_revenue >= self.revenue_threshold_high:
                directives.append(
                    f"Create more content like '{top_title}' (High Revenue). Focus on this niche."
                )
            else:
                directives.append(
                    f"Continue exploring content like '{top_title}', as it's the current top performer."
                )

        # Add directive for worst performer only if it's different from the top performer and exists
        if worst_performer and worst_performer != top_performer:
            worst_title = worst_performer.get("title", "Unknown Title")
            worst_sales = worst_performer.get("sales", 0)
            analysis_summary += (
                f"Low performer: '{worst_title}' (Sales: {worst_sales})."
            )
            directives.append(
                f"Avoid content similar to '{worst_title}' (Low Sales/Revenue). Consider alternative topics."
            )
        elif worst_performer and worst_performer == top_performer:
            analysis_summary += (
                "Only one product type analyzed or all performed similarly."
            )
            directives.append("Need more diverse product data for clearer directives.")
        else:
            analysis_summary += (
                "Could not identify a distinct low performer based on criteria."
            )

        logger.info(f"Analysis complete. Directives: {directives}")

        return create_skill_response(
            status="success",
            action=f"{SKILL_NAME}_analysis_completed",
            data={
                "directives": directives,
                "analysis_summary": analysis_summary.strip(),
                "top_performer": top_performer,
                "worst_performer_candidate": worst_performer,
            },
            message="Performance analysis completed successfully.",
        )


# Example Usage (if run directly)
# if __name__ == '__main__':
#     logging.basicConfig(level=logging.INFO)
#     analyzer = PerformanceAnalyzerSkill()
#
#     # Sample sales data (mimicking output from SalesFetcherSkill)
#     sample_data = {
#         "status": "success",
#         "action": "sales_fetcher_simulation_completed",
#         "data": {
#             "sales_records": [
#                 {"product_id": "sim_checklist_567", "title": "Checklist de automação com IA local", "sales": 10, "revenue": 25.50, "fetch_timestamp": "..."},
#                 {"product_id": "sim_prompt_pac_123", "title": "Prompt Pack para criadores de conteúdo", "sales": 25, "revenue": 110.75, "fetch_timestamp": "..."},
#                 {"product_id": "sim_template_n_890", "title": "Template Notion para organização pessoal", "sales": 5, "revenue": 45.00, "fetch_timestamp": "..."},
#                 {"product_id": "sim_guia_de_co_456", "title": "Guia de configuração de LLM local", "sales": 1, "revenue": 6.50, "fetch_timestamp": "..."},
#                 {"product_id": "sim_ebook:_pri_789", "title": "Ebook: Primeiros passos com Agentes Autônomos", "sales": 15, "revenue": 160.00, "fetch_timestamp": "..."}
#             ]
#         },
#         "message": "..."
#     }
#
#     print("\n--- Analyzing Sample Data ---")
#     analysis_result = analyzer.execute(sales_data=sample_data['data']['sales_records'])
#     import json
#     print(json.dumps(analysis_result, indent=2))
#
#     print("\n--- Analyzing Empty Data ---")
#     empty_result = analyzer.execute(sales_data=[])
#     print(json.dumps(empty_result, indent=2))
#
#     print("\n--- Analyzing Invalid Format Data ---")
#     invalid_result = analyzer.execute(sales_data=[{"name": "Product A", "value": 10}])
#     print(json.dumps(invalid_result, indent=2))
