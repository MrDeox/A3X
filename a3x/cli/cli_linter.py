import os
import ast
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

# --- Configuration --- #
MAX_FILE_LINES = 300
MAX_FUNC_LINES = 80 # Lower threshold for functions
TARGET_FILES_SIZE_CHECK = ["main.py", "commands.py"]
TARGET_FUNCS_RESPONSIBILITY_CHECK = {
    "commands.py": ["run_task", "run_skill_directly"],
    "main.py": ["main_async"],
}

# --- AST Visitor for Analysis --- #

class CliAstVisitor(ast.NodeVisitor):
    """Visits AST nodes to gather info about functions."""

    def __init__(self, target_funcs: List[str]):
        self.target_funcs = target_funcs
        self.results: Dict[str, Dict[str, Any]] = {}
        self._current_func_name: str | None = None

    def _count_lines(self, node: ast.AST) -> int:
        """Calculate the number of lines spanned by a node."""
        if hasattr(node, 'end_lineno') and hasattr(node, 'lineno') and node.end_lineno is not None:
            return node.end_lineno - node.lineno + 1
        return 1 # Default to 1 line if unable to determine range

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._visit_func(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self._visit_func(node)

    def _visit_func(self, node: ast.FunctionDef | ast.AsyncFunctionDef):
        func_name = node.name
        if func_name in self.target_funcs:
            self._current_func_name = func_name
            self.results[func_name] = {
                "line_count": self._count_lines(node),
                "has_parsing": False,
                "has_file_io": False,
                "has_agent_exec": False,
                "has_external_calls": False, # e.g., requests
                "has_complex_logic": False, # Placeholder for deeper analysis
                "calls": set()
            }
            self.generic_visit(node) # Visit children of the function
            self._current_func_name = None
        else:
            # Don't visit children of functions we don't care about
            pass

    def visit_Call(self, node: ast.Call):
        # Track calls made within target functions
        if self._current_func_name:
            call_name = ""
            if isinstance(node.func, ast.Name):
                call_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                # Track attribute calls like console.print, agent.run, json.load, etc.
                try:
                    # Reconstruct the full call path (simplistic)
                    parts = []
                    curr = node.func
                    while isinstance(curr, ast.Attribute):
                        parts.append(curr.attr)
                        curr = curr.value
                    if isinstance(curr, ast.Name):
                        parts.append(curr.id)
                        call_name = ".".join(reversed(parts))
                    else:
                        # Handle more complex cases if needed (e.g., calls on calls)
                        call_name = f"{type(curr).__name__}.{parts[-1]}"
                except Exception:
                    call_name = "complex_call"
            else:
                call_name = f"call_to_{type(node.func).__name__}"

            self.results[self._current_func_name]["calls"].add(call_name)

            # Heuristics for responsibilities based on calls
            if "parse_" in call_name or "argparse" in call_name or call_name in ["json.loads", "yaml.safe_load"]:
                self.results[self._current_func_name]["has_parsing"] = True
            if call_name in ["open", "Path.open"] or ".read" in call_name or ".write" in call_name:
                self.results[self._current_func_name]["has_file_io"] = True
            if "agent." in call_name or "handle_agent_interaction" in call_name:
                 self.results[self._current_func_name]["has_agent_exec"] = True
            if "requests." in call_name:
                 self.results[self._current_func_name]["has_external_calls"] = True

        self.generic_visit(node)

    def visit_If(self, node: ast.If):
        # Example placeholder for complexity check
        # Could check nesting depth later
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try):
        # Example placeholder for complexity check
        self.generic_visit(node)

    def visit_For(self, node: ast.For):
        # Example placeholder for complexity check
        self.generic_visit(node)
    def visit_AsyncFor(self, node: ast.AsyncFor):
        self.generic_visit(node)
    def visit_While(self, node: ast.While):
        self.generic_visit(node)

# --- Main Check Function --- #

def check_cli_modularity() -> List[str]:
    """Performs modularity checks on the a3x/cli package.

    Returns:
        A list of diagnostic strings suggesting improvements.
    """
    diagnostics: List[str] = []
    cli_dir = Path(__file__).parent

    # 1. File Size Check
    for filename in TARGET_FILES_SIZE_CHECK:
        file_path = cli_dir / filename
        if file_path.is_file():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    num_lines = len(lines)
                    if num_lines > MAX_FILE_LINES:
                        diagnostics.append(
                            f"File '{filename}' is too long ({num_lines} lines, max {MAX_FILE_LINES}). Consider splitting into smaller modules."
                        )
            except Exception as e:
                logger.error(f"Could not read file {filename} for size check: {e}")
        else:
             logger.warning(f"Target file for size check not found: {filename}")

    # 2. Function Responsibility Check
    for filename, func_names in TARGET_FUNCS_RESPONSIBILITY_CHECK.items():
        file_path = cli_dir / filename
        if file_path.is_file():
            logger.debug(f"Analyzing functions in: {filename}")
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    source_code = f.read()
                tree = ast.parse(source_code, filename=str(file_path))
                visitor = CliAstVisitor(target_funcs=func_names)
                visitor.visit(tree)

                for func_name, analysis in visitor.results.items():
                    # Check line count
                    if analysis["line_count"] > MAX_FUNC_LINES:
                        diagnostics.append(
                            f"Function '{func_name}' in '{filename}' is too long ({analysis['line_count']} lines, max {MAX_FUNC_LINES}). Consider refactoring."
                        )

                    # Check mixed responsibilities based on heuristics
                    responsibility_flags = [
                        analysis["has_parsing"],
                        analysis["has_file_io"],
                        analysis["has_agent_exec"],
                        analysis["has_external_calls"],
                        # Add more flags here, e.g., complex display logic
                    ]
                    num_responsibilities = sum(1 for flag in responsibility_flags if flag)

                    if num_responsibilities > 1:
                        responsibilities_found = []
                        if analysis["has_parsing"]: responsibilities_found.append("Parsing/Decoding")
                        if analysis["has_file_io"]: responsibilities_found.append("File I/O")
                        if analysis["has_agent_exec"]: responsibilities_found.append("Agent Execution/Interaction")
                        if analysis["has_external_calls"]: responsibilities_found.append("External Network Calls")

                        diagnostics.append(
                            f"Function '{func_name}' in '{filename}' potentially mixes responsibilities: {', '.join(responsibilities_found)}. Consider separating concerns."
                        )
                        # Log the specific calls for debugging
                        logger.debug(f"Calls detected in {func_name}: {analysis['calls']}")

            except SyntaxError as e:
                logger.error(f"Syntax error parsing {filename}: {e}")
                diagnostics.append(f"Could not analyze {filename} due to SyntaxError.")
            except Exception as e:
                logger.error(f"Could not analyze {filename}: {e}", exc_info=True)
                diagnostics.append(f"Could not analyze {filename} due to error.")
        else:
             logger.warning(f"Target file for function analysis not found: {filename}")

    if not diagnostics:
        diagnostics.append("CLI modularity check passed with no major issues detected.")

    return diagnostics

# --- Example Usage --- #
if __name__ == "__main__":
    # Configure basic logging for direct script execution
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    print("Running CLI Modularity Check...")
    results = check_cli_modularity()
    print("\n--- Diagnosis ---:")
    for msg in results:
        print(f"- {msg}") 