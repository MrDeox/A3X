# a3x/reflection/structure_reflector.py

import os
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Optional

# Import core components
try:
    from ..core.llm_interface import LLMInterface
    from ..core.config import PROJECT_ROOT
except ImportError as e:
    print(f"[StructureReflector Error] Failed to import core modules: {e}")
    # Keep fallbacks for runtime checks if necessary, but not for type hinting
    LLMInterface = None # Keep for runtime checks if used elsewhere, but not for hinting
    PROJECT_ROOT = Path(__file__).parent.parent.parent # Keep fallback path

logger = logging.getLogger(__name__)

class StructureReflector:
    """Analyzes code structure and modularity based on project manifestos."""

    def __init__(self, llm_interface: LLMInterface, docs_path: str = "docs/manifestos"):
        """
        Initializes the reflector.

        Args:
            llm_interface: An initialized instance of LLMInterface.
            docs_path: Path relative to PROJECT_ROOT containing manifesto .md files.
        """
        if not llm_interface:
            raise ValueError("LLMInterface instance is required.")
        if not PROJECT_ROOT:
             raise ValueError("PROJECT_ROOT configuration is missing.")

        self.llm_interface = llm_interface
        self.manifesto_path = Path(PROJECT_ROOT) / docs_path
        self.heuristics = self.load_manifesto_heuristics(self.manifesto_path)
        if not self.heuristics:
             logger.warning(f"No heuristics loaded from {self.manifesto_path}. Analysis may be ineffective.")

    def load_manifesto_heuristics(self, path: Path) -> str:
        """Loads and concatenates heuristics from all .md files in the specified path."""
        all_heuristics = []
        logger.info(f"Loading manifesto heuristics from: {path}")
        if not path.is_dir():
            logger.error(f"Manifesto directory not found: {path}")
            return ""

        try:
            for md_file in path.glob("*.md"):
                try:
                    with open(md_file, "r", encoding="utf-8") as f:
                        all_heuristics.append(f.read())
                    logger.debug(f"Loaded heuristics from: {md_file.name}")
                except Exception as e:
                    logger.error(f"Failed to read manifesto file {md_file}: {e}")

            if not all_heuristics:
                logger.warning(f"No .md files found or read in {path}")
                return ""

            concatenated = "\n\n---\n\n".join(all_heuristics)
            logger.info(f"Successfully loaded and concatenated heuristics from {len(all_heuristics)} files.")
            return concatenated
        except Exception as e:
            logger.exception(f"Error loading heuristics from directory {path}: {e}")
            return ""

    async def analyze_file_semantically(self, filepath: str | Path) -> str:
        """Analyzes a single Python file against the loaded heuristics using the LLM."""
        file_path = Path(filepath)
        logger.info(f"Analyzing file semantically: {file_path}")

        if not self.heuristics:
            return "Error: No architectural heuristics loaded."
        if not self.llm_interface:
             return "Error: LLMInterface instance not available."

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                file_content = f.read()
        except FileNotFoundError:
            logger.error(f"File not found for analysis: {file_path}")
            return f"Error: File not found - {file_path}"
        except Exception as e:
            logger.error(f"Error reading file {file_path} for semantic analysis: {e}")
            return f"Error reading file: {e}"

        prompt = self._build_semantic_analysis_prompt(file_content)
        messages = [{"role": "user", "content": prompt}]

        try:
            logger.debug(f"Sending semantic analysis request for {file_path.name}...")
            # Correctly consume the async generator
            response_content = ""
            async for chunk in self.llm_interface.call_llm(messages=messages, stream=False):
                response_content += chunk # Accumulate content (should be just one chunk)
                
            if not response_content:
                logger.warning(f"LLM returned empty semantic analysis for {file_path.name}")
                return "Analysis Error: LLM returned empty content."
            
            logger.debug(f"Received semantic analysis for {file_path.name}")
            return response_content.strip()
            
        except Exception as e:
            logger.error(f"LLM call failed during analysis of {file_path.name}: {e}")
            # Optionally log the full traceback
            # logger.exception(f"LLM call traceback for {file_path.name}:") 
            return f"Analysis Error: {e}"

    def _build_semantic_analysis_prompt(self, file_content: str) -> str:
        """Builds the prompt for the LLM to perform semantic analysis."""
        return f"""
Você é um avaliador arquitetural do sistema A³X.

Com base nas diretrizes dos manifestos abaixo, avalie se o código está modular, coeso e com responsabilidades bem separadas.

Responda:
- O módulo segue os princípios de fragmentação e separação?
- Há sinais de acoplamento excessivo?
- Você sugere alguma refatoração?

Diretrizes:
{self.heuristics}

Código:
```python
{file_content}
```
"""

    async def run_project_scan(self, base_path: str = "a3x/cli") -> Dict[str, str]:
        """Scans a directory for .py files and analyzes each semantically."""
        if not PROJECT_ROOT:
            logger.error("PROJECT_ROOT not set, cannot determine scan path.")
            return {"error": "PROJECT_ROOT not configured."}

        scan_dir = Path(PROJECT_ROOT) / base_path
        logger.info(f"Scanning directory for Python files: {scan_dir}")
        results: Dict[str, str] = {}
        analysis_tasks = []

        if not scan_dir.is_dir():
            logger.error(f"Scan directory does not exist or is not a directory: {scan_dir}")
            return {"error": f"Scan directory not found: {scan_dir}"}

        py_files = list(scan_dir.rglob("*.py"))
        if not py_files:
            logger.warning(f"No Python files found in {scan_dir}")
            return {"info": f"No Python files found in {scan_dir}"}

        logger.info(f"Found {len(py_files)} Python files to analyze.")

        # Create analysis tasks
        for py_file in py_files:
            # Use relative path for dictionary key for cleaner output
            relative_path = str(py_file.relative_to(PROJECT_ROOT))
            analysis_tasks.append(
                self.analyze_file_semantically(py_file)
            )

        # Run tasks concurrently
        logger.info("Running semantic analysis tasks concurrently...")
        try:
            analyses = await asyncio.gather(*analysis_tasks, return_exceptions=True)
            logger.info("Semantic analysis tasks completed.")
        except Exception as e:
             logger.exception("Error during asyncio.gather for analysis tasks:")
             return {"error": f"Failed during concurrent analysis execution: {e}"}

        # Collect results
        for i, py_file in enumerate(py_files):
            relative_path = str(py_file.relative_to(PROJECT_ROOT))
            if isinstance(analyses[i], Exception):
                results[relative_path] = f"Error during analysis: {analyses[i]}"
                logger.error(f"Analysis failed for {relative_path}: {analyses[i]}")
            else:
                results[relative_path] = analyses[i]

        return results

# --- Example Usage (for testing) --- #
async def _test_reflector():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("Starting StructureReflector test...")

    # Ensure core components are available
    if not LLMInterface or not PROJECT_ROOT:
        logger.error("Core components (LLMInterface, PROJECT_ROOT) not available. Cannot run test.")
        return

    # Initialize LLMInterface (adjust URL if needed)
    # You might need to start the server first: python -m llama_cpp.server ...
    try:
        from ..core.config import LLAMA_SERVER_URL
        llm_interface = LLMInterface(base_url=LLAMA_SERVER_URL)
        logger.info(f"LLMInterface initialized with URL: {LLAMA_SERVER_URL}")
    except Exception as e:
        logger.error(f"Failed to initialize LLMInterface: {e}")
        return

    reflector = StructureReflector(llm_interface=llm_interface)

    # Test 1: Analyze a single file (e.g., cli_linter.py itself)
    test_file = Path(__file__).parent / "cli_linter.py"
    if test_file.exists():
        print(f"\n--- Analyzing single file: {test_file.name} ---")
        analysis = await reflector.analyze_file_semantically(test_file)
        print(analysis)
    else:
        print(f"Test file {test_file} not found, skipping single file analysis.")

    # Test 2: Scan a project directory (e.g., a3x/cli)
    print("\n--- Scanning project directory: a3x/cli ---")
    scan_results = await reflector.run_project_scan(base_path="a3x/cli")
    print("Scan Results:")
    for filename, diagnosis in scan_results.items():
        print(f"\n[File: {filename}]\n{diagnosis}")
        print("-" * 20)

if __name__ == "__main__":
    # Note: Running this directly requires the LLM server to be accessible
    # and the project structure (docs/manifestos) to be correct relative to PROJECT_ROOT.
    asyncio.run(_test_reflector()) 