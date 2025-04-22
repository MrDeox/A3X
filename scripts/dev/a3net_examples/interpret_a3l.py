# Example script to interpret and execute commands from an .a3l file.
# Main logic moved to a3x.a3net.modules.executor

import sys
import logging
import argparse

# Import the main execution function from the new module
# Ensure the module path is correct relative to how this script is run
# If running as `python -m a3x.a3net.examples.interpret_a3l`, this should work.
from a3x.a3net.modules.executor import run_a3l_file

# Configure basic logging (can be kept here or moved depending on overall structure)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Main Execution Block --- #

if __name__ == '__main__':
    # Use the new path for the default script
    default_a3l_file = "data/a3l_scripts/sample.a3l" # Default relative to project root
    script_to_run = default_a3l_file

    parser = argparse.ArgumentParser(description="Interpret an A3L script file.")

    # Allow specifying a different file via command line argument
    if len(sys.argv) > 1:
        script_to_run = sys.argv[1]

    # Call the execution function from the executor module
    try:
        logger.info(f"Starting A3L execution for: {script_to_run}")
        run_a3l_file(script_to_run)
        logger.info(f"Finished A3L execution for: {script_to_run}")
    except ImportError as e:
        # Provide more guidance on potential issues
        logger.error(f"Error importing modules. Ensure PYTHONPATH is set correctly, run from the project root, or run as a module (-m). Details: {e}")
        print(f"ImportError: {e}. Could not run A3L file due to module import issue. Please check your execution method and Python path.")
    except FileNotFoundError as e:
        logger.error(f"A required file was not found: {e}")
        print(f"FileNotFoundError: {e}. Please ensure the A3L file and any required fragment files exist.")
    except Exception as e:
        logger.exception(f"An unexpected error occurred during A3L execution of {script_to_run}") # Log traceback
        print(f"An unexpected error occurred: {e}") 