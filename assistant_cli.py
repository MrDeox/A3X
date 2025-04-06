from a3x.core.logging_config import setup_logging
from a3x.cli.interface import run_cli # Use run_cli instead of main_interface

# Configure logging early
setup_logging()


if __name__ == "__main__":
    # Call the main CLI function
    run_cli() 