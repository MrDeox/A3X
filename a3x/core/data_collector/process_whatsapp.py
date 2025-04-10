# a3x/core/data_collector/process_whatsapp.py

import os
import re
import json
import logging
import sys

# --- Configuration ---
# Set this to the exact sender name used for Arthur's messages in the export files
ARTHUR_SENDER_NAME = "Arthur"

# --- Path Calculation (Relative to this script) ---
try:
    # Assumes script is in a3x/core/data_collector/
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    # Go up 3 levels to get to project root (a3x/core/data_collector -> a3x/core -> a3x -> project_root)
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
except NameError:
    # Fallback if __file__ is not defined (e.g., interactive execution)
    # Assumes running from project root in this case
    PROJECT_ROOT = os.path.abspath('.')

WHATSAPP_EXPORTS_DIR = os.path.join(PROJECT_ROOT, "data", "whatsapp_exports")
OUTPUT_DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "arthur_decision_dataset.jsonl")
LOG_FILE_PATH = os.path.join(PROJECT_ROOT, "process_whatsapp.log")

# --- Regex for WhatsApp Log Line Parsing ---
# Handles formats like: DD/MM/YYYY, HH:MM - Sender Name: Message
#                 or: MM/DD/YY, HH:MM - Sender Name: Message
# Captures timestamp_str, sender, message
# Improved to handle sender names with spaces but not colons immediately after
# Updated 2024-07-17: Removed comma after date to match observed format.
LINE_REGEX = re.compile(r"^(\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2})\s+-\s+([^:]+?):\s+(.*)")

# --- Ignored Message Patterns ---
# List of substrings indicating system messages or media to ignore
IGNORED_PATTERNS = [
    "<Media omitted>",
    "<Mídia oculta>", # Added Portuguese version
    "<Mensagem editada>", # Added edited message pattern
    "Messages and calls are end-to-end encrypted.",
    "changed the subject from",
    "created group",
    "added you",
    "changed this group's icon",
    "changed the group description",
    "left",
    "You created group",
    "You were added",
    "You joined using this group's invite link",
    "changed their phone number to a new number.",
    "changed to",
    "security code changed."
]

# --- Logging Setup ---
# Ensure logs directory exists if needed (logging to root here)
# os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)

logger = logging.getLogger("ProcessWhatsApp")
logger.setLevel(logging.INFO)

# Clear existing handlers to avoid duplication if script is run multiple times
if logger.hasHandlers():
    logger.handlers.clear()

# File Handler (overwrite mode)
fh = logging.FileHandler(LOG_FILE_PATH, mode='w', encoding='utf-8')
fh.setLevel(logging.INFO)

# Console Handler
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)

# Formatter
formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
fh.setFormatter(formatter)
ch.setFormatter(formatter)

# Add Handlers
logger.addHandler(fh)
logger.addHandler(ch)

# --- Helper Functions ---

def parse_line(line):
    """Parses a single line using regex. Returns (sender, message) or None."""
    match = LINE_REGEX.match(line)
    if match:
        _, sender, message = match.groups() # Timestamp not used directly in pairing logic
        sender = sender.strip()
        message = message.strip()
        # Check if message content should be ignored
        if not message or any(pattern in message for pattern in IGNORED_PATTERNS):
            return None
        return sender, message
    return None

def append_to_jsonl(filepath, data_record):
    """Appends a dictionary as a JSON line to the specified file."""
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'a', encoding='utf-8') as f:
            json.dump(data_record, f, ensure_ascii=False)
            f.write('\n')
            return True
    except Exception as e:
        logger.error(f"Failed to write record to {filepath}: {e}")
        return False

# --- Main Processing Logic (Refactored) ---

def process_whatsapp_exports(exports_dir, output_path, arthur_name):
    """Processes all .txt files using a two-pass approach for better context handling."""
    logger.info("--- Starting WhatsApp Export Processing (Refactored) ---")
    logger.info(f"Input directory: {exports_dir}")
    logger.info(f"Output file: {output_path} (appending)")
    logger.info(f"Identifying Arthur's messages by sender name: '{arthur_name}'")

    if not os.path.isdir(exports_dir):
        logger.error(f"Input directory not found: {exports_dir}")
        return

    processed_files_count = 0
    total_records_generated = 0
    skipped_files_count = 0

    try:
        txt_files = [f for f in os.listdir(exports_dir) if f.lower().endswith(".txt")]
        if not txt_files:
            logger.warning(f"No .txt files found in {exports_dir}")
            return
    except OSError as e:
        logger.error(f"Could not list files in directory {exports_dir}: {e}")
        return

    logger.info(f"Found {len(txt_files)} .txt files to process.")

    for filename in txt_files:
        file_path = os.path.join(exports_dir, filename)
        logger.info(f"Processing file: '{filename}'...")
        processed_files_count += 1
        file_records_count = 0
        parsed_messages = [] # Store all parsed messages from the file

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                current_full_message = ""
                current_sender = None

                for line_num, line in enumerate(f, 1):
                    line = line.rstrip()
                    if not line.strip():
                        continue

                    parsed = parse_line(line)

                    if parsed:
                        # Start of a new message
                        sender, message_start = parsed

                        # If there was a message being built, add it to our list
                        if current_sender is not None:
                            parsed_messages.append((current_sender, current_full_message))

                        # Start the new message
                        current_sender = sender
                        current_full_message = message_start
                    # Handle multi-line messages
                    elif current_sender is not None:
                        current_full_message += "\n" + line

                # Add the very last message in the file
                if current_sender is not None:
                    parsed_messages.append((current_sender, current_full_message))

            # --- Second Pass: Iterate through parsed messages to find pairs ---
            for i in range(1, len(parsed_messages)):
                prev_sender, prev_message = parsed_messages[i-1]
                current_sender, current_message = parsed_messages[i]

                # Check for the desired pattern: Other -> Arthur
                if current_sender == arthur_name and prev_sender != arthur_name:
                    record = {
                        "context": prev_message,
                        "arthur_response": current_message
                        # "source": f"{filename}#Msg{i}" # Optional source tracking
                    }
                    if append_to_jsonl(output_path, record):
                        file_records_count += 1
                        total_records_generated += 1

            if file_records_count > 0:
                logger.info(f"-> Generated {file_records_count} records from '{filename}'.")
            else:
                logger.info(f"-> No valid 'context -> {arthur_name}' sequences found in '{filename}'.")

        except UnicodeDecodeError as ude:
            logger.error(f"Encoding error processing file '{filename}'. Ensure it is UTF-8: {ude}")
            skipped_files_count += 1
        except Exception as e:
            # Use line_num from the first pass for approximate error location
            logger.error(f"Unexpected error processing file '{filename}': {e}", exc_info=True)
            skipped_files_count += 1

    logger.info("--- WhatsApp Export Processing Finished ---")
    logger.info(f"Processed {processed_files_count} .txt file(s).")
    if skipped_files_count > 0:
        logger.warning(f"Skipped {skipped_files_count} file(s) due to errors.")
    logger.info(f"Total records generated and appended: {total_records_generated}")
    logger.info(f"Output dataset path: {output_path}")
    logger.info(f"Log file path: {LOG_FILE_PATH}")

if __name__ == "__main__":
    # Add the project root to sys.path to allow running directly
    # and potentially finding other project modules if needed later.
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    process_whatsapp_exports(WHATSAPP_EXPORTS_DIR, OUTPUT_DATASET_PATH, ARTHUR_SENDER_NAME) 