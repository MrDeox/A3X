import os
import re
import json
import logging
from datetime import datetime

# --- Configuration ---
# Set ARTHUR_SENDER_NAME to the exact name used in the logs
# Assuming the name is 'Arthur' based on context.
# If different in the logs, please change this value manually.
ARTHUR_SENDER_NAME = "Arthur"

WHATSAPP_EXPORTS_DIR = os.path.join("data", "whatsapp_exports")
OUTPUT_DATASET_PATH = os.path.join("data", "arthur_decision_dataset.jsonl")

# Regex to capture typical WhatsApp lines (adjust based on actual format)
# Handles formats like: DD/MM/YYYY, HH:MM - Sender Name: Message
#                 or: MM/DD/YY, HH:MM - Sender Name: Message
# It assumes sender names don't contain ':' right after the name.
# It captures timestamp, sender, and message content.
# Modified to handle potential variations in date/time format and sender name
LINE_REGEX = re.compile(r"^(\d{1,2}/\d{1,2}/\d{2,4},\s+\d{1,2}:\d{2})\s+-\s+([^:]+):\s+(.*)")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper Functions ---

def parse_line(line):
    """Parses a single line using regex. Returns (timestamp_str, sender, message) or None."""
    match = LINE_REGEX.match(line)
    if match:
        timestamp_str, sender, message = match.groups()
        # Basic cleaning
        sender = sender.strip()
        message = message.strip()
        return timestamp_str, sender, message
    return None

def append_to_jsonl(filepath, data_record):
    """Appends a dictionary as a JSON line to the specified file."""
    try:
        with open(filepath, 'a', encoding='utf-8') as f:
            json.dump(data_record, f, ensure_ascii=False)
            f.write('\n')
    except Exception as e:
        logger.error(f"Failed to write record to {filepath}: {e}")

# --- Main Processing Logic ---

def process_whatsapp_exports(exports_dir, output_path, arthur_name):
    """Processes all .txt files in the exports directory and saves to JSONL."""
    logger.info("Starting WhatsApp export processing...")
    logger.info(f"Input directory: {exports_dir}")
    logger.info(f"Output file: {output_path}")
    logger.info(f"Identifying Arthur's messages by sender name: '{arthur_name}'")

    if not os.path.isdir(exports_dir):
        logger.error(f"Input directory not found: {exports_dir}")
        return

    # Clear the output file if it exists
    if os.path.exists(output_path):
        logger.warning(f"Output file {output_path} already exists. Overwriting.")
        try:
            os.remove(output_path)
        except OSError as e:
            logger.error(f"Could not remove existing output file: {e}")
            return

    processed_records = 0
    total_files = 0

    for filename in os.listdir(exports_dir):
        if filename.endswith(".txt"):
            total_files += 1
            file_path = os.path.join(exports_dir, filename)
            logger.info(f"Processing file: {filename}...")

            current_input_messages = []
            last_sender = None
            consecutive_other_messages = []

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue

                        parsed = parse_line(line)

                        if parsed:
                            timestamp_str, sender, message = parsed
                            # logger.debug(f"Parsed Line {line_num}: S={sender}, M={message[:50]}...")

                            # Message from Arthur
                            if sender == arthur_name:
                                # If there were preceding messages from others, create a record
                                if consecutive_other_messages:
                                    input_text = "\n".join([m['message'] for m in consecutive_other_messages])
                                    context = {  # Store more context if needed
                                        "source_file": filename,
                                        "first_input_timestamp": consecutive_other_messages[0]['timestamp'],
                                        "last_input_timestamp": consecutive_other_messages[-1]['timestamp'],
                                        "input_senders": list(set(m['sender'] for m in consecutive_other_messages))
                                    }
                                    record = {
                                        "input": input_text,
                                        # "context": context,  # Optional: Add more structured context
                                        "arthur_response": message,
                                        "source": f"{filename}#L{line_num}",  # Track source line
                                        "timestamp": timestamp_str  # Timestamp of Arthur's response
                                    }
                                    append_to_jsonl(output_path, record)
                                    processed_records += 1
                                    # logger.debug(f"Created record: Input={input_text[:50]}... -> Response={message[:50]}...")

                                # Reset for the next interaction
                                consecutive_other_messages = []
                            # Message from someone else
                            else:
                                consecutive_other_messages.append({
                                    "timestamp": timestamp_str,
                                    "sender": sender,
                                    "message": message
                                })

                            last_sender = sender
                        # Handle multi-line messages (append to the last parsed message)
                        elif last_sender is not None and consecutive_other_messages:  # Append to last *other* message
                            # Only append if the last message was from someone else
                            if consecutive_other_messages[-1]['sender'] != arthur_name:
                                consecutive_other_messages[-1]['message'] += "\n" + line
                        # else:  # Could be media omitted message or system message - ignore for now
                            # logger.debug(f"Skipping unparsed line {line_num}: {line}")

            except Exception as e:
                logger.error(f"Error processing file {filename}: {e}", exc_info=True)

    logger.info(f"Finished processing {total_files} file(s).")
    logger.info(f"Total records generated: {processed_records}")
    logger.info(f"Output dataset saved to: {output_path}")

if __name__ == "__main__":
    # Check if the name is still the default, warn if so.
    # No longer exiting automatically, just warning.
    if ARTHUR_SENDER_NAME == "Arthur":  # Default value check
        logger.warning("ARTHUR_SENDER_NAME is using the default value 'Arthur'. If this is not correct for your logs, please edit the script.")

    process_whatsapp_exports(WHATSAPP_EXPORTS_DIR, OUTPUT_DATASET_PATH, ARTHUR_SENDER_NAME)