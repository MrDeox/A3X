# a3x/core/data_collector/build_unified_dataset.py

import os
import re
import json
import logging
import sys
import glob

# --- Configuration ---
# Manifesto Processing
MAX_CHUNK_SIZE = 500

# WhatsApp Processing
ARTHUR_SENDER_NAME = "Arthur" # Exact name used in exports
# Regex: Handles DD/MM/YYYY HH:MM - Sender: Message (No comma after date)
LINE_REGEX = re.compile(r"^(\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2})\s+-\s+([^:]+?):\s+(.*)")
# Patterns to ignore in WhatsApp messages
IGNORED_WHATSAPP_PATTERNS = [
    "<Media omitted>",
    "<Mídia oculta>",
    "<Mensagem editada>",
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
    "security code changed.",
    "As mensagens e as ligações são protegidas com a criptografia de ponta a ponta"
]

# --- Path Calculation ---
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
except NameError:
    PROJECT_ROOT = os.path.abspath('.')

DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")
WHATSAPP_EXPORTS_DIR = os.path.join(PROJECT_ROOT, "data", "whatsapp_exports")
OUTPUT_DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "arthur_unified_dataset.jsonl")
LOG_FILE_PATH = os.path.join(PROJECT_ROOT, "build_unified_dataset.log")

# --- Logging Setup ---
logger = logging.getLogger("BuildUnifiedDataset")
logger.setLevel(logging.INFO)

if logger.hasHandlers():
    logger.handlers.clear()

# File Handler
try:
    os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)
    fh = logging.FileHandler(LOG_FILE_PATH, mode='w', encoding='utf-8')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
except Exception as e:
    print(f"Error setting up file logger: {e}", file=sys.stderr)

# Console Handler
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
ch_formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
ch.setFormatter(ch_formatter)
logger.addHandler(ch)


# --- Helper Functions ---

def chunk_text(text, max_size=MAX_CHUNK_SIZE):
    """Splits text into chunks of max_size, respecting line breaks."""
    chunks = []
    current_chunk_lines = []
    current_chunk_len = 0
    lines = text.splitlines()

    for line in lines:
        line_len = len(line)
        projected_len = current_chunk_len + line_len + (1 if current_chunk_lines else 0)

        if line_len > max_size:
            if current_chunk_lines:
                chunks.append("\n".join(current_chunk_lines))
                current_chunk_lines = []
                current_chunk_len = 0
            logger.warning(f"Single line exceeds max_size ({line_len} > {max_size}). Adding as separate chunk.")
            chunks.append(line)
        elif projected_len <= max_size:
            current_chunk_lines.append(line)
            current_chunk_len = projected_len - (1 if len(current_chunk_lines) > 1 else 0)
        else:
            chunks.append("\n".join(current_chunk_lines))
            current_chunk_lines = [line]
            current_chunk_len = line_len

    if current_chunk_lines:
        chunks.append("\n".join(current_chunk_lines))

    # Filter out empty chunks that might result from multiple blank lines
    return [chunk for chunk in chunks if chunk.strip()]

def parse_whatsapp_line(line):
    """Parses a single WhatsApp line. Returns (sender, message) or None."""
    match = LINE_REGEX.match(line)
    if match:
        _, sender, message = match.groups()
        sender = sender.strip()
        message = message.strip()
        # Check if message content should be ignored
        if not message or any(pattern in message for pattern in IGNORED_WHATSAPP_PATTERNS):
            return None
        return sender, message
    return None

def write_jsonl(filepath, data_records):
    """Writes a list of dictionaries to a JSONL file, overwriting it."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            for record in data_records:
                json.dump(record, f, ensure_ascii=False)
                f.write('\n')
        return True
    except Exception as e:
        logger.error(f"Failed to write unified dataset to {filepath}: {e}")
        return False

# --- Main Processing Logic ---

def build_dataset():
    logger.info("--- Starting Unified Dataset Build ---")
    unified_data = []
    total_manifesto_chunks = 0
    total_whatsapp_pairs = 0

    # 1. Process Manifestos
    logger.info(f"Processing manifestos from: {DOCS_DIR}")
    if not os.path.isdir(DOCS_DIR):
        logger.warning(f"Manifesto directory not found: {DOCS_DIR}")
    else:
        manifesto_files = glob.glob(os.path.join(DOCS_DIR, "*.txt")) + \
                          glob.glob(os.path.join(DOCS_DIR, "*.md"))
        logger.info(f"Found {len(manifesto_files)} manifesto files.")

        for file_path in manifesto_files:
            filename = os.path.basename(file_path)
            logger.info(f" Processing manifesto: '{filename}'...")
            file_chunks_count = 0
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                chunks = chunk_text(content, MAX_CHUNK_SIZE)
                for i, chunk in enumerate(chunks):
                    record = {
                        "type": "manifesto",
                        "source": filename,
                        "text": chunk,
                        "meta": {"chunk_index": i}
                    }
                    unified_data.append(record)
                    file_chunks_count += 1
                if file_chunks_count > 0:
                    logger.info(f" -> Extracted {file_chunks_count} chunks.")
                    total_manifesto_chunks += file_chunks_count
                else:
                    logger.info(f" -> No text chunks extracted.")
            except Exception as e:
                logger.error(f" Error processing manifesto file '{filename}': {e}", exc_info=True)

    # 2. Process WhatsApp Exports
    logger.info(f"Processing WhatsApp exports from: {WHATSAPP_EXPORTS_DIR}")
    if not os.path.isdir(WHATSAPP_EXPORTS_DIR):
        logger.warning(f"WhatsApp exports directory not found: {WHATSAPP_EXPORTS_DIR}")
    else:
        whatsapp_files = glob.glob(os.path.join(WHATSAPP_EXPORTS_DIR, "*.txt"))
        logger.info(f"Found {len(whatsapp_files)} WhatsApp export files.")

        for file_path in whatsapp_files:
            filename = os.path.basename(file_path)
            logger.info(f" Processing WhatsApp export: '{filename}'...")
            file_pairs_count = 0
            parsed_messages = []
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    current_full_message = ""
                    current_sender = None
                    for line in f:
                        line = line.rstrip()
                        if not line.strip(): continue

                        parsed = parse_whatsapp_line(line)
                        if parsed:
                            sender, message_start = parsed
                            # Finalize previous message if exists
                            if current_sender is not None:
                                parsed_messages.append((current_sender, current_full_message))
                            # Start new message
                            current_sender = sender
                            current_full_message = message_start
                        elif current_sender is not None: # Handle multi-line
                             # Skip ignored pattern lines even if they are continuations
                            if not any(pattern in line for pattern in IGNORED_WHATSAPP_PATTERNS):
                                current_full_message += "\n" + line

                    # Add the last message
                    if current_sender is not None:
                        parsed_messages.append((current_sender, current_full_message))

                # Find context -> Arthur pairs
                for i in range(1, len(parsed_messages)):
                    prev_sender, prev_message = parsed_messages[i-1]
                    current_sender, current_message = parsed_messages[i]

                    if current_sender == ARTHUR_SENDER_NAME and prev_sender != ARTHUR_SENDER_NAME:
                        record = {
                            "type": "whatsapp",
                            "source": filename,
                            "text": current_message, # Use Arthur's response as main text
                            "meta": {
                                "context": prev_message,
                                "arthur_response": current_message
                            }
                        }
                        unified_data.append(record)
                        file_pairs_count += 1

                if file_pairs_count > 0:
                    logger.info(f" -> Extracted {file_pairs_count} context-response pairs.")
                    total_whatsapp_pairs += file_pairs_count
                else:
                    logger.info(f" -> No valid 'context -> {ARTHUR_SENDER_NAME}' pairs found.")

            except UnicodeDecodeError as ude:
                 logger.error(f" Encoding error processing file '{filename}'. Ensure it is UTF-8: {ude}")
            except Exception as e:
                logger.error(f" Error processing WhatsApp file '{filename}': {e}", exc_info=True)

    # 3. Save Unified Dataset
    logger.info(f"Saving unified dataset to {OUTPUT_DATASET_PATH}...")
    if write_jsonl(OUTPUT_DATASET_PATH, unified_data):
        logger.info(f"Successfully wrote {len(unified_data)} records.")
    else:
        logger.error("Failed to write unified dataset.")

    logger.info("--- Unified Dataset Build Finished ---")
    logger.info(f"Total manifesto chunks: {total_manifesto_chunks}")
    logger.info(f"Total WhatsApp pairs: {total_whatsapp_pairs}")
    logger.info(f"Total records in unified dataset: {len(unified_data)}")
    logger.info(f"Output file: {OUTPUT_DATASET_PATH}")
    logger.info(f"Log file: {LOG_FILE_PATH}")

if __name__ == "__main__":
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    build_dataset() 