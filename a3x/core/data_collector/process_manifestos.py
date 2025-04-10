# a3x/core/data_collector/process_manifestos.py

import os
import json
import logging
import sys
import glob

# --- Configuration ---
MAX_CHUNK_SIZE = 500

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

DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")
OUTPUT_DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "arthur_manifest_dataset.jsonl")
LOG_FILE_PATH = os.path.join(PROJECT_ROOT, "process_manifestos.log")

# --- Logging Setup ---
logger = logging.getLogger("ProcessManifestos")
logger.setLevel(logging.INFO)

# Clear existing handlers
if logger.hasHandlers():
    logger.handlers.clear()

# File Handler (overwrite mode)
try:
    os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)
    fh = logging.FileHandler(LOG_FILE_PATH, mode='w', encoding='utf-8')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s',
                                datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
except Exception as e:
    print(f"Error setting up file logger: {e}", file=sys.stderr)


# Console Handler
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
ch_formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s',
                                 datefmt='%Y-%m-%d %H:%M:%S')
ch.setFormatter(ch_formatter)
logger.addHandler(ch)


# --- Helper Functions ---

def chunk_text(text, max_size=MAX_CHUNK_SIZE):
    """Splits text into chunks of max_size, respecting line breaks."""
    chunks = []
    current_chunk_lines = []
    current_chunk_len = 0
    lines = text.splitlines() # Split text into lines

    for line in lines:
        line_len = len(line)
        # +1 for the newline character that will be added when joining
        projected_len = current_chunk_len + line_len + (1 if current_chunk_lines else 0)

        if line_len > max_size:
            # If a single line is too long, add the current chunk (if any)
            # Then add the long line as its own chunk (potentially truncated or split further if needed)
            # For now, just add it as one chunk and log a warning.
            if current_chunk_lines:
                chunks.append("\n".join(current_chunk_lines))
                current_chunk_lines = []
                current_chunk_len = 0
            logger.warning(f"Single line exceeds max_size ({line_len} > {max_size}). Adding as separate chunk.")
            chunks.append(line) # Add the long line as its own chunk
        elif projected_len <= max_size:
            # Add line to current chunk
            current_chunk_lines.append(line)
            current_chunk_len = projected_len - (1 if len(current_chunk_lines) > 1 else 0) # Adjust length based on actual joining later
        else:
            # Current chunk is full, finalize it and start a new one
            chunks.append("\n".join(current_chunk_lines))
            current_chunk_lines = [line]
            current_chunk_len = line_len

    # Add the last remaining chunk
    if current_chunk_lines:
        chunks.append("\n".join(current_chunk_lines))

    return chunks

def append_to_jsonl(filepath, data_record):
    """Appends a dictionary as a JSON line to the specified file."""
    try:
        # Ensure the directory exists (redundant if overwrite logic is used, but safe)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'a', encoding='utf-8') as f:
            json.dump(data_record, f, ensure_ascii=False)
            f.write('\n')
            return True
    except Exception as e:
        logger.error(f"Failed to write record to {filepath}: {e}")
        return False

# --- Main Processing Logic ---

def process_manifests(docs_dir, output_path):
    """Processes .txt and .md files in docs_dir, chunks them, and saves to output_path."""
    logger.info("--- Starting Manifesto Processing ---")
    logger.info(f"Input directory: {docs_dir}")
    logger.info(f"Output file: {output_path} (will be overwritten)")

    if not os.path.isdir(docs_dir):
        logger.error(f"Input directory not found: {docs_dir}")
        return

    # Find relevant files
    txt_files = glob.glob(os.path.join(docs_dir, "*.txt"))
    md_files = glob.glob(os.path.join(docs_dir, "*.md"))
    all_files = txt_files + md_files

    if not all_files:
        logger.warning(f"No .txt or .md files found in {docs_dir}")
        return

    logger.info(f"Found {len(all_files)} files to process.")

    # Overwrite output file before starting
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            pass # Just truncate the file
        logger.info(f"Cleared/Created output file: {output_path}")
    except Exception as e:
        logger.error(f"Could not clear or create output file {output_path}: {e}")
        return

    total_chunks_generated = 0
    processed_files_count = 0

    for file_path in all_files:
        filename = os.path.basename(file_path)
        logger.info(f"Processing file: '{filename}'...")
        processed_files_count += 1
        file_chunks_count = 0

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            chunks = chunk_text(content, MAX_CHUNK_SIZE)

            for i, chunk in enumerate(chunks):
                if not chunk.strip(): # Skip empty chunks
                    continue
                record = {
                    "source": filename,
                    "chunk_index": i,
                    "text": chunk
                }
                if append_to_jsonl(output_path, record):
                    file_chunks_count += 1
                else:
                    logger.warning(f"Failed to write chunk {i} from {filename}.")

            if file_chunks_count > 0:
                logger.info(f"-> Extracted {file_chunks_count} chunks from '{filename}'.")
                total_chunks_generated += file_chunks_count
            else:
                 logger.info(f"-> No text chunks extracted from '{filename}'.")


        except UnicodeDecodeError as ude:
            logger.error(f"Encoding error processing file '{filename}'. Ensure it is UTF-8: {ude}")
        except Exception as e:
            logger.error(f"Unexpected error processing file '{filename}': {e}", exc_info=True)

    logger.info("--- Manifesto Processing Finished ---")
    logger.info(f"Processed {processed_files_count} file(s).")
    logger.info(f"Total chunks generated and saved: {total_chunks_generated}")
    logger.info(f"Output dataset path: {output_path}")
    logger.info(f"Log file path: {LOG_FILE_PATH}")


if __name__ == "__main__":
    # Add the project root to sys.path if needed
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    process_manifests(DOCS_DIR, OUTPUT_DATASET_PATH) 