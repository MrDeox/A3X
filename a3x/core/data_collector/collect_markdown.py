# a3x/core/data_collector/collect_markdown.py

"""Collector for extracting insights from Markdown documents."""

import os
import datetime
import re
from .collector_base import BaseCollector

class MarkdownCollector(BaseCollector):
    def __init__(self):
        super().__init__(source_name="markdown")
        # Regex to find the first H1 header (e.g., # Header Title)
        self.h1_regex = re.compile(r"^#\s+(.*)", re.MULTILINE)

    def _extract_first_h1(self, content):
        match = self.h1_regex.search(content)
        return match.group(1).strip() if match else None

    def collect(self, directory_path):
        """Scans a directory for Markdown files and extracts content.

        Args:
            directory_path (str): Path to the directory containing .md files.

        Returns:
            list: A list of formatted records.
        """
        records = []
        if not os.path.isdir(directory_path):
            print(f"Error: Markdown Directory not found at {directory_path}")
            return []

        # print(f"Scanning directory: {directory_path}") # Optional: Keep for less verbose logging
        for filename in os.listdir(directory_path):
            if filename.lower().endswith(".md") or filename.lower().endswith(".txt"):
                file_path = os.path.join(directory_path, filename)
                # print(f"Processing file: {filename}") # Optional
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    first_h1 = self._extract_first_h1(content)
                    input_text = first_h1 if first_h1 else filename
                    context = f"Document content from {filename}"
                    response = content
                    reasoning = "Document content analysis"
                    timestamp = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))

                    records.append(self._format_record(
                        input_text=input_text,
                        context=context,
                        response=response,
                        reasoning=reasoning,
                        timestamp=timestamp
                    ))
                    # print(f"  -> Record created for {filename}") # Optional

                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

        # print(f"Finished processing directory. Found {len(records)} potential records.") # Optional
        return records

# --- Example Usage (Keep for potential direct testing later if needed) ---
# if __name__ == '__main__':
#     # Logic to allow direct execution (e.g., adjusting sys.path)
#     # ... (requires more setup to handle relative imports)
#     pass 