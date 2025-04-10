# a3x/core/data_collector/collector_base.py

"""Base classes or common utilities for data collectors."""

class BaseCollector:
    """Abstract base class for data collectors."""
    def __init__(self, source_name):
        self.source_name = source_name

    def collect(self, *args, **kwargs):
        """Collects data from the source and returns structured records."""
        raise NotImplementedError

    def _format_record(self, input_text, context, response, reasoning, timestamp):
        """Formats a single record into the standard JSONL structure."""
        return {
            "input": input_text,
            "context": context,
            "arthur_response": response,
            "reasoning": reasoning,
            "source": self.source_name,
            "timestamp": str(timestamp) # Ensure timestamp is string
        }

# Add any common utility functions here if needed 