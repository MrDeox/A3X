import re

def snake_to_pascal_case(snake_str: str) -> str:
    \"\"\"Converts a snake_case or kebab-case string to PascalCase.\"\"\"
    # Replace hyphens with underscores first
    snake_str = snake_str.replace('-', '_')
    # Split by underscore and capitalize each part
    components = snake_str.split('_')
    # Filter out empty strings that might result from multiple underscores
    return \"\".join(x.title() for x in components if x) 