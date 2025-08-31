import os
import re
import ast
import black
from pathlib import Path
from typing import List, Tuple

def find_markdown_files(directory: str) -> List[Path]:
    """Find all markdown files in the specified directory."""
    return list(Path(directory).rglob("*.md"))

def extract_python_code_blocks(content: str) -> List[Tuple[str, int, int]]:
    """
    Extract Python code blocks from markdown content.
    Returns list of tuples (code, start_pos, end_pos)
    """
    pattern = r"```\s*python\n(.*?)```"
    matches = []
    for match in re.finditer(pattern, content, re.DOTALL):
        code = match.group(1)
        start_pos = match.start()
        end_pos = match.end()
        matches.append((code, start_pos, end_pos))
    return matches

def validate_python_syntax(code: str) -> bool:
    """Check if the Python code has valid syntax."""
    try:
        ast.parse(code)
        return True
    except SyntaxError as e:
        print(f"Syntax Error: {str(e)}")
        # Get the line with the error
        lines = code.split('\n')
        if e.lineno <= len(lines):
            error_line = lines[e.lineno - 1]
            print(f"Line {e.lineno}: {error_line}")
            # Print a pointer to the error position
            print(" " * (e.offset + len(f"Line {e.lineno}: ") - 1) + "^")
        return False
    except Exception as e:
        print(f"Error validating syntax: {str(e)}")
        return False

def format_python_code(code: str) -> str:
    """Format Python code using black."""
    try:
        return black.format_str(code, mode=black.FileMode())
    except Exception as e:
        print(f"Error formatting code: {e}")
        return code

def process_markdown_file(file_path: Path) -> None:
    """Process a single markdown file."""
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract Python code blocks
    code_blocks = extract_python_code_blocks(content)
    if not code_blocks:
        print(f"No Python code blocks found in {file_path}")
        return

    # Process each code block
    new_content = content
    offset = 0  # Track position changes due to replacements

    for code, start_pos, end_pos in code_blocks:
        # Validate syntax
        if not validate_python_syntax(code):
            print(f"Invalid Python syntax in {file_path}")
            continue

        # Format code
        formatted_code = format_python_code(code)
        
        # Create new code block
        new_block = f"```python\n{formatted_code}```"
        
        # Replace in content
        adjusted_start = start_pos + offset
        adjusted_end = end_pos + offset
        new_content = new_content[:adjusted_start] + new_block + new_content[adjusted_end:]
        
        # Update offset
        offset += len(new_block) - (end_pos - start_pos)

    # Write back to file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print(f"Processed {file_path}")

def main(directory: str):
    """Main function to process all markdown files in a directory."""
    markdown_files = find_markdown_files(directory)
    print(f"Found {len(markdown_files)} markdown files")
    
    for file_path in markdown_files:
        try:
            process_markdown_file(file_path)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python markdown_code_processor.py <directory>")
        sys.exit(1)
    main(sys.argv[1]) 