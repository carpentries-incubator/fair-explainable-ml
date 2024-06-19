import nbformat
import re
import argparse
import os

def convert_ipynb_to_markdown(input_path, input_file_name):
    input_file = os.path.join(input_path, f"{input_file_name}.ipynb")
    output_file = os.path.join(input_path, f"{input_file_name}.md")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    markdown_lines = []

    # Extract metadata
    metadata = nb.metadata
    if 'title' in metadata:
        markdown_lines.append(f"title: {metadata['title']}")
    if 'teaching' in metadata and 'exercises' in metadata:
        markdown_lines.append(f"teaching: {metadata['teaching']}")
        markdown_lines.append(f"exercises: {metadata['exercises']}")
    markdown_lines.append("---")

    for cell in nb.cells:
        cell_type = cell.cell_type
        if cell_type == 'markdown':
            markdown_lines.extend(process_markdown_cell(cell))
        elif cell_type == 'code':
            markdown_lines.extend(process_code_cell(cell))

    markdown_content = '\n'.join(markdown_lines)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(markdown_content)

def process_markdown_cell(cell):
    content = cell.source.split('\n')
    markdown_lines = []

    # Extract metadata
    objectives = extract_metadata(content, "objectives")
    if objectives:
        markdown_lines.append("::::::::::::::::::::::::::::::::::::::: objectives")
        markdown_lines.extend(objectives)
        markdown_lines.append("::::::::::::::::::::::::::::::::::::::::::::::::::")

    questions = extract_metadata(content, "questions")
    if questions:
        markdown_lines.append("::::::::::::::::::::::::::::::::::::::::::::::::::")
        markdown_lines.append(":::::::::::::::::::::::::::::::::::::::: questions")
        markdown_lines.extend(questions)
        markdown_lines.append("::::::::::::::::::::::::::::::::::::::::::::::::::")

    markdown_lines.extend(content)  # Include content from markdown cell

    callouts = extract_metadata(content, "callout")
    if callouts:
        markdown_lines.append(":::::::::::::::::::::::::::::::::::::::::  callout")
        markdown_lines.extend(callouts)
        markdown_lines.append("::::::::::::::::::::::::::::::::::::::::::::::::::")

    challenges = extract_metadata(content, "challenge")
    if challenges:
        markdown_lines.append(":::::::::::::::::::::::::::::::::::::::  challenge")
        markdown_lines.extend(challenges)
        markdown_lines.append(":::::::::::::::  solution")
        solutions = extract_metadata(content, "solution")
        markdown_lines.extend(solutions)
        markdown_lines.append(":::::::::::::::::::::::::")

    keypoints = extract_metadata(content, "keypoints")
    if keypoints:
        markdown_lines.append(":::::::::::::::::::::::::::::::::::::::: keypoints")
        markdown_lines.extend(keypoints)
        markdown_lines.append("::::::::::::::::::::::::::::::::::::::::::::::::::")

    return markdown_lines

def process_code_cell(cell):
    code_lines = cell.source.split('\n')
    markdown_lines = []

    # Surround code block with ```python``` tags
    markdown_lines.append("```python")
    markdown_lines.extend(code_lines)
    markdown_lines.append("```")

    return markdown_lines

def extract_metadata(content, metadata_marker):
    metadata = []
    in_metadata = False

    for line in content:
        if re.match(f"^::+ {metadata_marker}", line):
            in_metadata = True
        elif in_metadata and re.match("^::+:?", line):
            in_metadata = False
        elif in_metadata:
            metadata.append(line)

    return metadata

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Jupyter Notebook to Markdown")
    parser.add_argument("input_path", help="Path to the directory containing the input Jupyter Notebook file")
    parser.add_argument("input_file_name", help="Name of the input Jupyter Notebook file (without extension)")

    args = parser.parse_args()

    convert_ipynb_to_markdown(args.input_path, args.input_file_name)

    # E.g., ...
    # python convert_ipynb_to_markdown.py ../episodes 5b-probes
