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
    in_objectives = False
    in_questions = False
    temp_objectives = []
    temp_questions = []

    for line in content:
        if re.match(r'^#+ Objectives', line, re.IGNORECASE):
            in_objectives = True
            in_questions = False
            temp_objectives = []
        elif re.match(r'^#+ Questions', line, re.IGNORECASE):
            in_questions = True
            in_objectives = False
            temp_questions = []
        elif in_objectives:
            if re.match(r'^#', line):
                in_objectives = False
                # markdown_lines.append("::::::::::::::::::::::::::::::::::::::: objectives")
                # markdown_lines.extend(temp_objectives)
                # markdown_lines.append("::::::::::::::::::::::::::::::::::::::::::::::::::")
                # markdown_lines.append(line)
            else:
                temp_objectives.append(line)
        elif in_questions:
            if re.match(r'^#', line):
                in_questions = False
                # markdown_lines.append(":::::::::::::::::::::::::::::::::::::::: questions")
                # markdown_lines.extend(temp_questions)
                # markdown_lines.append("::::::::::::::::::::::::::::::::::::::::::::::::::")
                # markdown_lines.append(line)
            else:
                temp_questions.append(line)
        else:
            markdown_lines.append(line)
    
    # Append any remaining objectives or questions
    if in_objectives:
        markdown_lines.append("::::::::::::::::::::::::::::::::::::::: objectives")
        markdown_lines.extend(temp_objectives)
        markdown_lines.append("::::::::::::::::::::::::::::::::::::::::::::::::::")

    if in_questions:
        markdown_lines.append(":::::::::::::::::::::::::::::::::::::::: questions")
        markdown_lines.extend(temp_questions)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Jupyter Notebook to Markdown")
    parser.add_argument("input_path", help="Path to the directory containing the input Jupyter Notebook file")
    parser.add_argument("input_file_name", help="Name of the input Jupyter Notebook file (without extension)")

    args = parser.parse_args()

    convert_ipynb_to_markdown(args.input_path, args.input_file_name)

    # E.g., ...
    # python convert-ipynb-to-md.py ../ 7a-OOD-detection-output-based
