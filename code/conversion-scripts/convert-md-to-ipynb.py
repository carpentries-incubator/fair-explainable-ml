import re
import nbformat as nbf
import argparse

def rmd_to_notebook(rmd_file, notebook_file, base_image_url):
    # Read the R Markdown file
    with open(rmd_file, 'r') as file:
        rmd_lines = file.readlines()

    # Preview the first 100 lines of the R Markdown file
    print("Preview of the first 100 lines of the R Markdown file:")
    for i, line in enumerate(rmd_lines[:100]):
        print(f"{i+1}: {line}", end='')
    print("\n")

    rmd_content = ''.join(rmd_lines)

    # Initialize a new notebook
    nb = nbf.v4.new_notebook()
    cells = []

    # Extract the title from the YAML front matter
    # YAML front matter starts and ends with '---' and contains title information
    title_match = re.search(r'^---\s*title:\s*"(.*?)".*?---', rmd_content, flags=re.DOTALL | re.MULTILINE)
    if title_match:
        title = title_match.group(1)
        cells.append(nbf.v4.new_markdown_cell(f"# {title}"))

    # Remove the YAML front matter
    rmd_content = re.sub(r'^---.*?---', '', rmd_content, flags=re.DOTALL | re.MULTILINE)

    # Remove instructor notes and solutions
    # These are wrapped with ::: instructor or ::: solution
    rmd_content = re.sub(r':::\s*(instructor|solution)\s*.*?:::', '', rmd_content, flags=re.DOTALL)

    # Replace local image paths with GitHub raw URLs and handle alt text
    def replace_image_path(match):
        # Extract the alt text and image path from the match
        alt_text = match.group(3)
        image_path = match.group(2)
        # Construct the web image URL using the base image URL and the extracted image path
        web_image_url = f"{base_image_url}/{image_path.split('/')[-1]}"
        # Create a Markdown header for the alt text
        alt_header = f"#### {alt_text.strip('alt=')}"
        # Return the Markdown image link followed by the alt text header
        return f"![{alt_text}]({web_image_url})\n\n{alt_header}"

    # Update the base image URL to point to the raw content
    base_image_url = base_image_url.replace('/tree/', '/raw/')

    # Replace image links in the R Markdown content
    rmd_content = re.sub(r'!\[(.*?)\]\((.*?)\)\{(.*?)\}', replace_image_path, rmd_content)

    # Define regex patterns
    # Header pattern matches Markdown headers (e.g., # Header, ## Subheader)
    header_pattern = re.compile(r'^(#+) (.*)', re.MULTILINE)
    # Block start pattern matches lines starting with ':::', optionally followed by a word
    block_start_pattern = re.compile(r'^::+(\s*\w+)?')
    # Block end pattern matches lines starting with ':::'
    block_end_pattern = re.compile(r'^::+')

    # Split the content into lines
    lines = rmd_content.split('\n')

    code_buffer = []
    is_in_code_block = False
    text_buffer = []
    is_in_special_block = False
    special_block_type = ""

    def process_buffer(buffer):
        if buffer:
            cells.append(nbf.v4.new_markdown_cell('\n'.join(buffer)))
            buffer.clear()

    for line in lines:
        if line.startswith('```'):
            # Toggle the code block state
            is_in_code_block = not is_in_code_block
            if is_in_code_block:
                # Process text buffer when entering a code block
                process_buffer(text_buffer)
            else:
                # Add code cell when exiting a code block
                if code_buffer:
                    cells.append(nbf.v4.new_code_cell('\n'.join(code_buffer).strip()))
                    code_buffer = []
        elif is_in_code_block:
            # Collect lines within a code block
            code_buffer.append(line)
        else:
            header_match = header_pattern.match(line)
            block_start_match = block_start_pattern.match(line)
            block_end_match = block_end_pattern.match(line)

            if header_match:
                # Process text buffer when encountering a header
                process_buffer(text_buffer)
                header_level = len(header_match.group(1))
                header_text = header_match.group(2)
                cells.append(nbf.v4.new_markdown_cell(f"{'#' * header_level} {header_text}"))
            elif block_start_match:
                # Process text buffer when encountering a block start
                process_buffer(text_buffer)
                if block_start_match.group(1):
                    is_in_special_block = True
                    special_block_type = block_start_match.group(1).strip().capitalize()
                    text_buffer.append(f"### {special_block_type}")
            elif block_end_match and is_in_special_block:
                # Process text buffer when encountering a block end
                process_buffer(text_buffer)
                is_in_special_block = False
            elif not block_end_match:  # Ignore lines with only colons
                text_buffer.append(line)

    # Process any remaining text in the buffer
    process_buffer(text_buffer)

    # Remove empty cells
    cells = [cell for cell in cells if cell['source'].strip()]

    # Print the cells
    print("Cells in the notebook:")
    for cell_ct, cell in enumerate(cells[:40]):
        print(f"{cell_ct}: {cell}")

    print("\n")

    # Add cells to the notebook
    nb['cells'] = cells

    # Write the notebook to a file
    with open(notebook_file, 'w') as file:
        nbf.write(nb, file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert R Markdown file to Jupyter Notebook")
    parser.add_argument("input_dir", help="Directory of the input R Markdown file")
    parser.add_argument("base_image_url", help="Base URL for images")
    parser.add_argument("file_in", help="Filename of the input R Markdown file (without extension)")

    args = parser.parse_args()

    rmd_file = f"{args.input_dir}{args.file_in}.Rmd"
    notebook_file = f"{args.input_dir}{args.file_in}.ipynb"

    rmd_to_notebook(rmd_file, notebook_file, args.base_image_url)

#e.g., ...
# python convert_rmd_to_notebook.py ../episodes/ https://github.com/carpentries-incubator/deep-learning-intro/raw/main/episodes/fig 4-advanced-layer-types
