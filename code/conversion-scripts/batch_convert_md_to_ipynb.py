import os
import subprocess

def batch_convert_md_to_ipynb(input_directory, output_directory, image_url, exclude_files):
    """
    Batch process .md files in a directory, excluding specific files, converting them to .ipynb using convert-md-to-ipynb.py.

    Args:
        input_directory (str): Path to the directory containing .md files.
        output_directory (str): Path to the directory for output .ipynb files.
        image_url (str): Base URL for images to include in the conversion.
        exclude_files (list): List of file names (with .md extension) to exclude from conversion.
    """
    # List all .md files in the input directory
    md_files = [f for f in os.listdir(input_directory) if f.endswith('.md')]

    # Exclude specified files
    md_files = [f for f in md_files if f not in exclude_files]

    if not md_files:
        print(f"No valid .md files found in {input_directory}. Exiting.")
        return

    # Process each .md file
    for md_file in md_files:
        print(f"Processing: {md_file}")
        try:
            # Call the base script with required arguments
            subprocess.run(
                [
                    "python", "convert-md-to-ipynb.py",
                    input_directory, 
                    output_directory, 
                    image_url, 
                    md_file
                ],
                check=True
            )
            print(f"Successfully converted: {md_file}")
        except subprocess.CalledProcessError as e:
            print(f"Error processing {md_file}: {e}")
        except Exception as e:
            print(f"Unexpected error with {md_file}: {e}")

# Define paths and excluded files
if __name__ == "__main__":
    INPUT_DIR = "../../episodes"  # Adjust path as needed
    OUTPUT_DIR = "../"       # Adjust path as needed
    IMAGE_URL = "https://github.com/carpentries-incubator/fair-explainable-ml/raw/main/images"

    # Files to exclude (no code in these)
    EXCLUDE_FILES = [
        "0-introduction.md",
        "1-preparing-to-train.md",
        "2-model-eval-and-fairness.md",
        "4-explainability-vs-interpretability.md",
        "5a-explainable-AI-method-overview.md",
        "5b-deep-dive-into-methods.md",
        "6-confidence-intervals.md",
        "7a-OOD-detection-overview.md",
        "7e-OOD-detection-algo-design.md"
    ]

    batch_convert_md_to_ipynb(INPUT_DIR, OUTPUT_DIR, IMAGE_URL, EXCLUDE_FILES)


    # Example
    #python batch_convert_md_to_ipynb.py
