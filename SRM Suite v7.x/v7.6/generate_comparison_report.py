# --- START OF FILE generate_comparison_report.py ---

import argparse
import sys
import traceback
import re
from pathlib import Path
import pandas as pd
import csv
import os

# --- Helper Function (Copied for file selection) ---
def find_file_interactive( directory: Path, pattern: str = "*.tsv", file_type_desc: str = "file") -> Path | None:
    """ Finds files matching pattern, prompts user if multiple found. (Standalone Copy) """
    if not directory or not directory.is_dir(): print(f"Error: Directory for interactive search not found or invalid: {directory}"); return None
    try: files = sorted(list(directory.glob(pattern)))
    except Exception as e: print(f"Error searching for files in {directory}: {e}"); return None
    if not files: print(f"Info: No {file_type_desc} files matching '{pattern}' found in {directory}"); return None
    elif len(files) == 1: print(f"Auto-selected {file_type_desc} file: {files[0].name}"); return files[0].resolve()
    else:
        print(f"\nMultiple {file_type_desc} files found matching '{pattern}' in {directory.name}:")
        for i, fp in enumerate(files): print(f"  {i+1}: {fp.name}")
        selected_file = None
        while selected_file is None:
            try:
                choice_str = input(f"Enter the number of the {file_type_desc} file to use (1-{len(files)}), or 0 to cancel: ")
                choice = int(choice_str)
                if choice == 0: print("Selection cancelled."); return None
                choice_idx = choice - 1
                if 0 <= choice_idx < len(files): selected_file = files[choice_idx]; print(f"Selected {file_type_desc} file: {selected_file.name}")
                else: print(f"Invalid choice. Please enter a number between 1 and {len(files)}, or 0.")
            except ValueError: print("Invalid input. Please enter a number.")
            except (EOFError, KeyboardInterrupt): print("\nSelection cancelled."); return None
        return selected_file.resolve()

# --- Main Script Logic ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generates a human-readable Markdown report comparing prompts and outputs from an extracted TSV file.")
    parser.add_argument("--input_tsv_file", type=str, default=None, help="(Optional) Path to the input TSV file (e.g., 'RUN_NAME_extracted.tsv'). If omitted, prompts user to select.")
    parser.add_argument("--search_dir", type=str, default=".", help="Directory to search for the TSV file if --input_tsv_file is omitted.")
    parser.add_argument("--output_report_file", type=str, default=None, help="(Optional) Path for the output Markdown report file. Defaults to '[input_base]_report.md'.")

    args = parser.parse_args()

    # --- 1. Find and Select Input TSV File ---
    input_tsv_path = None
    if args.input_tsv_file:
        input_tsv_path = Path(args.input_tsv_file)
        if not input_tsv_path.is_file(): print(f"Error: Provided input TSV file not found: {input_tsv_path}"); sys.exit(1)
        print(f"Using provided input TSV file: {input_tsv_path}")
    else:
        search_path = Path(args.search_dir);
        if not search_path.is_dir(): print(f"Error: Search directory for TSV files not found: {search_path}"); sys.exit(1)
        print(f"\nSelecting Input TSV File from: {search_path}")
        input_tsv_path = find_file_interactive(search_path, "*_extracted.tsv", "extracted data TSV file") # Use local copy
        if input_tsv_path is None: print("No input TSV file selected. Exiting."); sys.exit(1)

    # --- 2. Determine Output File Path ---
    output_report_path = None
    if args.output_report_file:
        output_report_path = Path(args.output_report_file)
    else:
        default_filename = f"{input_tsv_path.stem}_report.md"
        output_report_path = Path(".") / default_filename

    print(f"Output report will be saved to: {output_report_path.resolve()}")

    # --- 3. Load Data from TSV ---
    print(f"\nLoading data from: {input_tsv_path.name}...")
    try:
        df = pd.read_csv(
            input_tsv_path,
            sep='\t',
            quoting=csv.QUOTE_MINIMAL,
            lineterminator='\n',
            keep_default_na=False,
            na_values=['']
        )
        print(f"Loaded {len(df)} rows.")
        # +++ Clean column names +++
        df.columns = df.columns.str.strip()
        print(f"DEBUG: Cleaned DataFrame Columns: {list(df.columns)}") # Add debug print

        # Basic validation after cleaning
        expected_cols = ['core_id', 'type', 'level', 'sweep', 'prompt', 'output']
        if not all(col in df.columns for col in expected_cols):
            print(f"Error: Input TSV missing expected columns after cleaning. Found: {list(df.columns)}."); sys.exit(1) # Exit if core columns are missing

        # Fill NaNs only after confirming columns exist and are cleaned
        df['prompt'] = df['prompt'].fillna('')
        df['output'] = df['output'].fillna('')

    except Exception as e: print(f"\nError loading TSV file: {e}"); traceback.print_exc(); sys.exit(1)

    # --- 4. Sort Data ---
    print("Sorting data for reporting...")
    try:
        # Convert sweep to numeric where possible for sorting, map 'baseline' to a low value
        def sort_key(sweep_val):
            if isinstance(sweep_val, str) and sweep_val.lower() == 'baseline':
                return -float('inf') # Sort baseline first
            try:
                return float(sweep_val)
            except (ValueError, TypeError):
                return float('inf') # Put non-numeric/non-baseline last

        df['sweep_sort_key'] = df['sweep'].apply(sort_key)
        # Ensure level and type are also sorted correctly (level as number, type as string)
        df['level'] = pd.to_numeric(df['level'], errors='coerce').fillna(0).astype(int) # Ensure level is numeric for sorting
        df_sorted = df.sort_values(by=['core_id', 'level', 'type', 'sweep_sort_key']).drop(columns=['sweep_sort_key'])
        print("Data sorted.")
    except Exception as e:
        print(f"Error during data sorting: {e}. Report may not be grouped correctly.")
        traceback.print_exc()
        df_sorted = df # Proceed with unsorted data if sorting fails

    # --- 5. Generate Markdown Report ---
    print(f"Generating report...")
    try:
        output_report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_report_path, 'w', encoding='utf-8') as f:
            f.write(f"# Prompt/Output Comparison Report\n\n")
            f.write(f"**Input TSV:** `{input_tsv_path.name}`\n")
            f.write(f"**Total Entries:** {len(df_sorted)}\n\n")
            f.write("---\n\n")

            last_core_id = None; last_level = None; last_type = None

            for index, row in df_sorted.iterrows():
                # Check for new Core ID
                if row['core_id'] != last_core_id:
                    f.write(f"## Core ID: {row['core_id']}\n\n")
                    last_core_id = row['core_id']
                    last_level = None; last_type = None # Reset level/type subgrouping

                # Check for new Level/Type subgroup
                if row['level'] != last_level or row['type'] != last_type:
                    f.write(f"### Level {row['level']} - Type: {row['type']}\n\n")
                    last_level = row['level']
                    last_type = row['type']
                    # Write the prompt only once per level/type group
                    f.write(f"**Prompt:**\n```\n{row['prompt']}\n```\n\n")

                # Write Sweep and Output for the current row
                f.write(f"**Sweep: {row['sweep']}**\n")
                f.write(f"```\n{row['output']}\n```\n")
                f.write("---\n\n") # Separator between sweeps

        print(f"Report successfully saved to: {output_report_path.resolve()}")

    except Exception as e:
        print(f"\nError writing report file: {e}")
        traceback.print_exc()

    print("Report generation complete.")

# --- END OF FILE generate_comparison_report.py ---