# --- START OF FILE analyze_textv2.py (Step 4b: Save Report) ---

import argparse
import sys
import traceback
import re
from pathlib import Path
from collections import defaultdict, Counter
import math
import pandas as pd
import csv
import os

# Make tqdm optional for progress bar
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# --- Constants ---
BASELINE_SWEEP_TAG = "baseline" # The literal tag used for the 'None' sweep

# --- Collapse Detection Function (from Step 1, Rev 2) ---
def is_output_collapsed(
    text: str,
    min_len_threshold: int = 6, # Consider very short outputs collapsed
    rep_char_threshold: float = 0.9, # More than 90% same char
    min_rep_pattern_len: int = 2, # Min length of pattern to check for repetition
    max_rep_pattern_len: int = 7, # Max length of pattern to check for repetition
    rep_pattern_threshold: float = 0.7 # If a short pattern makes up >70% of text
    ) -> bool:
    """
    Detects common collapse patterns in generated text.
    Returns True if a collapse pattern is detected, False otherwise.
    """
    if not text or not isinstance(text, str):
        return True # Treat empty or non-string as collapsed

    text = text.strip()
    n = len(text)

    if n < min_len_threshold:
        return True # Too short

    # Check for single character repetition (e.g., KKKK, '''''')
    if n > 0:
        char_counts = Counter(text)
        # Handle potential empty counter if text is whitespace only after strip
        if not char_counts:
             return True # Treat whitespace-only as collapsed for this check
        most_common_char_count = char_counts.most_common(1)[0][1]
        if most_common_char_count / n >= rep_char_threshold:
            return True

    # Check for dominant repeating substrings (more robust, slightly slower)
    if n > min_rep_pattern_len * 2: # Only check if text is long enough for repetition
        # Iterate through possible pattern lengths
        for pattern_len in range(min_rep_pattern_len, max_rep_pattern_len + 1):
            if pattern_len * 2 > n: continue # Pattern must be able to repeat at least once

            # Find all overlapping substrings of this length
            substrings = [text[i:i+pattern_len] for i in range(n - pattern_len + 1)]
            if not substrings: continue

            # Count occurrences of each substring
            substring_counts = Counter(substrings)
            most_common_substring, most_common_count = substring_counts.most_common(1)[0]

            # If a single substring repeats enough to dominate the text
            # Check if the total length covered by the most common substring meets the threshold
            if most_common_count > 1 and n > 0 and (most_common_count * pattern_len / n >= rep_pattern_threshold):
                # print(f"DEBUG: Dominant pattern '{most_common_substring}' found {most_common_count} times.") # DEBUG
                return True

    # Check for dominance of non-alphanumeric symbols (e.g., <<<<<, ※※※※)
    alphanumeric_chars = sum(1 for char in text if char.isalnum())
    if n > 0 and alphanumeric_chars / n < 0.1: # Less than 10% alphanumeric
         # Further check if it's mostly *one* symbol repeating (caught above)
         if char_counts and char_counts.most_common(1)[0][1] / n < rep_char_threshold: # Not caught by single char check
              non_alnum_symbols = sum(1 for char in text if not char.isalnum() and not char.isspace())
              if n > 0 and non_alnum_symbols / n > 0.8: # High proportion of non-space symbols
                   return True

    return False # If none of the above checks triggered


# --- Helper Functions ---
def find_file_interactive( directory: Path, pattern: str = "*.tsv", file_type_desc: str = "file") -> Path | None:
    """ Finds files matching pattern, prompts user if multiple found. """
    if not directory or not directory.is_dir():
        print(f"Error: Directory for interactive search not found or invalid: {directory}")
        return None
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

def calculate_text_metrics(text: str) -> dict:
    """ Calculates basic textual coherence metrics. """
    metrics = {
        'total_tokens': 0, 'unique_tokens': 0, 'num_sentences': 0,
        'avg_sentence_length': float('nan'), 'num_repeated_bigrams': 0,
        'token_entropy': 0.0, 'error': None
    }
    if not text or not isinstance(text, str) or not text.strip() or text == "(No new tokens)":
        metrics['error'] = "Empty or no generated text"
        for k in metrics:
            if k != 'error' and isinstance(metrics[k], (int, float)): metrics[k] = float('nan')
        return metrics
    try:
        # Basic tokenization by splitting on whitespace. More sophisticated tokenizers exist.
        tokens = re.findall(r'\S+', text) # Finds sequences of non-whitespace characters
        metrics['total_tokens'] = len(tokens)

        if metrics['total_tokens'] == 0:
             metrics['error'] = "No tokens found after splitting"
             for k in metrics:
                 if k != 'error' and k != 'total_tokens' and isinstance(metrics[k], (int, float)):
                     metrics[k] = float('nan')
             return metrics

        token_counts = Counter(tokens)
        metrics['unique_tokens'] = len(token_counts)

        # Sentence counting using common punctuation followed by space or end of string
        sentences = re.split(r'[.?!]\s+|$', text)
        # Filter out empty strings resulting from multiple delimiters together or at the end
        sentences = [s for s in sentences if s.strip()]
        metrics['num_sentences'] = len(sentences)
        if metrics['num_sentences'] == 0 and metrics['total_tokens'] > 0:
            metrics['num_sentences'] = 1 # Treat as one sentence if tokens exist but no end punctuation found

        metrics['avg_sentence_length'] = metrics['total_tokens'] / metrics['num_sentences'] if metrics['num_sentences'] > 0 else float('nan')

        # Bigram repetition calculation
        if metrics['total_tokens'] >= 2:
            bigrams = list(zip(tokens, tokens[1:]))
            bigram_counts = Counter(bigrams)
            # Count number of bigram *types* that repeat (occur more than once)
            repeated_bigram_types_count = sum(1 for count in bigram_counts.values() if count > 1)
            metrics['num_repeated_bigrams'] = repeated_bigram_types_count
        else:
            metrics['num_repeated_bigrams'] = 0

        # Token Entropy calculation
        entropy = 0.0
        # Ensure there's something to calculate entropy from (more than 1 token and more than 1 unique type)
        if metrics['total_tokens'] > 1 and metrics['unique_tokens'] > 1:
             # Use log base 2 for entropy in bits
             log_total_tokens = math.log2(metrics['total_tokens'])
             for count in token_counts.values():
                 # Calculate probability p_i = count / total_tokens
                 p_i = count / metrics['total_tokens']
                 # Entropy contribution of this token type: -p_i * log2(p_i)
                 if p_i > 0: # Avoid log(0)
                    entropy -= p_i * math.log2(p_i)

        metrics['token_entropy'] = entropy if metrics['total_tokens'] > 0 else 0.0

    except Exception as e:
        metrics['error'] = f"Calculation error: {e}"
        traceback.print_exc()
        # Set numeric metrics to NaN on error
        for k in metrics:
            if k != 'error' and isinstance(metrics[k], (int, float)):
                metrics[k] = float('nan')
    return metrics


# --- Main Script Logic ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculates and analyzes textual coherence metrics from a combined TSV data file, identifying improvements over collapsed baselines.")
    parser.add_argument("--input_tsv_file", type=str, default=None, help="(Optional) Path to the input TSV file (e.g., 'RUN_NAME_extracted.tsv'). If omitted, prompts user to select.")
    parser.add_argument("--search_dir", type=str, default=".", help="Directory to search for the TSV file if --input_tsv_file is omitted.")
    parser.add_argument("--output_metrics_csv", type=str, default=None, help="(Optional) Path to save the detailed metrics CSV file. Defaults to '[input_base]_metrics.csv'.")
    parser.add_argument("--output_report_file", type=str, default=None, help="(Optional) Path to save the improvement report file. Defaults to '[input_base]_improvement_report.md'.") # Added argument
    parser.add_argument("--report_improvements", action='store_true', help="Generate and display/save the detailed report of prompts where interventions improved over collapsed baselines.") # Modified help text slightly
    args = parser.parse_args()

    # --- 1. Find and Select Input TSV File ---
    input_tsv_path = None
    if args.input_tsv_file:
        input_tsv_path = Path(args.input_tsv_file)
        if not input_tsv_path.is_file():
            print(f"Error: Provided input TSV file not found: {input_tsv_path}")
            sys.exit(1)
        print(f"Using provided input TSV file: {input_tsv_path}")
    else:
        search_path = Path(args.search_dir)
        if not search_path.is_dir():
            print(f"Error: Search directory for TSV files not found: {search_path}")
            sys.exit(1)
        print(f"\nSelecting Input TSV File from: {search_path}")
        input_tsv_path = find_file_interactive(search_path, "*_extracted.tsv", "extracted data TSV")
        if input_tsv_path is None:
            print("No input TSV file selected. Exiting.")
            sys.exit(1)

    # --- 2. Load Data from TSV ---
    print(f"\nLoading data from: {input_tsv_path.name}")
    try:
        results_df = pd.read_csv(
            input_tsv_path,
            sep='\t',
            quoting=csv.QUOTE_MINIMAL,
            lineterminator='\n',
            keep_default_na=False,
            na_values=[''],
            dtype={'sweep': str}
        )
        print(f"Loaded {len(results_df)} rows.")

        results_df.columns = results_df.columns.str.strip()
        print(f"DEBUG: Cleaned DataFrame Columns: {list(results_df.columns)}")

        expected_cols = ['core_id', 'type', 'level', 'sweep', 'prompt', 'output']
        if not all(col in results_df.columns for col in expected_cols):
            missing_cols = [col for col in expected_cols if col not in results_df.columns]
            print(f"Error: Input TSV missing expected columns after cleaning. Missing: {missing_cols}. Found: {list(results_df.columns)}")
            sys.exit(1)

        results_df['prompt'] = results_df['prompt'].fillna('')
        results_df['output'] = results_df['output'].fillna('')

    except FileNotFoundError:
        print(f"\nError: Input file not found at {input_tsv_path}")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"\nError: Input file {input_tsv_path} is empty.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError loading or processing TSV file {input_tsv_path}: {e}")
        print("Please ensure the file is correctly formatted (Tab-Separated). Check quoting and delimiters.")
        traceback.print_exc()
        sys.exit(1)


    # --- 3. Calculate Metrics for Each Row ---
    print("\nCalculating coherence metrics for each output...")
    metrics_list = []
    row_iterator = results_df.iterrows()
    if TQDM_AVAILABLE:
        row_iterator = tqdm(results_df.iterrows(), total=len(results_df), desc="Calculating metrics", unit="row", ncols=100)

    for index, row in row_iterator:
        output_text = str(row.get('output', ''))
        metrics_list.append(calculate_text_metrics(output_text))

    metrics_df = pd.DataFrame(metrics_list, index=results_df.index)
    results_df = pd.merge(results_df, metrics_df, left_index=True, right_index=True, suffixes=('', '_metric'))
    print("Metrics calculation complete.")

    # --- Step 2: Apply Collapse Detection and Identify Collapsed Baselines ---
    print("\nApplying collapse detection to all outputs...")
    collapsed_baseline_prompt_ids = set() # Initialize empty set here
    baseline_found_step2 = False # Flag specifically for this step's baseline dependency

    try:
        if 'output' in results_df.columns:
            apply_iterator = results_df['output']
            if TQDM_AVAILABLE:
                 apply_iterator = tqdm(results_df['output'], total=len(results_df), desc="Detecting collapse", unit="row", ncols=100)
            results_df['is_collapsed'] = [is_output_collapsed(str(text)) for text in apply_iterator]
            print("Collapse detection applied.")

            if 'sweep' in results_df.columns:
                results_df['sweep'] = results_df['sweep'].astype(str)
                baseline_rows_check = results_df[results_df['sweep'] == BASELINE_SWEEP_TAG]
                if not baseline_rows_check.empty:
                    baseline_found_step2 = True
                else:
                    print("Note: Baseline tag not found, cannot identify collapsed baselines.")
            else:
                print("Error: 'sweep' column missing, cannot identify baseline rows.")

            if baseline_found_step2 and 'is_collapsed' in results_df.columns:
                collapsed_baselines_df = results_df[
                    (results_df['sweep'] == BASELINE_SWEEP_TAG) & \
                    (results_df['is_collapsed'] == True)
                ]
                print(f"Found {len(collapsed_baselines_df)} collapsed baseline outputs.")

                prompt_group_cols = ['core_id', 'type', 'level']
                if all(col in collapsed_baselines_df.columns for col in prompt_group_cols):
                    collapsed_baseline_prompt_ids = set(
                        collapsed_baselines_df[prompt_group_cols].itertuples(index=False, name=None)
                    )
                    print(f"Identified {len(collapsed_baseline_prompt_ids)} unique prompts with collapsed baselines.")
                else:
                    print("Warning: Missing ID columns ('core_id', 'type', 'level') in collapsed baseline data. Cannot create ID set.")

        else:
             print("Error: 'output' column not found in DataFrame. Cannot apply collapse detection.")
             results_df['is_collapsed'] = False

    except Exception as e:
         print(f"\nError during collapse detection or baseline identification: {e}")
         traceback.print_exc()
         results_df['is_collapsed'] = False

    # --- Step 3: Identify Interventions Showing Improvement ---
    print("\nIdentifying interventions that improve over collapsed baselines...")
    results_df['intervention_improves_baseline'] = False # Initialize column
    improved_prompt_count = 0
    prompts_checked_for_improvement = set()

    if 'is_collapsed' in results_df.columns and collapsed_baseline_prompt_ids:
        prompt_group_cols = ['core_id', 'type', 'level']
        if all(col in results_df.columns for col in prompt_group_cols):
            intervention_rows_indices = results_df[results_df['sweep'] != BASELINE_SWEEP_TAG].index
            iter_indices = intervention_rows_indices
            if TQDM_AVAILABLE:
                 iter_indices = tqdm(intervention_rows_indices, total=len(intervention_rows_indices), desc="Checking improvements", unit="row", ncols=100)

            for index in iter_indices:
                row = results_df.loc[index]
                prompt_key = tuple(row[col] for col in prompt_group_cols)
                if prompt_key in collapsed_baseline_prompt_ids:
                    if not row['is_collapsed']:
                         results_df.loc[index, 'intervention_improves_baseline'] = True
                         if prompt_key not in prompts_checked_for_improvement:
                              improved_prompt_count += 1
                              prompts_checked_for_improvement.add(prompt_key)

            print(f"Checked {len(intervention_rows_indices)} intervention outputs.")
            print(f"Found improvements for {improved_prompt_count} out of {len(collapsed_baseline_prompt_ids)} prompts with collapsed baselines.")
        else:
             print("Error: Missing ID columns ('core_id', 'type', 'level'). Cannot check for improvements.")
    elif not collapsed_baseline_prompt_ids:
         print("Skipping improvement check as no collapsed baselines were identified.")
    else:
        print("Error: 'is_collapsed' column missing. Cannot check for improvements.")

    # --- 4. Baseline Check (Existing check for delta analysis) ---
    if 'sweep' not in results_df.columns:
        print(f"\nError: 'sweep' column not found in the DataFrame. Cannot perform baseline analysis.")
        baseline_found = False
    else:
        results_df['sweep'] = results_df['sweep'].astype(str)
        baseline_rows = results_df[results_df['sweep'] == BASELINE_SWEEP_TAG]
        baseline_found = not baseline_rows.empty
        if not baseline_found:
            print("\n" + "*"*70 + f"\nWARNING: No baseline sweep tag ('{BASELINE_SWEEP_TAG}') found for delta analysis!\nDelta calculations will be skipped.\n" + "*"*70 + "\n")
        else:
            print(f"\nFound {len(baseline_rows)} baseline entries. Proceeding with delta analysis.")


    # --- 5. Console Analysis & Summary (Existing) ---
    print("\n--- Analysis Summary ---")
    metric_cols_numeric = [col for col in metrics_df.columns if col != 'error' and pd.api.types.is_numeric_dtype(results_df[col])]

    if not metric_cols_numeric:
        print("\nWarning: No numeric metric columns found to analyze.")
    else:
        print("\nAverage Metrics per Sweep Value:")
        try:
            results_df['sweep_str'] = results_df['sweep'].astype(str)
            results_df['sweep_numeric_sort'] = pd.to_numeric(results_df['sweep'], errors='coerce')
            avg_metrics_per_sweep = results_df.groupby('sweep_str')[metric_cols_numeric].mean()
            avg_metrics_per_sweep['sort_key'] = pd.to_numeric(avg_metrics_per_sweep.index, errors='coerce')
            avg_metrics_per_sweep['sort_key'] = avg_metrics_per_sweep['sort_key'].fillna(float('inf'))
            avg_metrics_per_sweep = avg_metrics_per_sweep.sort_values(by='sort_key').drop(columns=['sort_key'])
            print(avg_metrics_per_sweep.to_string(float_format="%.3f"))
        except Exception as e:
            print(f"Could not calculate average metrics per sweep: {e}")
            traceback.print_exc()


    if baseline_found and metric_cols_numeric: # Use flag from section 4
        print(f"\nAnalysis vs Internal Baseline ('{BASELINE_SWEEP_TAG}'):")
        try:
            prompt_group_cols = ['core_id', 'type', 'level']
            if not all(col in results_df.columns for col in prompt_group_cols):
                 print(f"Error: Missing grouping columns for delta analysis. Required: {prompt_group_cols}")
            else:
                baseline_metrics_df = baseline_rows.groupby(prompt_group_cols, as_index=False).first()
                baseline_metrics_df = baseline_metrics_df.set_index(prompt_group_cols)[metric_cols_numeric]
                intervention_df = results_df[results_df['sweep'] != BASELINE_SWEEP_TAG].copy()
                intervention_df['sweep_numeric'] = pd.to_numeric(intervention_df['sweep'], errors='coerce')

                delta_results = []
                for index, row in intervention_df.iterrows():
                    prompt_key = tuple(row[col] for col in prompt_group_cols)
                    if prompt_key in baseline_metrics_df.index:
                        baseline_vals_series = baseline_metrics_df.loc[prompt_key]
                        delta_dict = {'core_id': row['core_id'], 'type': row['type'], 'level': row['level'], 'sweep': row['sweep']}
                        for col in metric_cols_numeric:
                            if pd.notna(row[col]) and pd.notna(baseline_vals_series[col]):
                                delta_dict[f"{col}_delta"] = row[col] - baseline_vals_series[col]
                            else:
                                delta_dict[f"{col}_delta"] = float('nan')
                        delta_results.append(delta_dict)

                if delta_results:
                    full_delta_df = pd.DataFrame(delta_results)
                    delta_cols = [f"{col}_delta" for col in metric_cols_numeric]

                    if 'sweep' in full_delta_df.columns:
                        print("\nAverage Metric Delta vs Baseline per Sweep:")
                        full_delta_df['sweep_numeric_sort'] = pd.to_numeric(full_delta_df['sweep'], errors='coerce')
                        avg_delta_per_sweep = full_delta_df.groupby('sweep')[delta_cols].mean()
                        avg_delta_per_sweep['sort_key'] = pd.to_numeric(avg_delta_per_sweep.index, errors='coerce')
                        avg_delta_per_sweep['sort_key'] = avg_delta_per_sweep['sort_key'].fillna(float('inf'))
                        avg_delta_per_sweep = avg_delta_per_sweep.sort_values(by='sort_key').drop(columns=['sort_key'])
                        print(avg_delta_per_sweep.to_string(float_format="%+.3f"))

                        analysis_metric = 'token_entropy'
                        delta_col_name = f"{analysis_metric}_delta"
                        if delta_col_name in full_delta_df.columns and not full_delta_df[delta_col_name].isnull().all():
                            print(f"\nTop 5 Largest Increases in {analysis_metric.replace('_',' ').title()} vs Baseline:")
                            top_increases = full_delta_df.nlargest(5, delta_col_name)
                            print(top_increases[['core_id', 'type', 'level', 'sweep', delta_col_name]].to_string(index=False, float_format="%+.3f"))

                            print(f"\nTop 5 Largest Decreases in {analysis_metric.replace('_',' ').title()} vs Baseline:")
                            top_decreases = full_delta_df.nsmallest(5, delta_col_name)
                            print(top_decreases[['core_id', 'type', 'level', 'sweep', delta_col_name]].to_string(index=False, float_format="%+.3f"))
                        else:
                             print(f"\nNote: Cannot show top changes for '{analysis_metric}'. Delta column missing or all NaN.")
                    else:
                        print("\nWarning: 'sweep' column missing in delta results, cannot group by sweep.")
                else:
                    print("\nNo delta results calculated (possibly due to missing baselines or non-numeric data).")
        except KeyError as e:
             print(f"\nError during delta analysis: Missing expected column - {e}. Check TSV headers and data.")
             traceback.print_exc()
        except Exception as e:
            print(f"\nError during delta analysis: {e}")
            traceback.print_exc()
    elif not baseline_found:
         print("\nSkipping delta analysis because baseline sweep files were not found in the input TSV for delta calculation.")
    elif not metric_cols_numeric:
         print("\nSkipping delta analysis because no numeric metric columns were found for delta calculation.")

    # --- Step 4 & 4b: Generate and Optionally Save Focused Improvement Report ---
    if args.report_improvements:
        print("\n--- Collapsed Baseline Improvement Report ---") # Console header

        # Determine output report file path
        report_file_path = None
        if args.output_report_file:
            report_file_path = Path(args.output_report_file)
        else:
            default_report_filename = f"{input_tsv_path.stem}_improvement_report.md"
            report_file_path = input_tsv_path.parent / default_report_filename

        # Proceed only if improvements were found and necessary data exists
        if improved_prompt_count > 0 and 'intervention_improves_baseline' in results_df.columns:
            # Filter for rows marked as improvements
            improvement_rows_df = results_df[results_df['intervention_improves_baseline'] == True]

            prompt_group_cols = ['core_id', 'type', 'level'] # Define here for access
            if all(col in results_df.columns for col in prompt_group_cols):
                # Create multi-index for faster checking
                improvement_prompt_index = pd.MultiIndex.from_frame(improvement_rows_df[prompt_group_cols])

                # Filter baseline rows efficiently
                baseline_subset_df = results_df[
                    (results_df['sweep'] == BASELINE_SWEEP_TAG) &
                    (pd.MultiIndex.from_frame(results_df[prompt_group_cols]).isin(improvement_prompt_index))
                ].groupby(prompt_group_cols).first().reset_index() # Ensure unique baseline per group

                baseline_outputs_map = baseline_subset_df.set_index(prompt_group_cols)['output'].to_dict()

                # Group the improvement rows by the prompt identifier
                grouped_improvements = improvement_rows_df.groupby(prompt_group_cols)

                print(f"\nDisplaying details for {len(grouped_improvements)} prompts where intervention improved over collapsed baseline:") # Console msg
                if report_file_path: # Inform user about saving
                     print(f"Saving detailed improvement report to: {report_file_path.resolve()}")

                try:
                    # Open file for writing the report
                    with open(report_file_path, 'w', encoding='utf-8') as f:
                        # --- WRITE REPORT HEADER TO FILE ---
                        f.write("# Collapsed Baseline Improvement Report\n\n")
                        f.write(f"Generated from: `{input_tsv_path.name}`\n")
                        f.write(f"Found improvements for {improved_prompt_count} out of {len(collapsed_baseline_prompt_ids)} prompts with collapsed baselines.\n\n")

                        report_count = 0
                        for prompt_key, group_df in grouped_improvements:
                             report_count += 1
                             # --- CONSOLE ---
                             print(f"\n--- Prompt {report_count} / {len(grouped_improvements)} ---")
                             # --- FILE ---
                             f.write(f"---\n\n## Prompt {report_count} / {len(grouped_improvements)}\n\n")

                             # --- Print/Write Prompt ID ---
                             if len(prompt_key) == 3:
                                 id_str = f"ID: core_id='{prompt_key[0]}', type='{prompt_key[1]}', level='{prompt_key[2]}'"
                                 print(id_str)
                                 f.write(f"{id_str}\n")
                             else:
                                  print(f"ID: {prompt_key} (Unexpected format)")
                                  f.write(f"ID: {prompt_key} (Unexpected format)\n")


                             # --- Print/Write Baseline Output ---
                             baseline_output = baseline_outputs_map.get(prompt_key, "ERROR: Baseline output not found!")
                             # Simple print, avoid complex formatting issues in console
                             print(f"\nCollapsed Baseline Output (Sweep={BASELINE_SWEEP_TAG}):")
                             print(f"```\n{baseline_output}\n```") # Use markdown style block
                             f.write(f"\n**Collapsed Baseline Output (Sweep={BASELINE_SWEEP_TAG}):**\n")
                             f.write(f"```\n{baseline_output}\n```\n")


                             # --- Print/Write Improved Interventions ---
                             print("\nImproved Intervention Output(s):")
                             f.write(f"\n**Improved Intervention Output(s):**\n")
                             group_df['sweep_numeric_sort'] = pd.to_numeric(group_df['sweep'], errors='coerce')
                             sorted_group = group_df.sort_values(by='sweep_numeric_sort')

                             for idx, row in sorted_group.iterrows():
                                 sweep_val = row['sweep']
                                 output_text = str(row.get('output', ''))
                                 # Indent output for readability
                                 indented_output = "\n    ".join(output_text.split('\n'))

                                 print(f"  Sweep: {sweep_val}")
                                 print(f"  Output:\n    ```\n    {indented_output}\n    ```")
                                 print("-" * 20)
                                 f.write(f"\n*   **Sweep: {sweep_val}**\n") # Use markdown list
                                 f.write(f"    ```\n    {indented_output}\n    ```\n")


                             print("=" * 70)
                             f.write("\n" + "=" * 70 + "\n\n")

                    print("Improvement report saved successfully.") # Console confirmation
                except Exception as e:
                    print(f"\nError saving improvement report file to {report_file_path}: {e}") # Console error
                    traceback.print_exc()

            else:
                print("Error: Missing ID columns in DataFrame, cannot generate improvement report details.")

        # Cases where no improvements found or data missing
        elif improved_prompt_count == 0 and collapsed_baseline_prompt_ids:
             print("No interventions showed improvement over the identified collapsed baselines.")
        elif not collapsed_baseline_prompt_ids:
             print("No collapsed baselines were identified, so no improvement report generated.")
        else:
             print("Could not generate improvement report due to missing data ('intervention_improves_baseline' column).")
    elif not args.report_improvements:
        print("\nSkipping detailed improvement report generation (use --report_improvements flag to enable).")

    # --- 6. Save Metrics CSV (Now includes 'intervention_improves_baseline') ---
    output_csv_path = None
    if args.output_metrics_csv:
        output_csv_path = Path(args.output_metrics_csv)
    else:
        default_csv_filename = f"{input_tsv_path.stem}_metrics.csv"
        output_csv_path = input_tsv_path.parent / default_csv_filename

    if output_csv_path:
        print(f"\nSaving detailed metrics DataFrame to: {output_csv_path.resolve()}")
        try:
            id_cols = ['core_id', 'type', 'level', 'sweep']
            metric_cols_present = [col for col in metrics_df.columns if col in results_df.columns]
            error_col = ['error'] if 'error' in results_df.columns else []
            is_collapsed_col = ['is_collapsed'] if 'is_collapsed' in results_df.columns else []
            improvement_col = ['intervention_improves_baseline'] if 'intervention_improves_baseline' in results_df.columns else []

            # Define the desired order, including the new columns
            cols_to_save_order = id_cols + metric_cols_present + is_collapsed_col + improvement_col + error_col
            cols_to_save_existing = [col for col in cols_to_save_order if col in results_df.columns]

            df_to_save = results_df[cols_to_save_existing]

            output_csv_path.parent.mkdir(parents=True, exist_ok=True)
            df_to_save.to_csv(output_csv_path, index=False, encoding='utf-8', quoting=csv.QUOTE_MINIMAL)
            print("Metrics CSV saved successfully.")
        except Exception as e:
            print(f"\nError saving metrics CSV file: {e}")
            traceback.print_exc()

    print("\nAnalysis complete.")

# --- END OF FILE analyze_textv2.py (Step 4b: Save Report) ---