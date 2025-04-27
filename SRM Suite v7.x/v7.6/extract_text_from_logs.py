# --- START OF FILE extract_text_from_logs.py ---

import argparse
import sys
import traceback
import re
from pathlib import Path
import pandas as pd
import csv
import os
from collections import defaultdict

# --- Constants ---
LOGS_SUBFOLDER = "logs"
CAPTURE_SUBFOLDER = "capture"
DEFAULT_PROMPTSETS_SUBFOLDER = "promptsets"
BASELINE_SWEEP_TAG = "baseline"
VALID_GROUP_KEYS = ['core_id', 'type', 'level', 'sweep']

# --- Helper Functions (Copied/Adapted/Added) ---
# --- (Keep list_experiment_folders, select_experiment_folder, find_file_interactive here) ---
def list_experiment_folders(base_dir: Path, pattern: str = "*") -> list[Path]:
    """Lists directories matching a pattern within the base directory."""
    if not base_dir or not base_dir.is_dir(): return [] # Check None
    dirs = []
    try:
        for d in base_dir.iterdir():
            if d.is_dir() and not d.name.startswith('.'):
                 if d.match(pattern): dirs.append(d)
                 elif '*' not in pattern and '?' not in pattern and d.name.startswith(pattern): dirs.append(d)
    except Exception as e:
         print(f"Warning: Error listing directories in {base_dir}: {e}")
         return []
    return sorted(dirs)

def select_experiment_folder(base_dir: Path, prompt_text: str = "Select an experiment run directory:", pattern: str = "*") -> Path | None:
    """Interactively prompts the user to select a directory from a list."""
    run_dirs = list_experiment_folders(base_dir, pattern)
    if not run_dirs:
        print(f"Info: No directories matching '{pattern}' found in {base_dir}.")
        return None

    print(f"\n{prompt_text}")
    for i, dir_path in enumerate(run_dirs):
        print(f"  {i+1}: {dir_path.name}")

    selected_dir = None
    while selected_dir is None:
        try:
            choice_str = input(f"Enter the number of the directory (1-{len(run_dirs)}), or 0 to cancel: ")
            choice = int(choice_str)
            if choice == 0 : print("Selection cancelled."); return None
            choice_idx = choice - 1
            if 0 <= choice_idx < len(run_dirs):
                selected_dir = run_dirs[choice_idx]; print(f"Selected: {selected_dir.name}")
            else: print(f"Invalid choice. Please enter a number between 1 and {len(run_dirs)}, or 0.")
        except ValueError: print("Invalid input. Please enter a number.")
        except (EOFError, KeyboardInterrupt): print("\nSelection cancelled."); return None
    return selected_dir.resolve()

def find_file_interactive( directory: Path, pattern: str = "*.txt", file_type_desc: str = "file") -> Path | None:
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

def parse_structured_prompts(filepath: Path) -> list[dict] | None:
    """ Parses the structured prompt file based on the template format. (Copied) """
    if not filepath or not isinstance(filepath, Path): return None
    prompts_data = []; current_core_id = None; current_level = None
    level_pattern = re.compile(r"\[LEVEL (\d+)\]")
    core_id_pattern = re.compile(r">> CORE_ID:\s*(.*)")
    type_pattern = re.compile(r"^\s*([a-zA-Z_]+):\s*(.*)")
    line_num = 0
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('---') or line.startswith(">> PROPOSITION:"): continue
                core_id_match = core_id_pattern.match(line)
                if core_id_match:
                    raw_core_id = core_id_match.group(1).strip()
                    if not raw_core_id: current_core_id = None; continue
                    current_core_id = raw_core_id.replace(" ", "_")
                    current_core_id = re.sub(r'[^\w\-]+', '_', current_core_id)
                    current_level = None; continue
                if current_core_id:
                    level_match = level_pattern.match(line)
                    if level_match:
                        try: current_level = int(level_match.group(1))
                        except ValueError: current_level = None
                        continue
                if current_core_id and current_level is not None:
                    type_match = type_pattern.match(line)
                    if type_match:
                        prompt_type = type_match.group(1).lower().strip()
                        prompt_text = type_match.group(2).strip()
                        if not prompt_text: continue
                        prompts_data.append({'prompt_text': prompt_text, 'core_id': current_core_id, 'type': prompt_type, 'level': current_level})
    except FileNotFoundError: print(f"Error: Prompt file not found at '{filepath}'"); return None
    except Exception as e: print(f"Error parsing prompt file '{filepath}' (near line {line_num}): {e}"); traceback.print_exc(); return None
    if not prompts_data: print(f"Warning: No valid prompts parsed from '{filepath}'."); return []
    return prompts_data

def parse_log_prompt_header(header_content: str) -> dict | None:
    """ Parses 'core_id=X_type=Y_level=Z' """
    match = re.match(r"core_id=([\w-]+(?:_[\w-]+)*)_type=([\w]+)_level=(\d+)", header_content)
    if match:
        try:
            core_id = match.group(1); prompt_type = match.group(2); level = int(match.group(3))
            return {'core_id': core_id, 'type': prompt_type, 'level': level}
        except (IndexError, ValueError): print(f"Warning: Regex matched but failed to extract components from: '{header_content}'"); return None
    return None

# --- Main Script Logic ---
if __name__ == "__main__":
    # --- Argument parsing and setup code remains the same ---
    parser = argparse.ArgumentParser(description="Extracts prompts and generated text from run log files into a single combined TSV file.")
    parser.add_argument("--experiment_base_dir", type=str, required=True, help="Path to the base directory containing experiment run folders.")
    parser.add_argument("--intervention_run_dir", type=str, default=None, help="(Optional) Path to the specific intervention run directory. If omitted, will prompt.")
    parser.add_argument("--promptsets_dir", type=str, default=None, help=f"(Optional) Path to the directory containing prompt set files. If omitted, uses default locations and prompts.")
    parser.add_argument("--output_file", type=str, default=None, help="(Optional) Path for the output TSV file. Defaults to '[run_dir_name]_extracted.tsv' in the current directory.")
    args = parser.parse_args()
    base_dir = Path(args.experiment_base_dir)
    if not base_dir.is_dir(): print(f"Error: Experiment base directory not found: {base_dir}"); sys.exit(1)
    intervention_run_path = None
    if args.intervention_run_dir:
        intervention_run_path = Path(args.intervention_run_dir)
        if not intervention_run_path.is_dir(): print(f"Error: Provided intervention run directory not found: {intervention_run_path}"); sys.exit(1)
        print(f"Using provided intervention run directory: {intervention_run_path.name}")
    else:
        print(f"\nSelecting Intervention Run Directory from: {base_dir}")
        intervention_run_path = select_experiment_folder(base_dir, prompt_text="Select the INTERVENTION run directory to process:", pattern="INT-*")
        if intervention_run_path is None: print("No intervention run directory selected. Exiting."); sys.exit(1)
    prompt_file_path = None; promptsets_path = None
    if args.promptsets_dir:
        promptsets_path = Path(args.promptsets_dir)
        if not promptsets_path.is_dir(): print(f"Error: Provided promptsets directory not found: {promptsets_path}"); sys.exit(1)
    else:
        default_path1 = base_dir.parent / DEFAULT_PROMPTSETS_SUBFOLDER
        default_path2 = Path(".") / DEFAULT_PROMPTSETS_SUBFOLDER
        if default_path1.is_dir(): promptsets_path = default_path1; print(f"Info: Using default promptsets directory: {promptsets_path}")
        elif default_path2.is_dir(): promptsets_path = default_path2; print(f"Info: Using default promptsets directory: {promptsets_path} (relative to current dir)")
        else: print(f"Error: Default promptsets directory not found near {base_dir.parent} or current directory. Use --promptsets_dir."); sys.exit(1)
    if promptsets_path:
        print(f"\nSelecting Prompt File from: {promptsets_path}")
        prompt_file_path = find_file_interactive(promptsets_path, "*.txt", "prompt set file")
        if prompt_file_path is None: print("No prompt file selected. Exiting."); sys.exit(1)
    else: print("Error: Could not determine promptsets directory. Exiting."); sys.exit(1)
    print(f"Loading prompts from {prompt_file_path.name}...")
    structured_prompts = parse_structured_prompts(prompt_file_path)
    if structured_prompts is None: print("Failed to load or parse prompts. Exiting."); sys.exit(1)
    elif not structured_prompts: print("Warning: No prompts found in the selected prompt file."); sys.exit(1)
    prompt_lookup = {(p['core_id'], p['type'], p['level']): p['prompt_text'] for p in structured_prompts}
    print(f"Loaded {len(prompt_lookup)} prompts mapping.")
    log_dir_path = intervention_run_path / CAPTURE_SUBFOLDER / LOGS_SUBFOLDER
    if not log_dir_path.is_dir(): print(f"Error: Log directory not found: {log_dir_path}"); sys.exit(1)
    log_files = sorted(list(log_dir_path.glob("run_log_intervened.md")))
    log_file_path = None
    if not log_files: print(f"Error: No 'run_log_intervened.md' found in {log_dir_path}"); sys.exit(1)
    elif len(log_files) == 1: log_file_path = log_files[0]; print(f"Found log file: {log_file_path.name}")
    else: print(f"Warning: Multiple intervention log files found. Please select one:"); log_file_path = find_file_interactive(log_dir_path, "run_log_intervened.md", "intervention log file");
    if log_file_path is None: print("No log file selected. Exiting."); sys.exit(1)

    # --- 5. Parse Log, Combine Data (Linear Scan Logic with Debugging) ---
    print(f"Parsing log file and combining data: {log_file_path.name}...")

    # Regex patterns
    # <<< MODIFIED prompt_header_pattern to be slightly simpler/more specific >>>
    prompt_header_pattern = re.compile(r"^###\s+Prompt\s+\d+/\d+:\s+(core_id=.*)$", re.MULTILINE)
    intervention_header_pattern = re.compile(r"^#### Intervention:\s+sweep=([\w.-]+)\s*$", re.MULTILINE)
    text_block_pattern = re.compile(r"-\s+\*\*Generated Text:\*\*\n+```\n(.*?)\n?```", re.DOTALL)
    no_tokens_pattern = re.compile(r"-\s+\*\*Generated Text:\*\*\s+\(No new tokens\)", re.IGNORECASE)

    extracted_data = []
    parsing_errors = 0

    print("\nDEBUG: Starting Log Parsing...")

    try:
        log_content = log_file_path.read_text(encoding='utf-8')
        prompt_headers = []
        print("DEBUG: Searching for Prompt Headers...")
        for match in prompt_header_pattern.finditer(log_content):
            print(f"DEBUG: RAW Prompt Header Match Found: '{match.group(0).strip()}' at pos {match.start()}")
            key_string = match.group(1).strip()
            parsed = parse_log_prompt_header(key_string)
            if parsed:
                 print(f"DEBUG:   Successfully Parsed Key: {parsed}")
                 prompt_headers.append((match.start(), parsed)) # Use start_pos for ordering
            else:
                 print(f"DEBUG:   FAILED TO PARSE Key String: '{key_string}'")

        intervention_headers = [(m.start(), m.group(1)) for m in intervention_header_pattern.finditer(log_content)]
        text_blocks = [(m.start(), m.group(1).strip()) for m in text_block_pattern.finditer(log_content)]
        no_token_blocks = [(m.start(), "(No new tokens)") for m in no_tokens_pattern.finditer(log_content)]
        all_text_findings = sorted(text_blocks + no_token_blocks, key=lambda x: x[0])

        print(f"DEBUG: Found {len(prompt_headers)} VALID prompt headers.")
        print(f"DEBUG: Found {len(intervention_headers)} intervention headers.")
        print(f"DEBUG: Found {len(all_text_findings)} text blocks or 'no tokens' messages.")

        if not prompt_headers: print("ERROR: No valid prompt headers could be parsed from the log file."); sys.exit(1)

        for text_pos, text_content in all_text_findings:
            last_prompt_header = None
            for prompt_pos, key_info in reversed(prompt_headers):
                if prompt_pos < text_pos: last_prompt_header = (prompt_pos, key_info); break
            if not last_prompt_header: print(f"Warning: Found text at pos {text_pos} but no preceding prompt header. Skipping."); parsing_errors += 1; continue
            last_prompt_pos, current_key_info = last_prompt_header
            current_prompt_key = (current_key_info['core_id'], current_key_info['type'], current_key_info['level'])
            current_sweep = BASELINE_SWEEP_TAG
            for int_pos, sweep_val in reversed(intervention_headers):
                if last_prompt_pos < int_pos < text_pos: current_sweep = sweep_val; break
            original_prompt = prompt_lookup.get(current_prompt_key, "PROMPT NOT FOUND")
            if original_prompt == "PROMPT NOT FOUND": print(f"Warning: Prompt key {current_prompt_key} from log not found.")
            extracted_data.append({'core_id': current_key_info['core_id'], 'type': current_key_info['type'], 'level': current_key_info['level'], 'sweep': current_sweep, 'prompt': original_prompt, 'output': text_content})
            # print(f"DEBUG: Appended data for Key: {current_prompt_key}, Sweep: {current_sweep}")

    except Exception as e: print(f"\nAn error occurred during log parsing: {e}"); traceback.print_exc(); parsing_errors += 1

    # --- 6. Determine Output File Path and Save TSV ---
    if not extracted_data: print("\nError: No data was successfully extracted from the log file. Check log format or regex patterns."); sys.exit(1) # Added hint
    print(f"\nSuccessfully extracted {len(extracted_data)} entries.")
    if parsing_errors > 0: print(f"Encountered {parsing_errors} errors/warnings during parsing.")
    output_file_path = None
    if args.output_file: output_file_path = Path(args.output_file)
    else: default_filename = f"{intervention_run_path.name}_extracted.tsv"; output_file_path = Path(".") / default_filename
    print(f"Preparing to save combined data to: {output_file_path.resolve()}")
    try:
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        results_df = pd.DataFrame(extracted_data)
        cols_order = ['core_id', 'type', 'level', 'sweep', 'prompt', 'output']
        results_df = results_df[[col for col in cols_order if col in results_df.columns]]
        results_df.to_csv(output_file_path, sep='\t', index=False, encoding='utf-8', quoting=csv.QUOTE_MINIMAL)
        print("TSV file saved successfully.")
    except Exception as e: print(f"\nError saving TSV file: {e}"); traceback.print_exc()

    print("\nExtraction complete.")

# --- END OF FILE extract_text_from_logs.py ---