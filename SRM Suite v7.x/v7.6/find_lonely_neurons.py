# --- START OF FILE find_lonely_neurons.py ---

import torch
import argparse
import warnings
import sys
import traceback
import re  # <--- Make sure re is imported for the parser function
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import math # For sqrt

# Local application imports
from transformer_lens import HookedTransformer, utils as tl_utils
import utils # Import our updated utility functions

# --- Constants ---
DEFAULT_ID_PREFIX = "STATS" # Default prefix for neuron statistics runs
DEFAULT_OUTPUT_FILENAME = "neuron_stats.csv"
NUM_LAYERS = 12 # For GPT-2 Small
DIMENSION = utils.DIMENSION # 3072 for GPT-2 Small MLP

# --- Helper Functions ---

# --- Paste the parse_structured_prompts function definition here ---
def parse_structured_prompts(filepath: Path) -> list[dict] | None:
    """Parses the structured prompt file based on the template format."""
    prompts_data = []
    current_core_id = None
    current_level = None
    level_pattern = re.compile(r"\[LEVEL (\d+)\]")
    core_id_pattern = re.compile(r">> CORE_ID:\s*(.*)")
    type_pattern = re.compile(r"^\s*([a-zA-Z_]+):\s*(.*)")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('---') or line.startswith(">> PROPOSITION:"):
                    continue

                core_id_match = core_id_pattern.match(line)
                if core_id_match:
                    # Sanitize core_id: replace spaces with underscores
                    current_core_id = core_id_match.group(1).strip().replace(" ", "_")
                    current_level = None
                    continue

                level_match = level_pattern.match(line)
                if level_match:
                    current_level = int(level_match.group(1))
                    continue

                type_match = type_pattern.match(line)
                if type_match:
                    prompt_type = type_match.group(1).lower()
                    prompt_text = type_match.group(2).strip()
                    if current_core_id and current_level is not None:
                        prompts_data.append({
                            'prompt_text': prompt_text,
                            'core_id': current_core_id,
                            'type': prompt_type,
                            'level': current_level
                        })
                    else:
                        print(f"Warning: Found prompt '{prompt_text}' on line {line_num} but CORE_ID or LEVEL not set. Skipping.")
                        continue
    except FileNotFoundError:
        print(f"Error: Prompt file not found at '{filepath}'")
        return None
    except Exception as e:
        print(f"Error parsing prompt file '{filepath}': {e}")
        traceback.print_exc()
        return None

    if not prompts_data:
        print(f"Warning: No valid prompts parsed from '{filepath}'. Check file format and content.")
        return None # Indicate failure

    print(f"Successfully parsed {len(prompts_data)} prompts from {filepath.name}.")
    return prompts_data
# --- End of pasted function ---

# --- Main Script Logic ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture MLP activation statistics (mean, stddev) across all layers for structured prompts. Identifies potentially 'lonely' neurons.")
    parser.add_argument("--prompt_file", type=str, required=True, help="Path to the structured prompt file (e.g., promptsets/lonely_neuron_promptset.txt).")
    parser.add_argument("--experiment_base_dir", type=str, required=True, help="Path to the base directory where the unique run directory will be created (e.g., ./experiments).")
    parser.add_argument("--id_prefix", type=str, default=DEFAULT_ID_PREFIX, help=f"Prefix for the Run ID (e.g., 'STATS', 'LONELY'). Default: {DEFAULT_ID_PREFIX}.")
    parser.add_argument("--output_stats_file", type=str, default=DEFAULT_OUTPUT_FILENAME, help=f"Filename for the output CSV containing neuron statistics (saved in data/ subdir). Default: {DEFAULT_OUTPUT_FILENAME}")
    parser.add_argument("--top_n_report", type=int, default=10, help="Number of top 'loneliest' (lowest mean) and 'most active' (highest mean) neurons to report in the log/console. Default: 10.")

    args = parser.parse_args()

    # --- Generate Run ID and Timestamp ---
    try:
        run_id = utils.generate_run_id(args.id_prefix)
        timestamp_str = utils.get_formatted_timestamp()
        print(f"Generated Run ID: {run_id}")
        print(f"Generated Timestamp: {timestamp_str}")
    except Exception as e:
        print(f"Fatal Error: Could not generate Run ID or Timestamp: {e}")
        sys.exit(1)

    # --- Setup Paths ---
    base_dir = Path(args.experiment_base_dir)
    prompt_path = Path(args.prompt_file)
    if not prompt_path.is_file():
         print(f"Error: Prompt file not found at {prompt_path}"); sys.exit(1)
    prompt_basename_sanitized = utils.sanitize_label(prompt_path.stem)

    # Construct directory name
    run_dir_name = f"{run_id}_neuron_stats_{prompt_basename_sanitized}_{timestamp_str}"
    run_path = base_dir / run_dir_name

    # Subdirectories using utils constants
    capture_path = run_path / utils.CAPTURE_SUBFOLDER
    data_dir = capture_path / utils.DATA_SUBFOLDER # Save CSV here
    log_dir = capture_path / utils.LOGS_SUBFOLDER
    metadata_dir = capture_path / utils.METADATA_SUBFOLDER

    try:
        metadata_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(exist_ok=True)
        log_dir.mkdir(exist_ok=True)
        print(f"Created unique run directory structure: {run_path}")
        print(f" -> Capture outputs will be saved in: {capture_path}")
    except OSError as e:
        print(f"Error creating output directories in '{run_path}': {e}", file=sys.stderr)
        traceback.print_exc(); sys.exit(1)

    # Define internal filenames
    stats_filename = args.output_stats_file
    log_filename = f"run_log_neuron_stats.md"
    metadata_filename = f"run_metadata_neuron_stats.json"
    # Full paths
    stats_path = data_dir / stats_filename
    log_path = log_dir / log_filename
    metadata_path = metadata_dir / metadata_filename

    # --- Parse Prompts ---
    # Now calls the local function defined above
    parsed_prompts = parse_structured_prompts(prompt_path)
    if not parsed_prompts:
        print("Exiting due to prompt parsing failure.")
        sys.exit(1)

    # --- Load Model ---
    print("\nLoading GPT-2 Small model...")
    try:
        model = HookedTransformer.from_pretrained("gpt2")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        print(f"Using device: {device}")
        tokenizer = model.tokenizer
        if tokenizer.pad_token is None:
            print("Tokenizer does not have a pad token; setting to eos_token.")
            tokenizer.pad_token = tokenizer.eos_token
            if hasattr(model.config, 'pad_token_id'):
                 model.config.pad_token_id = tokenizer.eos_token_id

        # We will hook all layers
        print(f"Preparing to capture MLP activations from layers 0-{NUM_LAYERS-1} (dim: {DIMENSION})")
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    # --- Prepare Metadata ---
    run_metadata = {
        "run_id": run_id,
        "script_name": Path(__file__).name,
        "run_type": "neuron_stats_capture",
        "model_name": "gpt2",
        "num_layers_analyzed": NUM_LAYERS,
        "activation_dimension": DIMENSION,
        "prompt_file_path": str(prompt_path),
        "prompt_file_relative": str(prompt_path.relative_to(base_dir)) if prompt_path.is_relative_to(base_dir) else str(prompt_path),
        "prompt_file_basename": prompt_path.stem,
        "prompt_file_basename_sanitized": prompt_basename_sanitized,
        "num_prompts_parsed": len(parsed_prompts),
        "run_timestamp_str": timestamp_str,
        "run_directory": str(run_path),
        "run_directory_name": run_path.name,
        "capture_directory_relative": str(capture_path.relative_to(run_path)),
        "output_stats_file_relative": str(stats_path.relative_to(capture_path)), # Path to CSV
        "output_log_file_relative": str(log_path.relative_to(capture_path)),
        "output_metadata_file_relative": str(metadata_path.relative_to(capture_path)),
        "device": device,
        "cli_args": vars(args),
        "total_tokens_processed": None,
        "neurons_analyzed": None,
        "prompts_processed_successfully": None,
        "prompts_skipped_or_failed": None,
    }
    if utils.save_json_metadata(metadata_path, run_metadata):
         print(f"Initial metadata saved to: {metadata_path}")
    else:
         print(f"Warning: Could not save initial metadata file '{metadata_path}'. Continuing run.")

    # --- Main Processing ---
    print(f"\nStarting neuron statistics capture run ({run_id}) for {len(parsed_prompts)} prompts across {NUM_LAYERS} layers.")

    # Data structure for accumulating statistics: {(layer, neuron_idx): {'sum': float, 'sum_sq': float, 'count': int}}
    stats_accumulator = defaultdict(lambda: {'sum': 0.0, 'sum_sq': 0.0, 'count': 0})
    total_token_count = 0
    processed_prompts_count = 0
    skipped_prompt_count = 0

    # --- Define Hook Function ---
    def stats_hook(activation_tensor, hook):
        # activation_tensor shape: [batch, seq_len, d_mlp]
        # hook.name format e.g.: 'blocks.11.mlp.hook_post'
        try:
            layer_index_str = hook.name.split('.')[1]
            layer_index = int(layer_index_str)

            # Detach, move to CPU, convert to float32 numpy array for processing
            # Process entire tensor at once for efficiency
            acts_np = activation_tensor.detach().cpu().numpy().astype(np.float32)

            # Sum across batch and sequence length dimensions for each neuron
            # Shape becomes (d_mlp,)
            sum_activations = np.sum(acts_np, axis=(0, 1))
            sum_sq_activations = np.sum(np.square(acts_np), axis=(0, 1))
            # Number of tokens contributing to this hook call: batch_size * seq_len
            num_tokens = acts_np.shape[0] * acts_np.shape[1]

            # Update accumulator for each neuron in this layer
            for neuron_idx in range(DIMENSION):
                neuron_key = (layer_index, neuron_idx)
                stats_accumulator[neuron_key]['sum'] += sum_activations[neuron_idx]
                stats_accumulator[neuron_key]['sum_sq'] += sum_sq_activations[neuron_idx]
                stats_accumulator[neuron_key]['count'] += num_tokens

        except Exception as e:
            print(f"\nError in stats_hook for hook {hook.name}: {e}")
            traceback.print_exc()
    # --- End Hook Function ---

    # --- Register Hooks for ALL MLP Layers ---
    fwd_hooks = []
    for layer_i in range(NUM_LAYERS):
        hook_point_name = tl_utils.get_act_name("post", layer_i)
        fwd_hooks.append((hook_point_name, stats_hook))
    print(f"Registered {len(fwd_hooks)} hooks for MLP layers 0-{NUM_LAYERS-1}.")

    # --- Process Prompts ---
    try:
        with open(log_path, 'w', encoding='utf-8') as logfile:
            logfile.write(f"# Neuron Statistics Capture Log ({run_id})\n")
            logfile.write(f"## Run: {run_path.name}\n")
            logfile.write(f"## Timestamp: {timestamp_str}\n\n")
            logfile.write("## Run Parameters\n")
            for key, value in run_metadata.items():
                value_str = str(value)
                if len(value_str) > 200: value_str = value_str[:197] + "..."
                logfile.write(f"- **{key.replace('_', ' ').title()}**: `{value_str}`\n")
            logfile.write("\n---\n\n## Prompt Processing\n\n")
            logfile.flush()

            with torch.no_grad(), tqdm(total=len(parsed_prompts), desc=f"Processing prompts", unit="prompt") as pbar:
                for i, prompt_info in enumerate(parsed_prompts):
                    prompt_text = prompt_info['prompt_text']
                    core_id = prompt_info['core_id']
                    prompt_type = prompt_info['type']
                    level = prompt_info['level']
                    prompt_key_str = f"core_id={core_id}_type={prompt_type}_level={level}"

                    pbar.set_description(f"{core_id[:15]} L{level} {prompt_type[:4]}...")
                    logfile.write(f"### {i+1}/{len(parsed_prompts)}: {prompt_key_str}\n")
                    logfile.write(f"- **Prompt Text:** `{prompt_text}`\n")

                    try:
                        # --- Tokenize ---
                        input_ids = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=model.cfg.n_ctx)["input_ids"].to(device)
                        current_token_count = input_ids.shape[1]

                        if current_token_count == 0:
                            logfile.write("- **Result:** Error - Empty prompt after tokenization. Skipping.\n\n")
                            logfile.flush(); pbar.update(1); skipped_prompt_count += 1; continue

                        # --- Run Forward Pass (Triggers Hooks) ---
                        with model.hooks(fwd_hooks=fwd_hooks):
                            _ = model(input_ids, return_type=None) # Result ignored, hooks do the work

                        total_token_count += current_token_count
                        processed_prompts_count += 1
                        logfile.write(f"- **Result:** Processed successfully ({current_token_count} tokens).\n\n")

                    except Exception as e:
                        logfile.write(f"- **Result:** ERROR processing this prompt: {str(e)}\n\n")
                        print(f"\n--- ERROR processing {prompt_key_str} ---"); traceback.print_exc(); print(f"--- END ERROR ---")
                        skipped_prompt_count += 1
                    finally:
                        if device == 'cuda': torch.cuda.empty_cache()
                        logfile.flush()
                        pbar.update(1)

            logfile.write("---\nPrompt processing complete.\n")

    except Exception as e:
        print(f"\n--- FATAL ERROR during main processing loop ---"); traceback.print_exc()
        if 'logfile' in locals() and logfile and not logfile.closed:
            try: logfile.write(f"\n\nFATAL ERROR occurred: {e}\n"); traceback.print_exc(file=logfile)
            except Exception: pass

    # --- Post-Processing: Calculate Final Statistics ---
    print("\nCalculating final statistics from accumulated data...")
    neuron_stats_list = []
    neurons_analyzed = 0
    if not stats_accumulator:
        print("Error: No statistics were accumulated. Check hook function and forward pass.")
    else:
        for (layer, neuron_idx), stats in tqdm(stats_accumulator.items(), desc="Calculating stats", unit="neuron"):
            count = stats['count']
            if count > 0:
                mean = stats['sum'] / count
                # Calculate variance: (sum_sq / count) - mean^2
                variance = (stats['sum_sq'] / count) - (mean ** 2)
                # Handle potential floating point inaccuracies leading to small negative variance
                if variance < 0 and variance > -1e-9: # Allow tiny negative values
                    variance = 0.0
                elif variance < 0:
                     print(f"Warning: Negative variance ({variance:.2e}) for L{layer}N{neuron_idx}. Clamping to 0. (sum={stats['sum']:.2e}, sum_sq={stats['sum_sq']:.2e}, count={count}, mean={mean:.2e})")
                     variance = 0.0

                std_dev = math.sqrt(variance)
                neuron_stats_list.append({
                    'layer': layer,
                    'neuron': neuron_idx,
                    'mean': mean,
                    'std_dev': std_dev,
                    'tokens_observed': count
                })
                neurons_analyzed += 1
            else:
                # Should not happen if hooks fired, but handle just in case
                 neuron_stats_list.append({
                    'layer': layer,
                    'neuron': neuron_idx,
                    'mean': 0.0, # Or NaN? Let's use 0 for simplicity
                    'std_dev': 0.0,
                    'tokens_observed': 0
                })
                 neurons_analyzed += 1 # Still count it as analyzed

    print(f"Calculated statistics for {neurons_analyzed} neurons.")

    # --- Save Statistics to CSV ---
    if neuron_stats_list:
        stats_df = pd.DataFrame(neuron_stats_list)
        stats_df = stats_df.sort_values(by=['layer', 'neuron']).reset_index(drop=True) # Sort for consistency
        try:
            stats_df.to_csv(stats_path, index=False, float_format='%.6g') # Use efficient float format
            print(f"Neuron statistics saved to: {stats_path}")

            # --- Report Top N Lonely/Active ---
            if args.top_n_report > 0 and not stats_df.empty:
                print(f"\n--- Top {args.top_n_report} Loneliest Neurons (Lowest Mean Activation) ---")
                loneliest = stats_df.nsmallest(args.top_n_report, 'mean')
                print(loneliest.to_string(index=False))
                if 'logfile' in locals() and logfile and not logfile.closed: # Check logfile is open
                    logfile.write(f"\n\n--- Top {args.top_n_report} Loneliest Neurons (Lowest Mean Activation) ---\n")
                    logfile.write(loneliest.to_string(index=False) + "\n")


                print(f"\n--- Top {args.top_n_report} Most Active Neurons (Highest Mean Activation) ---")
                most_active = stats_df.nlargest(args.top_n_report, 'mean')
                print(most_active.to_string(index=False))
                # Ensure correct indentation and check logfile before writing
                if 'logfile' in locals() and logfile and not logfile.closed:
                    logfile.write(f"\n\n--- Top {args.top_n_report} Most Active Neurons (Highest Mean Activation) ---\n")
                    logfile.write(most_active.to_string(index=False)+ "\n")

                print(f"\n--- Top {args.top_n_report} Least Variant Neurons (Lowest Std Dev) ---")
                least_variant = stats_df.nsmallest(args.top_n_report, 'std_dev')
                print(least_variant.to_string(index=False))
                # Ensure correct indentation and check logfile before writing
                if 'logfile' in locals() and logfile and not logfile.closed:
                    logfile.write(f"\n\n--- Top {args.top_n_report} Least Variant Neurons (Lowest Std Dev) ---\n")
                    logfile.write(least_variant.to_string(index=False)+ "\n")

        except Exception as e:
            print(f"\n--- ERROR saving neuron statistics CSV ---"); traceback.print_exc()
            if 'logfile' in locals() and logfile and not logfile.closed: logfile.write(f"\nERROR saving CSV: {e}\n")

    else:
        print("\nNo neuron statistics were generated to save.")
        if 'logfile' in locals() and logfile and not logfile.closed: logfile.write("\nNo neuron statistics generated.\n")


    # --- Final Metadata Update ---
    run_metadata["total_tokens_processed"] = total_token_count
    run_metadata["neurons_analyzed"] = neurons_analyzed
    run_metadata["prompts_processed_successfully"] = processed_prompts_count
    run_metadata["prompts_skipped_or_failed"] = skipped_prompt_count

    if utils.save_json_metadata(metadata_path, run_metadata):
         print(f"\nFinal metadata updated and saved to: {metadata_path}")
    else:
         print(f"Warning: Failed to save final metadata update to {metadata_path}")


    print(f"\nScript finished. Results are in top-level directory: {run_path}")
    print(f"Capture outputs (stats CSV, logs, metadata) are within: {capture_path}")

# --- END OF FILE find_lonely_neurons.py ---