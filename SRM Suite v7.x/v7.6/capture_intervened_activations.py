# --- START OF FILE capture_intervened_activations.py ---

import torch
import argparse
import warnings
import sys
import traceback
import datetime # Keep for fallback timestamp if needed
import re
from pathlib import Path
import numpy as np
from tqdm import tqdm

# Local application imports
# Ensure transformer_lens is installed: pip install transformer_lens
try:
    from transformer_lens import HookedTransformer, utils as tl_utils # Use alias
except ImportError:
    print("Error: transformer_lens not found. Please install it: pip install transformer_lens")
    sys.exit(1)

# Ensure utils.py is accessible
try:
    import utils # Import our updated utility functions
except ImportError:
    print("Error: utils.py not found in the current directory or Python path.")
    sys.exit(1)

# --- Constants ---
DEFAULT_ID_PREFIX = "INT" # Default prefix for intervention runs
# DIMENSION defined in utils or determined dynamically

# --- Helper Functions ---

def parse_sweep_values(value_str: str | None) -> list[float | int | None] | None: # Allow None input
    """ Parses a comma-separated string of numbers and 'None' into a list. """
    if not value_str:
        return None
    values = []
    for item in value_str.split(','):
        item = item.strip()
        if item.lower() == 'none':
            values.append(None)
        else:
            try:
                # Try parsing as float first
                val_f = float(item)
                # If it's an integer, store as int, otherwise float
                if val_f.is_integer():
                    values.append(int(val_f))
                else:
                    values.append(val_f)
            except ValueError:
                 # If float parsing fails, it's not a valid number or 'None'
                 print(f"Warning: Could not parse sweep value '{item}' as number or 'None'. Skipping.")
    if not values: # Check if list is empty after parsing
        return None
    return values

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
    parser = argparse.ArgumentParser(description="Capture MLP activations and generated text for structured prompts with neuron interventions. Saves results in a structured experiment run directory with unique RunID.") # Modified description
    parser.add_argument("--prompt_file", type=str, required=True, help="Path to the structured prompt file (e.g., promptsets/epistemic_certainty_grid.txt).")
    parser.add_argument("--experiment_base_dir", type=str, required=True, help="Path to the base directory where the unique run directory will be created (e.g., ./experiments).")
    parser.add_argument("--id_prefix", type=str, default=DEFAULT_ID_PREFIX, help=f"Prefix for the Run ID (e.g., 'INT', 'EXP'). Default: {DEFAULT_ID_PREFIX}.")
    parser.add_argument("--generate_length", type=int, default=50, help="Number of new tokens to generate.")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k sampling parameter. If None or 0, use greedy decoding.")
    parser.add_argument("--layer", type=int, required=True, help="MLP layer index for intervention and capture.")
    parser.add_argument("--target_neuron", type=int, required=True, help="Index of the neuron to intervene on in the MLP layer.")
    parser.add_argument("--sweep_values", type=str, required=True,
                        help="Comma-separated list of values to clamp the target neuron to (e.g., 'None,10,-10'). 'None' performs a baseline run within the sweep.")
    args = parser.parse_args()

    intervention_values = parse_sweep_values(args.sweep_values)
    if not intervention_values:
        print("Error: No valid sweep values provided or parsed from --sweep_values argument. Exiting.")
        sys.exit(1)
    print(f"Intervention sweep values: {intervention_values}")

    # --- Generate Run ID and Timestamp ---
    try:
        run_id = utils.generate_run_id(args.id_prefix)
        timestamp_str = utils.get_formatted_timestamp()
        print(f"Generated Run ID: {run_id}")
        print(f"Generated Timestamp: {timestamp_str}")
    except Exception as e:
        print(f"Fatal Error: Could not generate Run ID or Timestamp: {e}")
        sys.exit(1)
    # --- End ID/Timestamp Generation ---

    # --- Setup Paths using pathlib and utils constants ---
    base_dir = Path(args.experiment_base_dir)
    prompt_path = Path(args.prompt_file)
    if not prompt_path.is_file():
         print(f"Error: Prompt file not found at {prompt_path}"); sys.exit(1)
    prompt_basename_sanitized = utils.sanitize_label(prompt_path.stem)

    # Construct directory name using RunID, descriptive parts, and Timestamp
    run_dir_name = f"{run_id}_intervened_L{args.layer}N{args.target_neuron}_{prompt_basename_sanitized}_{timestamp_str}"
    run_path = base_dir / run_dir_name

    # Subdirectories including the new text_outputs
    capture_path = run_path / utils.CAPTURE_SUBFOLDER
    vector_dir = capture_path / utils.VECTORS_SUBFOLDER
    log_dir = capture_path / utils.LOGS_SUBFOLDER
    metadata_dir = capture_path / utils.METADATA_SUBFOLDER
    text_output_dir = capture_path / "text_outputs" # <<< NEW PATH DEFINITION

    try:
        metadata_dir.mkdir(parents=True, exist_ok=True)
        vector_dir.mkdir(exist_ok=True)
        log_dir.mkdir(exist_ok=True)
        text_output_dir.mkdir(exist_ok=True) # <<< CREATE NEW DIRECTORY
        print(f"Created unique run directory structure: {run_path}")
        print(f" -> Capture outputs will be saved in: {capture_path}")
        print(f"   -> Text outputs will be saved in: {text_output_dir}") # Info message
    except OSError as e:
        print(f"Error creating output directories in '{run_path}': {e}", file=sys.stderr)
        traceback.print_exc(); sys.exit(1)

    # Define internal filenames (simplified)
    vector_filename = f"intervened_vectors.npz"
    log_filename = f"run_log_intervened.md"
    metadata_filename = f"run_metadata_intervened.json"
    # Full paths
    vector_path = vector_dir / vector_filename
    log_path = log_dir / log_filename
    metadata_path = metadata_dir / metadata_filename
    # --- End Path Setup ---

    # --- Parse Prompts ---
    parsed_prompts = parse_structured_prompts(prompt_path)
    if not parsed_prompts: print("Exiting due to prompt parsing failure."); sys.exit(1)

    # --- Load Model ---
    print("\nLoading GPT-2 Small model...")
    try:
        model = HookedTransformer.from_pretrained("gpt2")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device); model.eval()
        print(f"Using device: {device}")
        tokenizer = model.tokenizer
        if tokenizer.pad_token is None:
            print("Tokenizer does not have a pad token; setting to eos_token.")
            tokenizer.pad_token = tokenizer.eos_token
            if hasattr(model.config, 'pad_token_id'):
                model.config.pad_token_id = tokenizer.eos_token_id

        TARGET_HOOK_POINT = tl_utils.get_act_name("post", args.layer)
        capture_hook_point_name = TARGET_HOOK_POINT # Capture same layer as intervention
        # Determine DIMENSION dynamically
        try:
            DIMENSION = model.cfg.d_mlp
            if DIMENSION is None: raise AttributeError
        except AttributeError:
            print("Error: Could not determine MLP dimension (d_mlp) from model config.")
            sys.exit(1)

        if not (0 <= args.target_neuron < DIMENSION):
            print(f"Error: Target neuron index {args.target_neuron} is out of bounds for MLP dimension {DIMENSION}.")
            sys.exit(1)
        print(f"Targeting layer {args.layer}, neuron {args.target_neuron} (hook: '{TARGET_HOOK_POINT}', dim: {DIMENSION})")
    except Exception as e: print(f"Error loading model: {e}", file=sys.stderr); traceback.print_exc(); sys.exit(1)

    # --- Prepare Metadata ---
    run_metadata = {
        "run_id": run_id,
        "script_name": Path(__file__).name,
        "run_type": "intervention_capture",
        "model_name": "gpt2",
        "target_layer": args.layer,
        "target_neuron": args.target_neuron,
        "intervention_hook": TARGET_HOOK_POINT,
        "capture_hook": capture_hook_point_name,
        "activation_dimension": DIMENSION,
        "sweep_values_arg_string": args.sweep_values,
        "sweep_values_parsed": intervention_values,
        "prompt_file_path": str(prompt_path),
        "prompt_file_relative": str(prompt_path.relative_to(base_dir)) if prompt_path.is_relative_to(base_dir) else str(prompt_path),
        "prompt_file_basename": prompt_path.stem,
        "prompt_file_basename_sanitized": prompt_basename_sanitized,
        "num_prompts_parsed": len(parsed_prompts),
        "generate_length": args.generate_length,
        "top_k_setting": args.top_k if args.top_k is not None and args.top_k > 0 else "greedy",
        "run_timestamp_str": timestamp_str,
        "run_directory": str(run_path),
        "run_directory_name": run_path.name,
        "capture_directory_relative": str(capture_path.relative_to(run_path)),
        "output_vector_file_relative": str(vector_path.relative_to(capture_path)),
        "output_log_file_relative": str(log_path.relative_to(capture_path)),
        "output_metadata_file_relative": str(metadata_path.relative_to(capture_path)),
        "output_text_files_relative": str(text_output_dir.relative_to(capture_path)), # <<< NEW METADATA FIELD
        "device": device,
        "cli_args": vars(args),
        "final_vector_count": None,
        "prompts_completed_all_sweeps": None,
        "prompt_sweep_combinations_processed_successfully": None, # Vector captured successfully
        "prompt_sweep_combinations_skipped_or_failed": None, # Vector capture failed / text gen failed
        "text_outputs_saved": 0, # <<< NEW COUNTER
        "text_outputs_failed": 0, # <<< NEW COUNTER
    }
    if utils.save_json_metadata(metadata_path, run_metadata):
         print(f"Initial metadata saved to: {metadata_path}")
    else:
         print(f"Warning: Could not save initial metadata file '{metadata_path}'. Continuing run.")

    # --- Main Processing ---
    print(f"\nStarting intervention runs ({run_id}) for {len(parsed_prompts)} prompts across {len(intervention_values)} sweep values.")
    all_vectors = {} # Stores {vector_key_with_sweep: numpy_array}
    total_runs = len(parsed_prompts) * len(intervention_values)
    processed_prompts_count = 0 # Count prompts where *all* sweeps succeeded (for vector capture)
    processed_sweeps_count = 0 # Count successful prompt/sweep combinations (for vector capture)
    skipped_capture_count = 0 # Count failed/skipped prompt/sweep combinations (for vector capture)
    text_saved_count = 0 # <<< Initialize local counter
    text_failed_count = 0 # <<< Initialize local counter

    try:
        with open(log_path, 'w', encoding='utf-8') as logfile:
            logfile.write(f"# Intervention Activation Capture Log ({run_id})\n")
            logfile.write(f"## Run: {run_path.name}\n")
            logfile.write(f"## Timestamp: {timestamp_str}\n\n")
            logfile.write("## Run Parameters\n")
            for key, value in run_metadata.items():
                 value_str = str(value)
                 if len(value_str) > 200: value_str = value_str[:197] + "..."
                 logfile.write(f"- **{key.replace('_', ' ').title()}**: `{value_str}`\n")
            logfile.write("\n---\n\n## Prompt Processing\n\n"); logfile.flush()

            # Setup hook function for saving activations
            capture_container = {'activation': None}
            def save_hook(activation_tensor, hook):
                capture_container['activation'] = activation_tensor.detach().cpu()

            # Outer loop: Prompts
            with torch.no_grad(), tqdm(total=total_runs, desc="Prompt Interventions", unit="run") as pbar:
                for i, prompt_info in enumerate(parsed_prompts):
                    prompt_text = prompt_info['prompt_text']; core_id = prompt_info['core_id']
                    prompt_type = prompt_info['type']; level = prompt_info['level']
                    prompt_base_key = f"core_id={core_id}_type={prompt_type}_level={level}"
                    logfile.write(f"### Prompt {i+1}/{len(parsed_prompts)}: {prompt_base_key}\n")

                    prompt_sweep_success_flag = True # Track if all sweeps for this prompt succeed (vector capture)

                    # Inner loop: Intervention sweep values
                    for sweep_idx, sweep_value in enumerate(intervention_values):

                        # --- Define Intervention Hook specific to this sweep_value ---
                        def intervention_hook(activation, hook):
                            sweep_tensor_val = None
                            if sweep_value is not None:
                                try:
                                    sweep_tensor_val = torch.tensor(float(sweep_value), device=activation.device, dtype=activation.dtype)
                                except ValueError:
                                     # Logged below if needed, return unmodified here
                                     pass # Let the main code handle logging this type of error if needed

                            if sweep_tensor_val is not None:
                                try:
                                    if activation.ndim == 3:
                                        activation[:, :, args.target_neuron] = sweep_tensor_val
                                    elif activation.ndim == 2:
                                        activation[:, args.target_neuron] = sweep_tensor_val
                                    else:
                                        # Logged below if needed
                                        pass
                                except IndexError:
                                     print(f"ERROR: Neuron index {args.target_neuron} out of bounds for activation shape {activation.shape} in hook.")
                                     # Don't modify activation if index is bad
                                except Exception as hook_e:
                                     print(f"ERROR: Unexpected error during intervention assignment in hook: {hook_e}")
                                     # Don't modify activation
                            return activation # Always return activation
                        # --- End Intervention Hook Definition ---

                        sweep_tag_val = sweep_value if sweep_value is not None else 'baseline'
                        sweep_tag = f"sweep={sweep_tag_val}"
                        vector_key = f"{prompt_base_key}_{sweep_tag}"

                        pbar.set_description(f"{core_id[:10]}.. L{level} Swp:{sweep_tag_val}")
                        logfile.write(f"\n#### Intervention: {sweep_tag}\n");
                        logfile.write(f"- **Target Key:** `{vector_key}`\n")

                        try:
                            # --- Tokenize ---
                            input_ids = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=model.cfg.n_ctx)["input_ids"].to(device)
                            input_len = input_ids.shape[1]
                            if input_len == 0:
                                logfile.write("- **Result:** Error - Empty prompt after tokenization.\n\n")
                                logfile.flush(); pbar.update(1); skipped_capture_count += 1; prompt_sweep_success_flag = False; continue
                            if input_len >= model.cfg.n_ctx:
                                logfile.write(f"- **Result:** Warning - Prompt length ({input_len}) equals/exceeds model context ({model.cfg.n_ctx}). May affect intervention/generation.\n")

                            # --- Generate Text (WITH intervention hook ONLY) ---
                            gen_hooks = [(TARGET_HOOK_POINT, intervention_hook)]
                            with model.hooks(fwd_hooks=gen_hooks):
                                output_ids = model.generate(
                                     input_ids,
                                     max_new_tokens=args.generate_length,
                                     do_sample=(args.top_k is not None and args.top_k > 0),
                                     top_k=args.top_k if (args.top_k is not None and args.top_k > 0) else None,
                                     eos_token_id=tokenizer.eos_token_id
                                     )
                            # --- End Generate Text ---

                            # --- Process Generation Result ---
                            generated_len = output_ids.shape[1] - input_len

                            if generated_len <= 0:
                                 logfile.write("- **Generated Text:** (No new tokens)\n")
                                 logfile.write("- **Result:** Skipped capture (no new tokens).\n\n")
                                 logfile.flush(); pbar.update(1); skipped_capture_count += 1; prompt_sweep_success_flag = False; continue
                            else:
                                 result_text = tokenizer.decode(output_ids[0, input_len:], skip_special_tokens=True)
                                 logfile.write(f"- **Generated Text:**\n```\n{result_text}\n```\n")

                                 # <<< SAVE GENERATED TEXT TO FILE >>>
                                 text_output_filename = f"core_id={core_id}_type={prompt_type}_level={level}_sweep={sweep_tag_val}.txt"
                                 text_output_path = text_output_dir / text_output_filename
                                 try:
                                     with open(text_output_path, 'w', encoding='utf-8') as text_file:
                                         text_file.write(result_text)
                                     logfile.write(f"- **Output Text Saved:** `{text_output_filename}`\n")
                                     text_saved_count += 1 # <<< Increment counter
                                 except Exception as e_text:
                                     logfile.write(f"- **Result:** Error saving output text file `{text_output_filename}`: {e_text}\n")
                                     text_failed_count += 1 # <<< Increment counter
                                 # <<< END TEXT SAVING CODE >>>

                            # --- Capture Activations (Explicit Forward pass with BOTH hooks) ---
                            capture_container['activation'] = None
                            fwd_pass_hooks = [(TARGET_HOOK_POINT, intervention_hook), (capture_hook_point_name, save_hook)]
                            try:
                                with model.hooks(fwd_hooks=fwd_pass_hooks):
                                    _ = model(output_ids, return_type=None)
                            except Exception as e_inner:
                                logfile.write(f"- **Result:** Error during capture forward pass: {e_inner}\n\n")
                                traceback.print_exc(file=sys.stderr); logfile.flush(); pbar.update(1); skipped_capture_count += 1; prompt_sweep_success_flag = False; continue

                            captured_mlp_post_activation = capture_container['activation']

                            # --- Process and Store Vector ---
                            if captured_mlp_post_activation is None:
                                logfile.write("- **Result:** Error - Failed to capture activation (explicit pass failed or hook issue).\n\n")
                                skipped_capture_count += 1; logfile.flush(); pbar.update(1); prompt_sweep_success_flag = False; continue

                            expected_seq_len = output_ids.shape[1]
                            if not (captured_mlp_post_activation.ndim == 3 and
                                    captured_mlp_post_activation.shape[0] == 1 and
                                    captured_mlp_post_activation.shape[2] == DIMENSION):
                                 logfile.write(f"- **Result:** Error - Captured activation dimensions invalid. Expected (1, seq_len, {DIMENSION}), got {captured_mlp_post_activation.shape}. Skipping vector extraction.\n\n")
                                 skipped_capture_count += 1; logfile.flush(); pbar.update(1); prompt_sweep_success_flag = False; continue

                            actual_activation_seq_len = captured_mlp_post_activation.shape[1]
                            if actual_activation_seq_len != expected_seq_len:
                                logfile.write(f"- **Result:** Warning - Captured activation sequence length mismatch. Expected {expected_seq_len}, got {actual_activation_seq_len}. Will slice based on actual length.\n")

                            slice_start_idx = min(input_len, actual_activation_seq_len)
                            slice_end_idx = actual_activation_seq_len

                            if slice_start_idx >= slice_end_idx:
                                 logfile.write(f"- **Result:** Warning - Cannot slice generated tokens (start_idx {slice_start_idx} >= end_idx {slice_end_idx}). Captured shape: {captured_mlp_post_activation.shape}, input_len: {input_len}.\n\n")
                                 skipped_capture_count += 1; logfile.flush(); pbar.update(1); prompt_sweep_success_flag = False; continue

                            generated_vectors_tensor = captured_mlp_post_activation[0, slice_start_idx:slice_end_idx, :]

                            if generated_vectors_tensor.shape[0] > 0:
                                mean_vector_np = np.mean(generated_vectors_tensor.numpy(), axis=0)
                                if mean_vector_np.shape == (DIMENSION,):
                                    all_vectors[vector_key] = mean_vector_np.astype(np.float32)
                                    logfile.write(f"- **Result:** Vector captured successfully (shape: {mean_vector_np.shape}, num tokens averaged: {generated_vectors_tensor.shape[0]}).\n\n")
                                    processed_sweeps_count += 1
                                else:
                                    logfile.write(f"- **Result:** Error - Mean vector shape mismatch ({mean_vector_np.shape}). Expected ({DIMENSION},).\n\n")
                                    skipped_capture_count += 1; prompt_sweep_success_flag = False
                            else:
                                logfile.write(f"- **Result:** Warning - Sliced activation tensor for generated tokens had 0 length after explicit pass (captured shape: {captured_mlp_post_activation.shape}, input_len: {input_len}, start: {slice_start_idx}, end: {slice_end_idx}).\n\n")
                                skipped_capture_count += 1; prompt_sweep_success_flag = False

                        # --- Error handling for this specific prompt/sweep combination ---
                        except Exception as e:
                            logfile.write(f"- **Result:** ERROR processing this prompt/sweep combination: {str(e)}\n\n")
                            print(f"\n--- ERROR processing {vector_key} ---"); traceback.print_exc(); print(f"--- END ERROR ---")
                            skipped_capture_count += 1
                            prompt_sweep_success_flag = False
                        finally:
                             if device == 'cuda': torch.cuda.empty_cache()
                             logfile.flush(); pbar.update(1)

                    # --- End of sweep value loop ---
                    if prompt_sweep_success_flag:
                        processed_prompts_count += 1
                    logfile.write("\n---\n"); logfile.flush()
                # --- End of prompt loop ---

            logfile.write("\nRun Complete.\n")

    except Exception as e:
        print(f"\n--- FATAL ERROR during main processing loop ---"); traceback.print_exc()
        if 'logfile' in locals() and logfile and not logfile.closed:
             try: logfile.write(f"\n\nFATAL ERROR occurred: {e}\n"); traceback.print_exc(file=logfile)
             except Exception: pass

    # --- Final Saving ---
    final_vector_count = len(all_vectors)
    # Update metadata with final counts
    run_metadata["final_vector_count"] = final_vector_count
    run_metadata["prompts_completed_all_sweeps"] = processed_prompts_count
    run_metadata["prompt_sweep_combinations_processed_successfully"] = processed_sweeps_count
    run_metadata["prompt_sweep_combinations_skipped_or_failed"] = skipped_capture_count
    run_metadata["text_outputs_saved"] = text_saved_count # <<< UPDATE METADATA
    run_metadata["text_outputs_failed"] = text_failed_count # <<< UPDATE METADATA

    if utils.save_json_metadata(metadata_path, run_metadata):
         print(f"\nFinal metadata updated and saved to: {metadata_path}")
    else:
         print(f"Warning: Failed to save final metadata update to {metadata_path}")

    if all_vectors:
        print(f"Saving {final_vector_count} collected mean vectors to {vector_path}...")
        try:
            np.savez_compressed(vector_path, **all_vectors, __metadata__=np.array(run_metadata, dtype=object))
            print("Vectors and embedded metadata saved successfully.")
        except Exception as e:
            print(f"\n--- ERROR saving final vectors ---"); traceback.print_exc()
            try:
                print("Attempting to save vectors without embedded metadata...")
                np.savez_compressed(vector_path, **all_vectors)
                print("Fallback save successful (vectors only). Metadata embedding failed earlier.")
            except Exception as e_fallback:
                 print(f"Fallback vector save failed: {e_fallback}")
    else:
        print(f"\nNo vectors were successfully collected to save to {vector_path}.")

    print(f"\nText outputs saved: {text_saved_count}, Failed: {text_failed_count}") # Summary message
    print(f"Script finished. Results are in top-level directory: {run_path}")
    print(f"Capture outputs (vectors, logs, metadata, text_outputs) are within: {capture_path}")


# --- END OF FILE capture_intervened_activations.py ---