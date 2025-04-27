# --- START OF FILE generate_basis_vectors.py ---

import argparse
import datetime
import traceback
from pathlib import Path
import numpy as np
from collections import defaultdict
import re
import sys

# Local application imports
import utils # Import our updated utility functions

# --- Constants ---
DEFAULT_EXPERIMENT_BASE_DIR = "experiments"
DEFAULT_BASIS_KEY_ENSEMBLE = 'basis_vectors'
MIN_VECTORS_DEFAULT = 1
DEFAULT_ONEHOT_OUTPUT_DIR = Path("./generated_basis") # Default relative path
BASELINE_RUN_TYPE = "baseline_capture"

# --- Helper Functions ---
def _concise_filter_repr(filter_dict: dict | None) -> str:
    if not filter_dict: return "all"
    parts = []
    key_map = {'type': 't', 'level': 'l', 'sweep': 's', 'core_id': 'cid'}
    sorted_keys = sorted(filter_dict.keys())
    for key in sorted_keys:
        prefix = key_map.get(key, str(key)[:3])
        value = str(filter_dict[key])
        # Simple shortening examples
        value_short = value.replace('declarative', 'decl').replace('rhetorical', 'rhet')
        value_short = value_short.replace('observational', 'obs').replace('authoritative', 'auth')
        value_short = value_short.replace('baseline','base')
        value_short = value_short.replace('descriptive', 'desc') # Added
        value_short = value_short[:8] # Limit length
        parts.append(f"{prefix}_{value_short}")
    return "_".join(parts) if parts else "all"

def generate_default_basis_label(args: argparse.Namespace) -> str:
    if args.mode == 'single_plane':
        f1 = utils.parse_filter_string(args.filter_set_1) if args.filter_set_1 else None
        f2 = utils.parse_filter_string(args.filter_set_2) if args.filter_set_2 else None
        f1_repr = _concise_filter_repr(f1) if f1 else "filter1-error"
        f2_repr = _concise_filter_repr(f2) if f2 else "filter2-error"
        return f"single_plane_{f1_repr}_vs_{f2_repr}"
    elif args.mode == 'ensemble':
        group_key = args.ensemble_group_key or "nogroup"
        fixed_f = utils.parse_filter_string(args.fixed_filters) if args.fixed_filters else None
        fixed_repr = _concise_filter_repr(fixed_f) if fixed_f else "all"
        output_key = args.output_key or DEFAULT_BASIS_KEY_ENSEMBLE
        output_key_sanitized = utils.sanitize_label(output_key)
        label_parts = ["ensemble", f"grp_{group_key}"]
        if fixed_repr != "all": label_parts.append(f"fixed_{fixed_repr}")
        if output_key_sanitized != DEFAULT_BASIS_KEY_ENSEMBLE:
             label_parts.append(f"key_{output_key_sanitized}")
        return "_".join(label_parts)
    elif args.mode == 'onehot': # Added onehot default label generation
        n1 = args.neuron_1
        n2 = args.neuron_2
        # Note: Layer isn't available in args here, so label doesn't include it by default
        return f"onehot_N{n1}_vs_N{n2}"
    else:
        return f"unknown_mode_{datetime.datetime.now().strftime('%Y%m%d')}"

# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"Generate basis vectors (.npz) and metadata (.json). "
                    f"Modes 'single_plane'/'ensemble' derive basis from selected run's '{utils.CAPTURE_SUBFOLDER}/' data. "
                    f"Mode 'onehot' generates vectors programmatically.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    # --- General Arguments ---
    parser.add_argument("--mode", type=str, choices=['single_plane', 'ensemble', 'onehot'], required=True,
                        help="Operation mode:\n"
                             "  'single_plane': Generate basis_1, basis_2 from filtered data in a selected run.\n"
                             "  'ensemble': Generate multiple basis vectors grouped by a key from filtered data in a selected run.\n"
                             "  'onehot': Generate basis from two specific neuron indices (requires --neuron_1, --neuron_2).")
    parser.add_argument("--output_basis_label", type=str, default=None,
                        help="[Optional] Custom descriptive label for the output basis file (e.g., 'declarative1_vs_rhetorical5', 'lonely_L0N631_vs_L0N0').\nIf None, a default is auto-generated based on mode/filters/neurons.")
    parser.add_argument("--min_vectors_per_group", type=int, default=MIN_VECTORS_DEFAULT,
                        help=f"[Data Modes Only] Minimum number of vectors required to calculate a mean for a group/set (default: {MIN_VECTORS_DEFAULT}).")
    parser.add_argument("--force_non_baseline", action='store_true',
                        help="[Data Modes Only] Allow basis generation even if source data run_type isn't 'baseline_capture'. Issues a warning.")
    # --- Data-Derived Mode Arguments (single_plane, ensemble) ---
    parser.add_argument("--experiment_base_dir", type=str, default=DEFAULT_EXPERIMENT_BASE_DIR,
                        help="[Data Modes Only] Top-level directory containing experiment run folders (e.g., ./experiments).")
    parser.add_argument("--filter_set_1", type=str, default=None,
                        help="[Single Plane] Filters for basis_1 (e.g., 'type=declarative,level=1'). Required if mode=single_plane.")
    parser.add_argument("--filter_set_2", type=str, default=None,
                        help="[Single Plane] Filters for basis_2 (e.g., 'type=rhetorical,level=5'). Required if mode=single_plane.")
    parser.add_argument("--ensemble_group_key", type=str, choices=utils.VALID_GROUP_KEYS, default=None,
                        help="[Ensemble] Metadata key to group by for basis generation (e.g., 'type'). Required if mode=ensemble.")
    parser.add_argument("--fixed_filters", type=str, default=None,
                        help="[Ensemble] Additional fixed filters applied to ALL groups before grouping (e.g., 'level=5').")
    parser.add_argument("--output_key", type=str, default=DEFAULT_BASIS_KEY_ENSEMBLE,
                        help=f"[Ensemble] Key name for the basis array in the output NPZ (default: {DEFAULT_BASIS_KEY_ENSEMBLE}).")
    # --- Procedural Mode Arguments (onehot) ---
    parser.add_argument("--neuron_1", type=int, default=None,
                        help="[OneHot Mode] Index of the first neuron (0-based). Required if mode='onehot'.")
    parser.add_argument("--neuron_2", type=int, default=None,
                        help="[OneHot Mode] Index of the second neuron (0-based). Required if mode='onehot'.")
    parser.add_argument("--onehot_output_dir", type=str, default=str(DEFAULT_ONEHOT_OUTPUT_DIR),
                        help="[OneHot Mode] Directory to save one-hot basis files.")

    args = parser.parse_args()

    # --- Input Validation ---
    if args.mode == 'single_plane' and (not args.filter_set_1 or not args.filter_set_2): parser.error("--filter_set_1 and --filter_set_2 are required for mode 'single_plane'.")
    if args.mode == 'ensemble' and not args.ensemble_group_key: parser.error("--ensemble_group_key is required for mode 'ensemble'.")
    if args.mode in ['single_plane', 'ensemble'] and args.min_vectors_per_group < 1: print(f"Warning: --min_vectors_per_group must be >= 1. Setting to {MIN_VECTORS_DEFAULT}."); args.min_vectors_per_group = MIN_VECTORS_DEFAULT
    if args.mode == 'onehot' and (args.neuron_1 is None or args.neuron_2 is None): parser.error("--neuron_1 and --neuron_2 are required for mode 'onehot'.")
    if args.mode == 'onehot':
        if not (0 <= args.neuron_1 < utils.DIMENSION and 0 <= args.neuron_2 < utils.DIMENSION): parser.error(f"Neuron indices must be between 0 and {utils.DIMENSION - 1}.")
        if args.neuron_1 == args.neuron_2: parser.error("Neuron indices (--neuron_1, --neuron_2) must be different.")

    # --- Define shared variables ---
    basis_dir = None
    output_npz_path = None
    output_json_path = None
    basis_metadata = {}
    basis_saved = False
    generation_timestamp = datetime.datetime.now()
    source_run_id = None
    base_experiments_dir = None # Initialize

    # Resolve experiment base dir path carefully for data modes
    if args.mode != 'onehot':
        try:
            base_experiments_dir = Path(args.experiment_base_dir).resolve()
        except Exception as e:
            print(f"Error resolving experiment base directory '{args.experiment_base_dir}': {e}")
            # Decide if this is fatal for data modes? Likely yes.
            sys.exit(1)

    # --- Mode-Specific Logic ---
    try:
        if args.mode == 'onehot':
            # --- OneHot Mode Logic ---
            print("\n--- Generating One-Hot Basis ---")
            # Resolve onehot output dir (can be relative or absolute)
            basis_dir = Path(args.onehot_output_dir).resolve()
            try:
                basis_dir.mkdir(parents=True, exist_ok=True)
                print(f"Output directory set to: {basis_dir}")
            except OSError as e:
                print(f"Error: Could not create output directory '{basis_dir}': {e}")
                sys.exit(1)

            n1, n2 = args.neuron_1, args.neuron_2
            basis_1 = np.zeros(utils.DIMENSION, dtype=np.float32); basis_1[n1] = 1.0
            basis_2 = np.zeros(utils.DIMENSION, dtype=np.float32); basis_2[n2] = 1.0
            print(f"Generated one-hot basis vectors for Neuron {n1} and Neuron {n2}.")

            if args.output_basis_label:
                final_output_label = utils.sanitize_label(args.output_basis_label)
                print(f"Using provided (sanitized) label: '{final_output_label}'")
            else:
                # Generate default label including neuron numbers
                final_output_label = utils.sanitize_label(f"onehot_N{n1}_vs_N{n2}")
                print(f"Using auto-generated label: '{final_output_label}'")

            base_filename = f"basis_{final_output_label}"
            output_npz_path = basis_dir / f"{base_filename}.npz"
            output_json_path = basis_dir / f"{base_filename}.json"
            print(f"Output basis NPZ will be saved to: {output_npz_path}")
            print(f"Output metadata JSON will be saved to: {output_json_path}")

            basis_metadata = {
                "script_name": Path(__file__).name,
                "generation_timestamp": generation_timestamp.isoformat(),
                "basis_generation_mode": args.mode,
                "neuron_1": n1, "neuron_2": n2,
                "output_basis_directory": str(basis_dir), # Saved resolved path
                "output_basis_file": str(output_npz_path.name),
                "output_metadata_file": str(output_json_path.name),
                "user_provided_output_basis_label": args.output_basis_label,
                "generated_filename_label": final_output_label,
                "dimension": utils.DIMENSION,
                "cli_args": vars(args),
                # Fields not relevant to onehot mode
                "source_run_id": None, "source_run_directory": None, "source_run_directory_name": None,
                "source_vector_file_relative": None, "source_vector_metadata": None, "min_vectors_per_group": None,
                "single_plane_details": None, "ensemble_details": None,
                "force_non_baseline_used": None,
                "output_basis_directory_relative": None, # Not applicable here
                "output_basis_file_relative": None,
                "output_metadata_file_relative": None,
            }
            basis_saved = utils.save_basis_vectors(output_npz_path, basis_1=basis_1, basis_2=basis_2, metadata=basis_metadata)

            # --- Save Metadata JSON for onehot ---
            if basis_saved:
                 print(f"\nSaving basis generation metadata to: {output_json_path}")
                 if not utils.save_json_metadata(output_json_path, basis_metadata):
                     print("--- ERROR saving onehot basis metadata JSON file ---")
            else:
                 print("\nOnehot basis vectors were not saved. Skipping metadata JSON file generation.")

        # --- Data-Derived Modes ---
        elif args.mode in ['single_plane', 'ensemble']:
            print(f"\n--- Generating {args.mode.replace('_', ' ').title()} Basis (Data-Derived) ---")

            if base_experiments_dir is None: # Should have been resolved earlier
                print("Critical Error: Experiment base directory is not defined for data mode.")
                sys.exit(1)

            print(f"\nSearching for Experiment Run folders in '{base_experiments_dir}'...")
            source_run_dir = utils.select_experiment_folder(
                base_experiments_dir,
                prompt_text="Select the source run directory containing vectors:",
                pattern="*" # Allow selecting any run type
            )
            if not source_run_dir: print("No source run directory selected. Exiting."); sys.exit(1)
            print(f"Using source run directory: {source_run_dir.name}")

            # Find vector file within the selected source run
            input_vector_path = utils.find_vector_file(source_run_dir)
            if not input_vector_path:
                # Try finding simplified names if interactive selection failed
                found_simple = False
                for simple_name in ["baseline_vectors.npz", "intervened_vectors.npz"]:
                     potential_path = source_run_dir / utils.CAPTURE_SUBFOLDER / utils.VECTORS_SUBFOLDER / simple_name
                     if potential_path.is_file():
                         print(f"Info: Auto-selected vector file with simplified name: {potential_path.name}")
                         input_vector_path = potential_path
                         found_simple = True
                         break
                if not found_simple:
                     print(f"Error: Could not find required vector input file in {source_run_dir / utils.CAPTURE_SUBFOLDER / utils.VECTORS_SUBFOLDER}. Exiting.")
                     sys.exit(1)

            print(f"Using input vector file: {input_vector_path.relative_to(source_run_dir)}")

            # Load data and embedded metadata
            structured_data, source_metadata = utils.load_vector_data(input_vector_path, expected_dim=utils.DIMENSION)
            if structured_data is None: print("Exiting: Failed to load input vectors."); sys.exit(1)
            if not structured_data: print("Warning: No valid vectors were loaded from the input file. Cannot generate basis."); sys.exit(0)

            # Determine Source Run ID
            source_run_id = source_metadata.get('run_id')
            if not source_run_id:
                 match = re.match(r"([A-Z]+-\d{3})_", source_run_dir.name)
                 if match:
                     source_run_id = match.group(1)
                     print(f"Warning: 'run_id' missing in source metadata. Parsed from directory name: {source_run_id}")
                 else:
                     # If still no RunID, consider if this is acceptable. For basis generation, it's critical provenance.
                     print("Error: Could not determine 'run_id' from source metadata or directory name. Basis provenance unclear. Cannot proceed.")
                     sys.exit(1)
            else:
                 print(f"Confirmed source RunID from metadata: {source_run_id}")

            # Check Run Type (Important for Basis Generation)
            source_run_type = source_metadata.get('run_type')
            print(f"Source data run_type detected in metadata: '{source_run_type}'")
            if source_run_type != BASELINE_RUN_TYPE and not args.force_non_baseline:
                 print(f"\n{'='*15} WARNING: Non-Baseline Source {'='*15}")
                 print(f" Generating basis from data whose run_type is '{source_run_type}'. Expected '{BASELINE_RUN_TYPE}'.")
                 print(f" Source Run: {source_run_dir.name} (RunID: {source_run_id})")
                 print(f" Source vector file: {input_vector_path.name}")
                 print("\n Basis generation aborted. Use --force_non_baseline to override.")
                 print(f"{'='*58}")
                 sys.exit(1)
            elif source_run_type != BASELINE_RUN_TYPE and args.force_non_baseline:
                 print(f"\n{'='*15} WARNING: Non-Baseline Source {'='*15}")
                 print(" --force_non_baseline flag detected. Proceeding with basis generation.")
                 print(f" Source Run: {source_run_dir.name} (RunID: {source_run_id})")
                 print(" WARNING: Resulting basis may be influenced by interventions.")
                 print(f"{'='*58}\n")
            else:
                 print(f"Source data validated as '{BASELINE_RUN_TYPE}'. Proceeding.")

            # Define output directory *within the source run*
            basis_dir = source_run_dir / utils.BASIS_SUBFOLDER
            try:
                basis_dir.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                print(f"Error creating basis directory '{basis_dir}': {e}")
                sys.exit(1)

            # Determine output label
            final_output_label = None
            if args.output_basis_label:
                sanitized_user_label = utils.sanitize_label(args.output_basis_label)
                # Check if sanitized label is meaningful
                if sanitized_user_label and sanitized_user_label != "unlabeled":
                    final_output_label = sanitized_user_label
                    print(f"Using provided (sanitized) label: '{final_output_label}'")
                else:
                    print("Warning: Provided label was empty or invalid after sanitization. Generating default label.")
            # Generate default if no valid user label provided
            if not final_output_label:
                 print("Generating default basis label.")
                 default_label_raw = generate_default_basis_label(args)
                 final_output_label = utils.sanitize_label(default_label_raw)
                 print(f"Using auto-generated label: '{final_output_label}'")

            # Define output paths relative to the source run directory
            base_filename = f"basis_{final_output_label}"
            output_npz_path = basis_dir / f"{base_filename}.npz"
            output_json_path = basis_dir / f"{base_filename}.json"
            # Print paths relative to the source run for clarity
            print(f"Output basis NPZ will be saved to: {output_npz_path.relative_to(source_run_dir)}")
            print(f"Output metadata JSON will be saved to: {output_json_path.relative_to(source_run_dir)}")

            # Prepare Metadata for data-derived modes
            basis_metadata = {
                "source_run_id": source_run_id, # Critical provenance link
                "script_name": Path(__file__).name,
                "generation_timestamp": generation_timestamp.isoformat(),
                "basis_generation_mode": args.mode,
                "source_run_directory": str(source_run_dir),
                "source_run_directory_name": source_run_dir.name,
                "source_vector_file_relative": str(input_vector_path.relative_to(source_run_dir)),
                "source_vector_metadata": source_metadata, # Embed metadata from source vectors
                "output_basis_directory_relative": str(basis_dir.relative_to(source_run_dir)),
                "output_basis_file_relative": str(output_npz_path.relative_to(basis_dir)), # Relative to basis_dir itself
                "output_metadata_file_relative": str(output_json_path.relative_to(basis_dir)), # Relative to basis_dir itself
                "user_provided_output_basis_label": args.output_basis_label,
                "generated_filename_label": final_output_label,
                "dimension": utils.DIMENSION,
                "min_vectors_per_group": args.min_vectors_per_group,
                "force_non_baseline_used": args.force_non_baseline,
                "cli_args": vars(args),
                "single_plane_details": None,
                "ensemble_details": None,
                # Fields not relevant to data modes
                "neuron_1": None, "neuron_2": None, "output_basis_directory": None,
                "output_basis_file": None, "output_metadata_file": None,
            }

            # --- Calculate Basis Vectors ---
            if args.mode == 'single_plane':
                filters1 = utils.parse_filter_string(args.filter_set_1)
                filters2 = utils.parse_filter_string(args.filter_set_2)
                if filters1 is None or filters2 is None:
                    print("Error parsing filter strings. Please check format (e.g., 'key=value,key2=value2').")
                    sys.exit(1)

                print(f"Filtering for Basis 1 with: {filters1}")
                vectors1 = utils.filter_data(structured_data, filters1)
                num_vec1 = len(vectors1)
                print(f"Found {num_vec1} vectors for Basis 1.")

                print(f"Filtering for Basis 2 with: {filters2}")
                vectors2 = utils.filter_data(structured_data, filters2)
                num_vec2 = len(vectors2)
                print(f"Found {num_vec2} vectors for Basis 2.")

                basis_metadata["single_plane_details"] = {
                    "filter_set_1_str": args.filter_set_1,
                    "filter_set_2_str": args.filter_set_2,
                    "filter_set_1_parsed": filters1,
                    "filter_set_2_parsed": filters2,
                    "num_vectors_basis_1": num_vec1,
                    "num_vectors_basis_2": num_vec2
                }

                basis_1 = None
                basis_2 = None
                if num_vec1 < args.min_vectors_per_group or num_vec2 < args.min_vectors_per_group:
                     print(f"Error: Insufficient vectors found (Min required: {args.min_vectors_per_group}). Basis file NOT saved.")
                     basis_saved = False
                else:
                    basis_1 = utils.calculate_mean_vector(vectors1)
                    basis_2 = utils.calculate_mean_vector(vectors2)
                    if basis_1 is not None and basis_2 is not None:
                        print(f"Calculated basis_1 (shape {basis_1.shape}) and basis_2 (shape {basis_2.shape}).")
                        basis_saved = utils.save_basis_vectors(output_npz_path, basis_1=basis_1, basis_2=basis_2, metadata=basis_metadata)
                    else:
                        print("Basis vectors could not be calculated (mean calculation failed). No files saved.")
                        basis_saved = False

            elif args.mode == 'ensemble':
                fixed_filters = utils.parse_filter_string(args.fixed_filters)
                # Check if parsing failed (returned None)
                if fixed_filters is None and args.fixed_filters is not None: # Check if user provided string but parsing failed
                     print(f"Error parsing fixed filters string: '{args.fixed_filters}'. Please check format.")
                     sys.exit(1)

                if fixed_filters:
                    print(f"Applying fixed filters: {fixed_filters}")
                    # Filter data first
                    filtered_structured_data = utils.filter_data(structured_data, fixed_filters)
                    # Re-structure data for grouping (filter_data returns list of vectors)
                    # This requires going back to structured_data unfortunately, or redesigning filter_data
                    # Let's re-filter structured_data properly:
                    filtered_structured_data_items = [
                         item for item in structured_data if all(
                              str(item['key_components'].get(f_key)).lower() == str(f_value).lower()
                              for f_key, f_value in fixed_filters.items()
                              # Handle case where filter key might not be in components cleanly
                              if f_key in item['key_components'] or (f_key not in item['key_components'] and f_value is None) # Match if key missing AND filter value is None? Risky. Let's require key presence.
                         ) and all(f_key in item['key_components'] for f_key in fixed_filters) # Ensure all filter keys are present
                    ]

                    print(f"Found {len(filtered_structured_data_items)} vectors matching fixed filters.")
                    if not filtered_structured_data_items:
                        print("Error: No data remaining after applying fixed filters.")
                        sys.exit(0) # Exit gracefully if no data matches
                else:
                    print("No fixed filters applied.")
                    filtered_structured_data_items = structured_data

                # Group the potentially pre-filtered data
                group_key = args.ensemble_group_key
                print(f"Grouping by key: '{group_key}'")
                grouped_data = defaultdict(list)
                missing_group_key_count = 0
                for item in filtered_structured_data_items:
                    group_val_raw = item['key_components'].get(group_key)
                    if group_val_raw is not None:
                        # Convert group value to string for consistent dict keys
                        grouped_data[str(group_val_raw)].append(item['vector'])
                    else:
                        missing_group_key_count += 1

                if not grouped_data:
                    print(f"Error: No vectors found with group key '{group_key}' after applying any fixed filters.")
                    sys.exit(0)
                if missing_group_key_count > 0:
                    print(f"Warning: {missing_group_key_count} vectors lacked the group key '{group_key}'.")

                # Calculate mean for each valid group
                ensemble_vectors = []
                group_labels = []
                group_details_meta = {}
                skipped_groups_count = 0
                valid_groups_found = 0
                # Sort group keys for deterministic output order
                sorted_group_keys = sorted(grouped_data.keys())
                print(f"Found {len(sorted_group_keys)} potential groups based on key '{group_key}'. Processing...")

                for group_value in sorted_group_keys:
                    vectors_in_group = grouped_data[group_value]
                    count = len(vectors_in_group)
                    if count >= args.min_vectors_per_group:
                        mean_vec = utils.calculate_mean_vector(vectors_in_group)
                        if mean_vec is not None:
                            ensemble_vectors.append(mean_vec)
                            group_labels.append(str(group_value)) # Store the group value as label
                            group_details_meta[str(group_value)] = count # Store count for metadata
                            valid_groups_found += 1
                            # print(f"  Processed group '{group_value}': {count} vectors -> mean calculated.")
                        else:
                            print(f"Warning: Mean calculation failed for group '{group_value}' ({count} vectors). Skipping group.")
                            skipped_groups_count += 1
                    else:
                        # print(f"  Skipping group '{group_value}': Only {count} vectors (min required: {args.min_vectors_per_group}).")
                        skipped_groups_count += 1

                num_generated = len(ensemble_vectors)
                print(f"\nGenerated {num_generated} basis vectors for {valid_groups_found} valid groups.")
                if skipped_groups_count > 0:
                    print(f"Skipped {skipped_groups_count} groups due to insufficient vectors (min {args.min_vectors_per_group}) or mean calculation error.")

                basis_metadata["ensemble_details"] = {
                    "ensemble_group_key": args.ensemble_group_key,
                    "fixed_filters_str": args.fixed_filters,
                    "fixed_filters_parsed": fixed_filters,
                    "output_key_in_npz": args.output_key,
                    "group_vector_counts": group_details_meta, # Dict of {group_value: count}
                    "generated_group_labels": group_labels,   # List of group values used
                    "num_potential_groups_found": len(grouped_data),
                    "num_groups_skipped": skipped_groups_count,
                    "num_basis_vectors_generated": num_generated,
                }

                # Check if enough vectors were generated for ensemble analysis
                if num_generated < 2:
                     print(f"Error: Need at least 2 basis vectors for ensemble mode (found {num_generated}). Basis file NOT saved.")
                     basis_saved = False
                else:
                    ensemble_array = np.array(ensemble_vectors, dtype=np.float32) # Ensure float32
                    print(f"Final ensemble basis shape: {ensemble_array.shape}")
                    basis_saved = utils.save_basis_vectors(
                        output_npz_path,
                        ensemble_basis=ensemble_array,
                        ensemble_key=args.output_key,
                        group_labels=group_labels,
                        metadata=basis_metadata
                    )

            # --- Save Standalone Metadata JSON (Data-Derived Modes) ---
            if basis_saved:
                # Print path relative to source run dir for clarity
                print(f"\nSaving basis generation metadata to: {output_json_path.relative_to(source_run_dir)}")
                if not utils.save_json_metadata(output_json_path, basis_metadata):
                     print("--- ERROR saving basis metadata JSON file ---")
            else:
                 print("\nBasis vectors were not saved. Skipping metadata JSON file generation.")


        else:
             # This case should not be reachable due to argparse choices
             print(f"Error: Unknown mode '{args.mode}'.")
             sys.exit(1)

    except Exception as e:
        print(f"\n--- An error occurred during basis generation ---")
        traceback.print_exc()
        print("Basis generation failed.")
        basis_saved = False # Ensure flag is false on error

    print("\nScript finished.")
    # Optional: Add exit code based on success
    # sys.exit(0 if basis_saved else 1)

# --- END OF FILE generate_basis_vectors.py ---