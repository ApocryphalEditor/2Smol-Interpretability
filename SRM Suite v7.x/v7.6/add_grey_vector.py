# --- START OF FILE add_grey_vector.py ---

import argparse
import json
import math
import sys
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
import datetime # Added import

# Assume utils.py is in the same directory or accessible
try:
    import utils # Still need utils for other functions like load_metadata etc.
    # --- Define Constants using utils if available ---
    DEFAULT_EXPERIMENT_BASE_DIR = getattr(utils, 'DEFAULT_EXPERIMENT_BASE_DIR', "experiments")
    ANALYSES_SUBFOLDER_NAME = getattr(utils, 'ANALYSES_SUBFOLDER', "analyses")
    METADATA_SUBFOLDER_NAME = getattr(utils, 'METADATA_SUBFOLDER', "metadata")
    BASELINE_GROUP_NAME_HEURISTIC = getattr(utils, 'BASELINE_GROUP_NAME_HEURISTIC', "all")
    DIMENSION = getattr(utils, 'DIMENSION', 3072)
    EPSILON = getattr(utils, 'EPSILON', 1e-9)
    CAPTURE_SUBFOLDER_NAME = getattr(utils, 'CAPTURE_SUBFOLDER', 'capture') # Added for fallback
    VECTORS_SUBFOLDER_NAME = getattr(utils, 'VECTORS_SUBFOLDER', 'vectors') # Added for fallback
    # Get the analysis prefix, use fallback if not defined in utils
    try:
        # Use analysis prefix constant if defined in utils, otherwise fallback
        ANALYSIS_PREFIX = getattr(utils, 'DEFAULT_ANALYSIS_LABEL_PREFIX', "srm_analysis")
    except AttributeError: # Handle case where constant name itself might change
        print("Warning: Failed to get 'DEFAULT_ANALYSIS_LABEL_PREFIX' from utils.py, using fallback 'srm_analysis'.")
        ANALYSIS_PREFIX = "srm_analysis"
    # --- End Constants ---
except ImportError:
    print("Error: Could not import 'utils.py'. Make sure it's in the same directory or Python path.")
    sys.exit(1)


# --- Core Logic Function ---
def calculate_and_add_grey_vector(analysis_dir: Path):
    """
    Loads analysis metadata, calculates the Grey Vector for the baseline group,
    and updates the metadata file.
    """
    print(f"\n--- Processing Analysis Directory: {analysis_dir.name} ---")
    # Use constant for subfolder names
    metadata_path = analysis_dir / METADATA_SUBFOLDER_NAME / "analysis_metadata.json"
    if not metadata_path.is_file(): print(f"Error: Metadata file not found: {metadata_path}"); return False
    print(f"Loading analysis metadata from: {metadata_path.name}")
    metadata = utils.load_metadata(metadata_path) # Use load_metadata from utils
    if not metadata: print("Error: Failed to load analysis metadata."); return False

    if metadata.get("grey_vector_summary") is not None:
        print("Info: 'grey_vector_summary' already exists. Skipping calculation.")
        return True
    if metadata.get("analysis_mode") != 'single_plane':
        print(f"Info: Grey Vector calculation only applicable for 'single_plane' mode. Found '{metadata.get('analysis_mode')}'. Skipping.")
        return True

    try: # Extract info
        source_run_dir_str = metadata.get("source_capture_run_directory")
        source_vec_rel_path_str = metadata.get("source_capture_vector_file_relative") # Get the string path
        basis_file_path_str = metadata.get("basis_file_path_resolved")
        grouping_key = metadata.get("grouping_key") # This might be None, 'all', or 'all_vectors'

        # --- CORRECTED BASELINE GROUP NAME CHECK ---
        baseline_group_name = BASELINE_GROUP_NAME_HEURISTIC if grouping_key in [BASELINE_GROUP_NAME_HEURISTIC, "all_vectors", None] else None

        if not (source_run_dir_str and source_vec_rel_path_str and basis_file_path_str):
            print("Error: Missing required paths in metadata."); return False
        if baseline_group_name is None:
             print(f"Error: Cannot determine baseline group name (grouping_key: '{grouping_key}'). Expected 'all', 'all_vectors', or None."); return False

        source_run_dir = Path(source_run_dir_str)
        source_vec_rel_path = Path(source_vec_rel_path_str) # Convert string to Path
        source_vector_path = source_run_dir / source_vec_rel_path # Construct path
        basis_file_path = Path(basis_file_path_str)

        # Check and apply fallbacks for source_vector_path
        if not source_vector_path.is_file():
             print(f"Warning: Source vector file not found at primary path: {source_vector_path}")
             # Use constants (local or from utils)
             alt_path = source_run_dir / CAPTURE_SUBFOLDER_NAME / source_vec_rel_path
             if alt_path.is_file(): source_vector_path = alt_path; print(f"Info: Used fallback path 1: {alt_path}")
             else:
                 alt_path2 = source_run_dir / CAPTURE_SUBFOLDER_NAME / VECTORS_SUBFOLDER_NAME / source_vec_rel_path.name
                 if alt_path2.is_file(): source_vector_path = alt_path2; print(f"Info: Used fallback path 2: {alt_path2}")
                 else: print(f"Error: Source vector file still not found."); return False
        if not basis_file_path.is_file(): print(f"Error: Basis file not found: {basis_file_path}"); return False

    except Exception as e: print(f"Error extracting info from metadata: {e}"); traceback.print_exc(); return False

    print(f"Identified Baseline Group: '{baseline_group_name}'")
    print(f"Source Vectors: {source_vector_path.name} (in {source_run_dir.name})")
    print(f"Basis File: {basis_file_path.name}")

    # Load Source Vectors
    print("Loading baseline vectors...")
    expected_dim = DIMENSION # Use constant (local or from utils)
    structured_data, _ = utils.load_vector_data(source_vector_path, expected_dim=expected_dim)
    if structured_data is None: print("Error: Failed to load baseline vectors."); return False
    baseline_vectors = [item['vector'] for item in structured_data]
    if not baseline_vectors: print("Error: No vectors found in source file."); return False
    print(f"Loaded {len(baseline_vectors)} baseline vectors.")

    # Load Basis Vectors
    print("Loading basis vectors...")
    try:
        basis_data = np.load(basis_file_path, allow_pickle=True)
        if 'basis_1' not in basis_data or 'basis_2' not in basis_data: print(f"Error: Keys 'basis_1'/'basis_2' not found in {basis_file_path.name}."); return False
        basis_1 = basis_data['basis_1'].astype(np.float32); basis_2 = basis_data['basis_2'].astype(np.float32)
        if basis_1.shape != (expected_dim,) or basis_2.shape != (expected_dim,): print(f"Error: Basis dim mismatch."); return False
    except Exception as e: print(f"Error loading basis vectors: {e}"); return False

    # Calculate Grey Vector
    print("Calculating Grey Vector...")
    try:
        # Use utils.normalise and utils.EPSILON if available
        normalise_func = getattr(utils, 'normalise', lambda arr, axis=-1: arr / np.linalg.norm(arr, axis=axis, keepdims=True) if np.linalg.norm(arr) > 0 else arr) # More robust fallback
        epsilon_val = EPSILON # Use constant (local or from utils)

        norm_basis_1 = normalise_func(basis_1); norm_basis_2 = normalise_func(basis_2)
        u1 = norm_basis_1; u2_prime = norm_basis_2 - np.dot(norm_basis_2, u1) * u1
        u2_prime_norm = np.linalg.norm(u2_prime)
        if u2_prime_norm < epsilon_val: print("Error: Basis vectors collinear."); return False
        u2 = u2_prime / u2_prime_norm

        baseline_vectors_np = np.stack(baseline_vectors, axis=0)
        normalized_baseline_vectors = normalise_func(baseline_vectors_np, axis=1)
        proj_x = np.dot(normalized_baseline_vectors, u1); proj_y = np.dot(normalized_baseline_vectors, u2)
        mean_x = np.mean(proj_x); mean_y = np.mean(proj_y)

        grey_vec_xy = [float(mean_x), float(mean_y)]
        r_g = math.sqrt(mean_x**2 + mean_y**2)
        theta_g_rad = math.atan2(mean_y, mean_x)
        theta_g_deg = (math.degrees(theta_g_rad) + 360) % 360

        grey_vector_stats = {'vector_xy': grey_vec_xy, 'r': r_g, 'theta_deg': theta_g_deg}
        print(f"  Grey Vector calculated: r={r_g:.4f}, theta={theta_g_deg:.2f}\u00b0")
        # Use the determined baseline_group_name as the key
        grey_vector_summary = {baseline_group_name: grey_vector_stats}

    except Exception as e: print(f"Error during Grey Vector calculation: {e}"); traceback.print_exc(); return False

    # Update & Save Metadata
    metadata["grey_vector_summary"] = grey_vector_summary
    metadata["analysis_timestamp_grey_vector_added"] = datetime.datetime.now().isoformat()
    print(f"Saving updated metadata back to: {metadata_path.name}")
    # Use utils.save_json_metadata
    if utils.save_json_metadata(metadata_path, metadata):
        print("Successfully updated metadata file with Grey Vector summary.")
        return True
    else:
        print("Error: Failed to save updated metadata file.")
        return False

# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate Grey Vector (baseline drift) for a completed single-plane SRM analysis run and add it to its metadata."
    )
    parser.add_argument(
        "--experiment_base_dir",
        type=str,
        default=DEFAULT_EXPERIMENT_BASE_DIR, # Use constant
        help="Base directory containing experiment run folders."
    )

    args = parser.parse_args()

    base_dir = Path(args.experiment_base_dir).resolve()
    if not base_dir.is_dir():
        print(f"Error: Experiment base directory not found: {base_dir}")
        sys.exit(1)

    # --- Find and Select Analysis Directory with DEBUG PRINTS ---
    print(f"\nSearching for analysis folders in '{base_dir}'...")
    analysis_dirs = []
    try:
        # Find all 'analyses' subdirectories first
        print(f"DEBUG: Searching for subdirs named '{ANALYSES_SUBFOLDER_NAME}' using rglob from '{base_dir}'...") # DEBUG
        analyses_subdirs = list(base_dir.rglob(ANALYSES_SUBFOLDER_NAME))
        print(f"DEBUG: Found {len(analyses_subdirs)} potential '{ANALYSES_SUBFOLDER_NAME}' paths:") # DEBUG
        if not analyses_subdirs: print("DEBUG:   (None found)")
        for p in analyses_subdirs: print(f"DEBUG:   - {p}") # DEBUG

        # Look inside each 'analyses' subdir for folders matching the prefix
        print(f"DEBUG: Filtering based on directory structure...") # DEBUG
        for analyses_path in analyses_subdirs:
            print(f"DEBUG: Checking path: {analyses_path}") # DEBUG
            is_dir = analyses_path.is_dir()
            is_correct_name = analyses_path.name == ANALYSES_SUBFOLDER_NAME

            # --- MODIFIED: Expanded try-except for parent check ---
            parent_is_dir = False # Default
            try:
                parent_is_dir = analyses_path.parent.is_dir()
            except Exception:
                pass # Ignore errors if parent access fails
            # --- END MODIFICATION ---

            print(f"DEBUG:   Is Dir: {is_dir}, Name OK: {is_correct_name}, Parent Is Dir: {parent_is_dir}") # DEBUG

            if is_dir and is_correct_name and parent_is_dir:
                print(f"DEBUG:   '{analyses_path.name}' is a valid 'analyses' dir. Iterating children...") # DEBUG
                for potential_analysis_dir in analyses_path.iterdir():
                    print(f"DEBUG:     Checking child: {potential_analysis_dir.name} (Is Dir: {potential_analysis_dir.is_dir()})") # DEBUG
                    # --- Modified Line: Removed startswith check ---
                    if potential_analysis_dir.is_dir(): # Just check if it's a directory inside 'analyses'
                    # --- End Modification ---
                        print(f"DEBUG:       MATCH FOUND (is directory): {potential_analysis_dir.name}") # DEBUG
                        analysis_dirs.append(potential_analysis_dir)
                    # else: print(f"DEBUG:       No match") # DEBUG
            else:
                print(f"DEBUG:   '{analyses_path.name}' failed validation checks.") # DEBUG

        analysis_dirs = sorted(analysis_dirs) # Sort the final list
        print(f"DEBUG: Final list of analysis_dirs found: {len(analysis_dirs)}") # DEBUG
        if not analysis_dirs: print(f"DEBUG: No directories passed the filter criteria.") # DEBUG

    except Exception as e:
        print(f"Error searching for analysis directories: {e}")
        traceback.print_exc()
        sys.exit(1)

    if not analysis_dirs:
        print(f"Error: No analysis sub-directories found within any '{ANALYSES_SUBFOLDER_NAME}' folder in {base_dir}.")
        sys.exit(1)

    # --- Selection logic ---
    selected_analysis_dir = None
    if len(analysis_dirs) == 1:
        selected_analysis_dir = analysis_dirs[0]
        try: rel_path = selected_analysis_dir.relative_to(base_dir); print(f"Auto-selected analysis directory: {rel_path}")
        except ValueError: print(f"Auto-selected analysis directory: {selected_analysis_dir}")
    else:
        print("\nMultiple analysis directories found:")
        for i, dir_path in enumerate(analysis_dirs):
            # Safely get run_dir_name, handle potential IndexError if structure is unexpected
            try:
                run_dir_name = dir_path.parent.parent.name
            except IndexError:
                run_dir_name = "[Unknown Run Dir]"

            try: rel_path = dir_path.relative_to(base_dir); print(f"  {i+1}: {rel_path}  [Run: {run_dir_name}]")
            except ValueError: print(f"  {i+1}: {dir_path}  [Run: {run_dir_name}]") # Fallback

        while selected_analysis_dir is None:
            try:
                choice = input(f"Enter the number to process (1-{len(analysis_dirs)}), or 0 to cancel: ")
                if choice == '0': print("Selection cancelled."); sys.exit(0)
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(analysis_dirs):
                    selected_analysis_dir = analysis_dirs[choice_idx]
                    try: rel_path = selected_analysis_dir.relative_to(base_dir); print(f"Selected: {rel_path}")
                    except ValueError: print(f"Selected: {selected_analysis_dir}")
                else: print("Invalid choice.")
            except ValueError: print("Invalid input. Please enter a number.")
            except (EOFError, KeyboardInterrupt): print("\nSelection cancelled."); sys.exit(1)
    # --- End Selection logic ---

    success = calculate_and_add_grey_vector(selected_analysis_dir.resolve())

    if success: print("\nScript finished successfully."); sys.exit(0)
    else: print("\nScript finished with errors."); sys.exit(1)

# --- END OF FILE add_grey_vector.py ---