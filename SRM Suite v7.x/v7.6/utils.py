# --- START OF FILE utils.py ---

import os
import numpy as np
import json
import re # <--- Make sure re is imported
import datetime
import traceback
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as cm
import random
import itertools
import time # Added for potential locking delay
import threading # Added for basic counter locking

# --- Constants ---
DIMENSION = 3072 # Assuming GPT-2 small MLP dimension
VALID_GROUP_KEYS = ['core_id', 'type', 'level', 'sweep']
CAPTURE_SUBFOLDER = "capture"
ANALYSES_SUBFOLDER = "analyses"
BASIS_SUBFOLDER = "basis"
VECTORS_SUBFOLDER = "vectors"
LOGS_SUBFOLDER = "logs"
METADATA_SUBFOLDER = "metadata"
PLOTS_SUBFOLDER = "plots" # Added constant for clarity in plotting
DATA_SUBFOLDER = "data" # As above
DEFAULT_RUN_COUNTER_FILE = Path("./_run_counter.txt") # Default path for the run counter
BASELINE_GROUP_NAME_HEURISTIC = "all" # Default name for the group containing all baseline vectors
EPSILON = 1e-9 # Small value for numerical stability (e.g., checking collinearity, normalization)
# Get default onehot dir from utils if possible, otherwise use local default
DEFAULT_ONEHOT_OUTPUT_DIR = Path("./generated_basis") # Needs to be defined if used directly
# Use analysis prefix constant if defined in utils, otherwise fallback
DEFAULT_ANALYSIS_LABEL_PREFIX = "srm_analysis"


# === NEW Run ID and Timestamp Generation (Step 5) ===
_counter_lock = threading.Lock() # Basic lock for thread safety if run in parallel threads

def generate_run_id(prefix: str, counter_file_path: Path = DEFAULT_RUN_COUNTER_FILE) -> str:
    """
    Generates a unique run ID using a prefix and an incrementing counter
    stored in a file. Uses basic threading lock for safety.
    Format: PREFIX-NNN (e.g., EXP-001)
    """
    if not prefix.isalnum():
        prefix = "EXP" # Default to EXP if prefix is weird
        print(f"Warning: Invalid prefix provided. Using default '{prefix}'.")

    with _counter_lock:
        counter = 1 # Default if file doesn't exist or is invalid
        try:
            if counter_file_path.is_file():
                current_val_str = counter_file_path.read_text().strip()
                if current_val_str.isdigit():
                    counter = int(current_val_str) + 1
                else:
                    print(f"Warning: Invalid content in counter file {counter_file_path}. Resetting counter to 1.")
            else:
                print(f"Info: Counter file {counter_file_path} not found. Starting counter at 1.")

            # Write the new counter value back
            counter_file_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
            counter_file_path.write_text(str(counter))

        except Exception as e:
            print(f"Error accessing counter file {counter_file_path}: {e}. Using counter value {counter} but not saving.")
            # Decide if you want to proceed without saving or raise error? For now, proceed.

    return f"{prefix.upper()}-{counter:03d}" # Use upper case prefix, 3-digit padding

def get_formatted_timestamp() -> str:
    """
    Returns the current timestamp in DDMMMYY_HHMM format (e.g., 22APR25_1610).
    """
    # Use %b for locale's abbreviated month name, upper() for consistency
    return datetime.datetime.now().strftime("%d%b%y_%H%M").upper()

# === Directory and File Handling ===

def list_experiment_folders(base_dir: Path, pattern: str = "run_*") -> list[Path]:
    """Lists directories matching a pattern within the base directory."""
    if not base_dir.is_dir():
        return []
    # MODIFIED: More flexible pattern matching to handle new RunID format too
    dirs = []
    for d in base_dir.iterdir():
        if d.is_dir() and not d.name.startswith('.'):
            # Check if the pattern is a simple prefix match or contains wildcards
            if '*' in pattern or '?' in pattern: # Allow standard glob patterns
                 if d.match(pattern):
                     dirs.append(d)
            elif d.name.startswith(pattern): # Original behavior for simple prefixes
                 dirs.append(d)
            # Add specific check for new RunID format if pattern is generic like "run_*" or "*"
            elif pattern in ["run_*", "*"] and re.match(r"^[A-Z]+-\d{3}_.*", d.name): # Matches EXP-001_...
                 dirs.append(d)

    return sorted(dirs)


def select_experiment_folder(base_dir: Path, prompt_text: str = "Select an experiment run directory:", pattern: str = "*") -> Path | None:
    # MODIFIED: Default pattern to "*" to find all types of run folders
    """Interactively prompts the user to select a directory from a list."""
    run_dirs = list_experiment_folders(base_dir, pattern)
    if not run_dirs:
        print(f"Error: No directories matching '{pattern}' found in {base_dir}.")
        return None

    print(f"\n{prompt_text}")
    for i, dir_path in enumerate(run_dirs):
        print(f"  {i+1}: {dir_path.name}")

    selected_dir = None
    while selected_dir is None:
        try:
            choice = input(f"Enter the number of the directory (1-{len(run_dirs)}): ")
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(run_dirs):
                selected_dir = run_dirs[choice_idx]
                print(f"Selected: {selected_dir.name}")
            else:
                print("Invalid choice.")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except (EOFError, KeyboardInterrupt):
            print("\nSelection cancelled.")
            return None
    return selected_dir.resolve()

def find_latest_file(directory: Path, pattern: str = "*.npz") -> Path | None:
    """Finds the most recently modified file matching the pattern in a directory."""
    if not directory.is_dir():
        return None
    try:
        files = sorted(directory.glob(pattern), key=lambda x: x.stat().st_mtime, reverse=True)
        return files[0] if files else None
    except PermissionError:
        print(f"Warning: Permission denied when accessing directory {directory}")
        return None
    except Exception as e:
        print(f"Warning: Error finding latest file in {directory}: {e}")
        return None


def find_file_interactive(
    directory: Path,
    pattern: str = "*.npz",
    file_type_desc: str = "file",
    run_dir_level: int | None = None # NEW: Levels up from file path to find run dir name
    ) -> Path | None:
    """
    Finds files matching a pattern and prompts the user if multiple are found.
    Optionally displays parent run directory name for context if run_dir_level is provided.
    """
    if not directory.is_dir():
        print(f"Error: Directory for interactive search not found: {directory}")
        return None

    try:
        # Sort alphabetically for consistent listing
        files = sorted(list(directory.glob(pattern)))
    except PermissionError:
        print(f"Error: Permission denied when searching for files in {directory}")
        return None
    except Exception as e:
        print(f"Error searching for files in {directory}: {e}")
        return None


    if not files:
        print(f"Info: No {file_type_desc} files matching '{pattern}' found in {directory}")
        return None
    elif len(files) == 1:
        print(f"Auto-selected {file_type_desc} file: {files[0].name}")
        return files[0]
    else:
        print(f"Multiple {file_type_desc} files found matching '{pattern}' in {directory.name}:")
        for i, fp in enumerate(files):
            context_str = ""
            # --- START MODIFICATION for Run Dir Context ---
            if run_dir_level is not None and run_dir_level > 0:
                try:
                    # fp.parents is 0-indexed (parent 0 is immediate dir)
                    # So level 1 = parent 0, level 2 = parent 1, etc.
                    # We need parent at index (run_dir_level - 1)
                    if run_dir_level <= len(fp.parents):
                         run_folder_name = fp.parents[run_dir_level - 1].name
                         context_str = f"  [Run Dir: {run_folder_name}]"
                    else:
                         context_str = "  [Run Dir: Path too short]"
                except IndexError:
                    context_str = "  [Run Dir: Error finding parent]"
                except Exception as e_ctx:
                    context_str = f"  [Run Dir: Error ({e_ctx})]"
            # --- END MODIFICATION ---
            print(f"  {i+1}: {fp.name}{context_str}") # Display file name and context


        selected_file = None
        while selected_file is None:
            try:
                choice = input(f"Enter the number of the {file_type_desc} file to use (1-{len(files)}): ")
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(files):
                    selected_file = files[choice_idx]
                    print(f"Selected {file_type_desc} file: {selected_file.name}")
                else:
                    print("Invalid choice.")
            except ValueError:
                print("Invalid input. Please enter a number.")
            except (EOFError, KeyboardInterrupt):
                print("\nSelection cancelled.")
                return None
        return selected_file

def find_vector_file(run_dir: Path) -> Path | None:
    """
    Finds the vector NPZ file within the standard capture/vectors subdirectory.
    Prompts if multiple files exist, showing run directory context.
    """
    vector_dir = run_dir / CAPTURE_SUBFOLDER / VECTORS_SUBFOLDER
    # run_dir_level = 3 relative to the file path `fp`.
    return find_file_interactive(vector_dir, "*.npz", "vector", run_dir_level=3)

def find_basis_file(basis_search_dir: Path, specific_filename: str | None = None) -> Path | None:
    """
    Finds a basis NPZ file within the given search directory (expected to be a 'basis' subfolder).
    If specific_filename is given, looks for that. Otherwise, prompts if multiple are found.
    Shows run directory context during interactive selection.
    """
    basis_dir = basis_search_dir
    # Check if basis_search_dir is the run dir itself, or already the basis subfolder
    if not basis_dir.name == BASIS_SUBFOLDER:
         potential_basis_dir = basis_search_dir / BASIS_SUBFOLDER
         if potential_basis_dir.is_dir():
              basis_dir = potential_basis_dir
         # else: # Keep basis_dir as basis_search_dir if BASIS_SUBFOLDER doesn't exist directly under it
             # print(f"Info: Basis subfolder '{BASIS_SUBFOLDER}' not found directly under {basis_search_dir}. Searching in {basis_search_dir} itself.")

    if not basis_dir.is_dir():
         print(f"Info: Basis directory not found: {basis_dir}")
         return None

    # run_dir_level = 2 relative to the file path `fp`.
    file_finder_func = find_file_interactive

    if specific_filename:
        target_file = basis_dir / specific_filename
        if target_file.is_file():
            print(f"Found specified basis file: {target_file.name}")
            return target_file
        else:
            print(f"Info: Specified basis file '{specific_filename}' not found in {basis_dir}.")
            print("Searching for other basis files...")
            # Fallback to interactive search if specific file not found
            return file_finder_func(basis_dir, "*.npz", "basis", run_dir_level=2)
    else:
        # Standard interactive search
        return file_finder_func(basis_dir, "*.npz", "basis", run_dir_level=2)


# === Data Loading and Parsing ===

def sanitize_label(label: str | None) -> str:
    """ Replaces potentially problematic characters in a label for filenames/tags. """
    if not label: return "unlabeled"
    # Keep alphanumeric, underscores, hyphens. Replace others with underscore.
    sanitized = re.sub(r'[^\w\-]+', '_', label)
    # Collapse multiple underscores, remove leading/trailing underscores
    sanitized = re.sub(r'_+', '_', sanitized).strip('_')
    return sanitized[:100] # Limit length


def parse_vector_key(key_str: str) -> dict:
    """Parses key=value pairs from vector NPZ keys (designed for original format)."""
    # This function parses the OLD key format (core_id=..._type=..._level=...)
    components = {}
    # Simple parsing based on expected keys and separators
    parts = key_str.split('_')
    for part in parts:
        if '=' in part:
            key, value = part.split('=', 1)
            key = key.strip()
            if key in VALID_GROUP_KEYS:
                 if key == 'level': # Attempt to cast level to int
                     try: value = int(value)
                     except ValueError: pass
                 elif key == 'sweep': # Attempt to cast sweep to float/int or handle 'baseline'
                     if value.lower() == 'baseline': # Keep 'baseline' as string
                          pass
                     else:
                          try:
                              val_f = float(value)
                              value = int(val_f) if val_f.is_integer() else val_f
                          except ValueError: pass # Keep as string if not number
                 components[key] = value
            elif key == 'core_id': # Handle core_id separately as it might contain underscores
                 components[key] = value # Assume value is everything after '=' for core_id
    return components


def load_vector_data(npz_path: Path, expected_dim: int = DIMENSION) -> tuple[list[dict], dict] | tuple[None, None]:
    """Loads vectors from NPZ, parses keys, and returns structured list and source metadata."""
    structured_vectors = []
    metadata = {} # This will hold metadata *embedded* in the NPZ

    if not npz_path.is_file():
        print(f"Error: Input vector file not found: '{npz_path}'")
        return None, None

    print(f"\nLoading vectors and keys from: {npz_path.name}")
    try:
        with np.load(npz_path, allow_pickle=True) as loaded_data:
            # Load embedded metadata first
            if '__metadata__' in loaded_data:
                try:
                    # Ensure it's treated as a Python object
                    metadata_obj = loaded_data['__metadata__'].item()
                    if isinstance(metadata_obj, dict):
                        metadata = metadata_obj
                        print("Loaded metadata embedded in NPZ file.")
                        # --- Check for RunID in loaded metadata ---
                        if 'run_id' in metadata:
                            print(f"  Found embedded RunID: {metadata['run_id']}")
                        else:
                            print("  Warning: Embedded metadata lacks 'run_id'.")
                        # ---
                    else:
                        print("Warning: Embedded '__metadata__' is not a dictionary. Ignoring.")
                        metadata = {}
                except Exception as meta_e:
                    print(f"Warning: Could not load embedded '__metadata__': {meta_e}")
                    metadata = {}

            # Process vector keys
            vector_keys = [k for k in loaded_data.files if k != '__metadata__']
            print(f"Found {len(vector_keys)} potential vector keys in the input file.")
            valid_count = 0
            skipped_count = 0

            for key in tqdm(vector_keys, desc="Loading vectors", leave=False, unit="key"):
                try:
                    vec = loaded_data[key]
                    if not isinstance(vec, np.ndarray) or vec.shape != (expected_dim,):
                        skipped_count += 1
                        continue

                    key_components = parse_vector_key(key)
                    if not key_components and key != '__metadata__':
                         key_components = {'raw_key': key}

                    structured_vectors.append({
                        'key': key,
                        'key_components': key_components,
                        'vector': vec
                    })
                    valid_count += 1
                except Exception as e:
                    print(f"\nError processing key '{key}': {e}")
                    traceback.print_exc()
                    skipped_count += 1
                    continue
    except FileNotFoundError:
        print(f"Error: Input vector file not found: '{npz_path}'")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred loading {npz_path}: {e}")
        traceback.print_exc()
        return None, None

    if skipped_count > 0:
        print(f"Skipped {skipped_count} invalid or unparsable entries during loading.")
    if not structured_vectors:
        print(f"Warning: No valid vectors loaded from {npz_path}.")
        return [], metadata # Return empty list and potentially loaded metadata
    print(f"Successfully loaded {valid_count} vectors with keys.")
    return structured_vectors, metadata


def load_metadata(metadata_path: Path) -> dict | None:
    """Loads JSON metadata from a file."""
    if not metadata_path.is_file():
        print(f"Warning: Metadata file not found: {metadata_path}")
        return None
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        # --- Check for RunID in loaded metadata ---
        if isinstance(metadata, dict) and 'run_id' in metadata:
             print(f"Loaded JSON metadata from {metadata_path.name}. Found RunID: {metadata['run_id']}")
        elif isinstance(metadata, dict):
             print(f"Loaded JSON metadata from {metadata_path.name}. RunID not found.")
        # ---
        return metadata
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in metadata file: {metadata_path}")
        return None
    except Exception as e:
        print(f"Error loading metadata file {metadata_path}: {e}")
        return None

# === Basis Generation Helpers ===

def parse_filter_string(filter_str: str | None) -> dict | None:
    """ Parses 'key1=value1,key2=value2' into {'key1': 'value1', 'key2': 'value2'} """
    filters = {}
    if not filter_str:
        return filters
    try:
        pairs = filter_str.split(',')
        for pair in pairs:
            pair = pair.strip()
            if not pair: continue
            if '=' not in pair:
                raise ValueError(f"Invalid filter format in pair: '{pair}' (missing '=')")
            key, value = pair.split('=', 1)
            key = key.strip()
            value = value.strip()
            if not key:
                raise ValueError("Filter key cannot be empty.")
            # Attempt to convert numeric values if possible (esp. level)
            if key == 'level' and value.isdigit():
                 value = int(value)
            elif key == 'sweep':
                 if value.lower() == 'baseline':
                     pass # keep as string
                 else:
                     try:
                         val_f = float(value)
                         value = int(val_f) if val_f.is_integer() else val_f
                     except ValueError: pass # keep as string
            filters[key] = value
    except Exception as e:
        print(f"Error parsing filter string '{filter_str}': {e}")
        return None # Indicate error
    return filters


def filter_data(structured_data: list[dict], filters: dict) -> list[np.ndarray]:
    """ Filters the loaded data based on the provided filter dictionary. """
    if not filters:
        return [item['vector'] for item in structured_data]

    matching_vectors = []
    for item in structured_data:
        components = item['key_components']
        match = True
        for key, filter_value in filters.items():
            if key not in components:
                match = False
                break
            component_value = components[key]
            # Compare appropriately (case-insensitive string, or direct for numbers)
            if isinstance(filter_value, str):
                 if not isinstance(component_value, str) or component_value.lower() != filter_value.lower():
                     match = False
                     break
            elif component_value != filter_value: # Direct comparison for numbers/other
                 match = False
                 break
        if match:
            matching_vectors.append(item['vector'])
    return matching_vectors


def calculate_mean_vector(vectors: list[np.ndarray], expected_dim: int = DIMENSION) -> np.ndarray | None:
    """ Calculates the mean vector from a list of vectors. """
    if not vectors:
        return None
    try:
        # Ensure all vectors are numpy arrays before stacking
        vector_list_np = [np.asarray(v) for v in vectors]
        vector_array = np.stack(vector_list_np, axis=0)

        if vector_array.ndim != 2 or vector_array.shape[1] != expected_dim:
            print(f"Error: Invalid shape for mean calculation. Expected [N, {expected_dim}], got {vector_array.shape}")
            return None
        mean_vec = np.mean(vector_array, axis=0)
        # Ensure output is float32 for consistency
        return mean_vec.astype(np.float32)
    except Exception as e:
        print(f"Error calculating mean vector: {e}")
        return None

def save_basis_vectors(
    path: Path,
    basis_1: np.ndarray | None = None,
    basis_2: np.ndarray | None = None,
    ensemble_basis: np.ndarray | None = None,
    ensemble_key: str = 'basis_vectors',
    group_labels: list[str] | np.ndarray | None = None,
    metadata: dict | None = None # Metadata to be embedded
    ):
    """Saves basis vectors (single plane or ensemble) and optional metadata to NPZ."""
    save_dict = {}
    basis_type = "unknown"
    if basis_1 is not None and basis_2 is not None:
        save_dict['basis_1'] = basis_1.astype(np.float32) # Ensure float32
        save_dict['basis_2'] = basis_2.astype(np.float32)
        basis_type = "single_plane"
        print(f"Preparing to save single plane basis (basis_1, basis_2) to {path.name}")
    elif ensemble_basis is not None:
        save_dict[ensemble_key] = ensemble_basis.astype(np.float32) # Ensure float32
        basis_type = "ensemble"
        if group_labels is not None:
            # Ensure labels are saved as numpy array of strings
            save_dict['group_labels'] = np.array(group_labels, dtype=str)
            print(f"Preparing to save ensemble basis (key: '{ensemble_key}') and {len(group_labels)} labels to {path.name}")
        else:
             print(f"Preparing to save ensemble basis (key: '{ensemble_key}') without labels to {path.name}")
    else:
        print("Error: No valid basis vectors provided for saving.")
        return False

    if metadata:
        # --- Embed RunID Check ---
        if 'run_id' in metadata: # Check if source run ID is included if applicable
             print(f"  Including metadata with RunID: {metadata.get('source_run_id', 'N/A')} in NPZ file.")
        elif 'source_run_id' in metadata:
             print(f"  Including metadata with Source RunID: {metadata['source_run_id']} in NPZ file.")
        else:
             # For onehot mode, run_id might not be relevant
             if metadata.get("basis_generation_mode") != 'onehot':
                 print("  Warning: Metadata provided for embedding, but 'source_run_id' is missing.")
        # ---
        save_dict['__metadata__'] = np.array(metadata, dtype=object)

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **save_dict)
        print(f"Successfully saved {basis_type} basis file: {path}")
        return True
    except Exception as e:
        print(f"Error saving basis file {path}: {e}")
        traceback.print_exc()
        return False


def save_json_metadata(path: Path, metadata: dict):
    """Saves a dictionary as a JSON file."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            # Convert Path objects to strings for JSON serialization
            serializable_metadata = json.loads(json.dumps(metadata, default=str))
            json.dump(serializable_metadata, f, indent=4)
        print(f"Successfully saved JSON metadata: {path}")
        # --- Check for RunID ---
        run_id = metadata.get('run_id') or metadata.get('source_capture_run_id') or metadata.get('baseline_run_id')
        if run_id:
             print(f"  JSON metadata includes RunID: {run_id}")
        # ---
        return True
    except TypeError as e:
         print(f"Error serializing metadata to JSON for {path}: {e}")
         print("Metadata dump:", metadata) # Log metadata that failed
         return False
    except Exception as e:
        print(f"Error saving JSON metadata file {path}: {e}")
        traceback.print_exc()
        return False

# === Analysis Helpers (Moved/Adapted from analyze_srm_sweep.py) ===

def normalise(array: np.ndarray, axis: int = -1) -> np.ndarray:
    # Default axis to -1 for common vector normalization
    """Normalizes vectors along a specified axis."""
    norms = np.linalg.norm(array, axis=axis, keepdims=True)
    # Prevent division by zero, set norm to 1 for zero vectors (result will be zero vector)
    norms[norms == 0] = 1.0
    normalized_array = array / norms
    return normalized_array

def vectors_to_bivectors(vector1: np.ndarray, vector2: np.ndarray) -> np.ndarray:
    """Computes the bivector (rotation generator) from two normalized vectors."""
    # Ensure inputs are 1D arrays
    if vector1.ndim != 1 or vector2.ndim != 1:
        raise ValueError("Input vectors must be 1-dimensional.")
    if vector1.shape != vector2.shape:
        raise ValueError("Input vectors must have the same shape.")

    # Ensure vectors are normalized (normalise handles 1D arrays)
    v1_norm = normalise(vector1)
    v2_norm = normalise(vector2)

    # Check for collinearity
    dot_product = np.dot(v1_norm, v2_norm)
    if np.abs(dot_product) > 1.0 - EPSILON: # Use constant EPSILON
        print(f"Warning: Basis vectors are nearly collinear (dot product: {dot_product:.4f}). Bivector will be near zero.")
        # Return a zero matrix of the correct shape
        return np.zeros((vector1.shape[0], vector1.shape[0]), dtype=vector1.dtype)

    outer_product = np.outer(v1_norm, v2_norm)
    rotation_generator = outer_product - outer_product.T
    return rotation_generator

def hermitian_conjugate(array: np.ndarray) -> np.ndarray:
    """Computes the Hermitian conjugate (conjugate transpose)."""
    return np.conj(array).T

def generate_special_orthogonal_matrices(generator: np.ndarray, angles_rad: np.ndarray, debug_log_file=None) -> np.ndarray:
    """Generates SO(D) rotation matrices using the generator and angles (Rodrigues' formula generalization)."""
    # (Original robust implementation kept - no changes needed for Step 5)
    if debug_log_file: debug_log_file.write("\n--- [DEBUG] generate_special_orthogonal_matrices ---\n")
    D = generator.shape[0]
    num_angles = len(angles_rad)
    identity_matrix = np.identity(D, dtype=generator.dtype) # Match dtype

    # Check if generator is near zero
    generator_norm = np.linalg.norm(generator, 'fro') # Use Frobenius norm for matrices
    if debug_log_file: debug_log_file.write(f"[DEBUG] Generator Frobenius norm: {generator_norm:.4e}\n")
    if generator_norm < EPSILON: # Use constant EPSILON
        if debug_log_file: debug_log_file.write("[DEBUG] Generator is near zero. Returning identity matrices.\n")
        return np.array([identity_matrix] * num_angles)

    # Check anti-symmetry
    antisymmetry_check = np.max(np.abs(generator + generator.T))
    if debug_log_file: debug_log_file.write(f"[DEBUG] Antisymmetry check (max|G + G.T|): {antisymmetry_check:.4e}\n")
    if antisymmetry_check > 1e-6: # Check if significantly non-anti-symmetric
        if debug_log_file: debug_log_file.write("[DEBUG] Warning: Generator matrix is not sufficiently anti-symmetric.\n")

    # Use matrix exponential (more robust than eigendecomposition for rotations)
    try:
        from scipy.linalg import expm
        rotation_matrices = np.array([expm(angle * generator) for angle in angles_rad])

        # --- Debug Checks ---
        if debug_log_file:
             debug_log_file.write(f"[DEBUG] Shape of final rotation_matrices: {rotation_matrices.shape}\n")
             if len(rotation_matrices) > 0:
                 matrix_0 = rotation_matrices[0]
                 identity_check = np.max(np.abs(matrix_0 - identity_matrix))
                 debug_log_file.write(f"[DEBUG] Max diff R(0) vs Identity: {identity_check:.4e}\n")
                 ortho_check_0 = np.max(np.abs(matrix_0 @ matrix_0.T - identity_matrix))
                 debug_log_file.write(f"[DEBUG] Orthogonality check R(0)@R(0).T vs I: {ortho_check_0:.4e}\n")
                 det_0 = np.linalg.det(matrix_0)
                 debug_log_file.write(f"[DEBUG] Determinant R(0): {det_0:.4f}\n")

                 if num_angles > 1:
                     idx_mid = num_angles // 2
                     matrix_mid = rotation_matrices[idx_mid]
                     ortho_check_mid = np.max(np.abs(matrix_mid @ matrix_mid.T - identity_matrix))
                     debug_log_file.write(f"[DEBUG] Ortho check R(mid): {ortho_check_mid:.4e}\n")
                     det_mid = np.linalg.det(matrix_mid)
                     debug_log_file.write(f"[DEBUG] Determinant R(mid): {det_mid:.4f}\n")
        # --- End Debug Checks ---

    except ImportError:
         if debug_log_file: debug_log_file.write("[DEBUG] SciPy not found. Matrix exponential method unavailable. Returning identities.\n")
         print("Warning: SciPy not installed. Cannot generate rotation matrices accurately. Please install scipy.")
         return np.array([identity_matrix] * num_angles)
    except Exception as e:
         if debug_log_file: debug_log_file.write(f"[DEBUG] Matrix exponential failed: {e}. Returning identity matrices.\n")
         print(f"Error during matrix exponentiation: {e}")
         return np.array([identity_matrix] * num_angles)


    if debug_log_file: debug_log_file.write("--- [DEBUG] generate_special_orthogonal_matrices END ---\n")
    return rotation_matrices


# --- Analysis Execution ---
def perform_srm_analysis(
    structured_data: list[dict],
    basis_details: dict,
    group_by: str | None,
    signed: bool,
    thresholds: list[float],
    num_angles: int = 72,
    ensemble_max_planes: int | None = None,
    ensemble_plane_selection: str = 'comb',
    debug_log_file=None
    ) -> dict[str, pd.DataFrame]: # MODIFIED Return Signature (Original Version)
    """
    Performs SRM analysis (single plane or ensemble) on grouped data.
    Returns: Dictionary mapping group names to pandas DataFrames with SRM results.
    """
    analysis_mode = basis_details.get('mode')
    if analysis_mode not in ['single_plane', 'ensemble']:
        print(f"Error: Invalid analysis mode '{analysis_mode}' in basis_details.")
        return {}

    print(f"\nGrouping vectors by: '{group_by if group_by else 'All Vectors'}'")
    grouped_vectors = defaultdict(list)
    if group_by:
        valid_vectors_in_grouping = 0; missing_key_count = 0
        for item in structured_data:
            # Get group value, converting Nones or numerics to string for consistent keys
            group_val_raw = item['key_components'].get(group_by)
            group_val = str(group_val_raw) if group_val_raw is not None else None

            if group_val is not None:
                grouped_vectors[group_val].append(item['vector'])
                valid_vectors_in_grouping += 1
            else: missing_key_count += 1
        if valid_vectors_in_grouping == 0:
            print(f"Warning: No vectors found with key '{group_by}'. Skipping analysis for this grouping.")
            return {}
        print(f"Found {len(grouped_vectors)} groups for key '{group_by}': {sorted(list(grouped_vectors.keys()))}")
        if missing_key_count > 0: print(f"  (Note: {missing_key_count} vectors lacked the '{group_by}' key)")
    else:
        # Special handling: If no grouping key, 'all' is the only group.
        group_name_for_all = BASELINE_GROUP_NAME_HEURISTIC # Use the constant
        grouped_vectors[group_name_for_all] = [item['vector'] for item in structured_data]
        if not grouped_vectors[group_name_for_all]: print("Error: No vectors loaded to analyze."); return {}
        print(f"Analyzing all {len(grouped_vectors[group_name_for_all])} vectors together (group: '{group_name_for_all}').")

    results_by_group = {}
    # Grey vector calculation is NOT done here in the original version
    angles_deg = np.linspace(0, 360, num_angles, endpoint=False)
    angles_rad = np.radians(angles_deg)

    if analysis_mode == 'single_plane':
        basis_vector_1 = basis_details.get('basis_1'); basis_vector_2 = basis_details.get('basis_2')
        rotation_mode = basis_details.get('rotation_mode', 'linear') # Default to linear if not specified
        if basis_vector_1 is None or basis_vector_2 is None: print("Error: basis_1 or basis_2 missing."); return {}

        # Normalize basis vectors ONCE
        try:
            norm_basis_1 = normalise(basis_vector_1.astype(np.float32)) # Ensure float32
            norm_basis_2 = normalise(basis_vector_2.astype(np.float32))
        except Exception as e:
            print(f"Error normalizing basis vectors: {e}"); return {}

        dot_prod = np.abs(np.dot(norm_basis_1, norm_basis_2));
        if dot_prod > 1.0 - EPSILON: print(f"Warning: Basis vectors nearly collinear (dot product: {dot_prod:.4f}). SRM results may be unstable.")

        print(f"Running Single Plane SRM (Rotation: {rotation_mode})...")
        group_iterator = tqdm(grouped_vectors.items(), desc=f"SRM for groups", leave=False, unit="group")
        for group_name, vector_list in group_iterator:
             group_iterator.set_postfix_str(f"Group: {group_name[:20]}...") # Show current group
             if not vector_list: print(f"Skipping empty group '{group_name}'."); continue
             try:
                  # Stack vectors and ensure float32
                  data_vectors = np.stack([np.asarray(v, dtype=np.float32) for v in vector_list], axis=0)
             except ValueError as e:
                  print(f"Skipping group '{group_name}': Error stacking vectors (likely inconsistent shapes) - {e}"); continue
             except Exception as e:
                  print(f"Skipping group '{group_name}': Unexpected error preparing vectors - {e}"); continue

             if data_vectors.ndim != 2 or data_vectors.shape[1] != DIMENSION: print(f"Skipping group '{group_name}': Invalid vector shape {data_vectors.shape}."); continue
             N, D = data_vectors.shape
             normalized_data_vectors = normalise(data_vectors, axis=1) # Normalize along vector dimension
             results_list = []

             if rotation_mode == 'linear':
                 # Precompute sines and cosines
                 cos_angles = np.cos(angles_rad)
                 sin_angles = np.sin(angles_rad)
                 # Calculate all spotlight vectors at once
                 spotlight_vectors = cos_angles[:, np.newaxis] * norm_basis_1 + sin_angles[:, np.newaxis] * norm_basis_2
                 # Normalize spotlight vectors (handle potential zero norms if basis vectors were identical)
                 norm_spotlights = np.linalg.norm(spotlight_vectors, axis=1, keepdims=True)
                 norm_spotlights[norm_spotlights == 0] = 1.0
                 normalized_spotlights = spotlight_vectors / norm_spotlights
                 # Calculate all similarities at once (N vectors x A angles)
                 all_similarities = normalized_data_vectors @ normalized_spotlights.T # Shape (N, A)

                 for i in range(num_angles):
                     similarities_at_angle = all_similarities[:, i]
                     angle_results = {"angle_deg": angles_deg[i], "mean_similarity": np.mean(similarities_at_angle)}
                     for thresh in thresholds:
                         positive_mask = similarities_at_angle >= thresh
                         positive_count = np.sum(positive_mask)
                         angle_results[f"count_thresh_{thresh}"] = positive_count
                         if signed:
                             negative_mask = similarities_at_angle <= -thresh
                             negative_count = np.sum(negative_mask)
                             angle_results[f"signed_count_thresh_{thresh}"] = positive_count - negative_count
                     results_list.append(angle_results)

             elif rotation_mode == 'matrix':
                 try:
                     rotation_generator = vectors_to_bivectors(norm_basis_1, norm_basis_2)
                     # Check if generator is zero (collinear vectors)
                     if np.linalg.norm(rotation_generator) < EPSILON: # Use constant
                          print(f"Info: Skipping matrix rotation for group '{group_name}' due to collinear basis vectors. Treating as no rotation.")
                          # Calculate similarity only with basis_1 (angle 0)
                          probe_vector = norm_basis_1
                          similarities_at_0 = normalized_data_vectors @ probe_vector
                          angle_results = {"angle_deg": 0.0, "mean_similarity": np.mean(similarities_at_0)}
                          for thresh in thresholds:
                              positive_count = np.sum(similarities_at_0 >= thresh); angle_results[f"count_thresh_{thresh}"] = positive_count
                              if signed: negative_count = np.sum(similarities_at_0 <= -thresh); angle_results[f"signed_count_thresh_{thresh}"] = positive_count - negative_count
                          # Fill results for other angles with the angle 0 result or NaNs? Let's use angle 0.
                          results_list = [angle_results.copy() for _ in angles_deg]
                          for i, ang_d in enumerate(angles_deg): results_list[i]["angle_deg"] = ang_d

                     else:
                         rotation_matrices = generate_special_orthogonal_matrices(rotation_generator, angles_rad, debug_log_file=debug_log_file)
                         if rotation_matrices.shape[0] != num_angles: # Check if generation failed
                              print(f"Error generating rotation matrices for group {group_name}. Skipping group.")
                              continue # Skip this group

                         probe_vector = norm_basis_1 # Probe direction is along the first basis vector at angle 0
                         # Apply rotations to probe: R_a * v_probe => rotated_probes[a] = R_a * v_probe
                         # einsum: 'aij,j->ai' where a=angle, i=row, j=col. Correct for R*v.
                         rotated_probes = np.einsum('aij,j->ai', rotation_matrices, probe_vector, optimize='optimal')

                         # Calculate similarities: data_vecs . rotated_probes
                         # einsum: 'bi,ai->ba' where b=batch(data), a=angle, i=dimension
                         all_similarities = np.einsum("bi,ai->ba", normalized_data_vectors, rotated_probes, optimize='optimal')
                         if debug_log_file: debug_log_file.write(f"[DEBUG] Matrix Mode Similarities shape for group {group_name}: {all_similarities.shape}\n")

                         for a in range(num_angles):
                             similarities_at_angle = all_similarities[:, a]
                             angle_results = {"angle_deg": angles_deg[a], "mean_similarity": np.mean(similarities_at_angle)}
                             for thresh in thresholds:
                                 positive_mask = similarities_at_angle >= thresh
                                 positive_count = np.sum(positive_mask)
                                 angle_results[f"count_thresh_{thresh}"] = positive_count
                                 if signed:
                                     negative_mask = similarities_at_angle <= -thresh
                                     negative_count = np.sum(negative_mask)
                                     angle_results[f"signed_count_thresh_{thresh}"] = positive_count - negative_count
                             results_list.append(angle_results)

                 except np.linalg.LinAlgError as e: print(f"\nError during matrix rotation for group {group_name} (LinAlgError): {e}. Skipping group."); results_list = None
                 except Exception as e: print(f"\nUnexpected error during matrix rotation for group {group_name}: {e}"); traceback.print_exc(); results_list = None

             if results_list: results_by_group[group_name] = pd.DataFrame(results_list)

    elif analysis_mode == 'ensemble':
        ensemble_basis = basis_details.get('ensemble_basis'); ensemble_labels = basis_details.get('ensemble_labels')
        if ensemble_basis is None: print("Error: ensemble_basis missing."); return {}
        m, d = ensemble_basis.shape
        if d != DIMENSION: print(f"Error: Ensemble basis dimension ({d}) != expected dimension ({DIMENSION})."); return {}
        if m < 2: print(f"Error: Ensemble basis must contain at least 2 vectors (found {m})."); return {}

        # Normalize ensemble basis ONCE
        try:
             ensemble_basis_norm = normalise(ensemble_basis.astype(np.float32), axis=1) # Ensure float32
        except Exception as e:
             print(f"Error normalizing ensemble basis vectors: {e}"); return {}


        basis_indices = list(range(m))
        if ensemble_plane_selection == 'comb': plane_index_pairs = list(itertools.combinations(basis_indices, 2))
        else: plane_index_pairs = list(itertools.permutations(basis_indices, 2)) # Default 'perm'
        num_total_planes = len(plane_index_pairs); print(f"Generated {num_total_planes} total plane index pairs using '{ensemble_plane_selection}'.")
        if ensemble_max_planes is not None and ensemble_max_planes > 0 and ensemble_max_planes < num_total_planes:
            print(f"Sampling {ensemble_max_planes} random planes from {num_total_planes}."); random.seed(42); plane_index_pairs = random.sample(plane_index_pairs, ensemble_max_planes)
        num_planes_to_run = len(plane_index_pairs); print(f"Analyzing {num_planes_to_run} planes.")

        if num_planes_to_run == 0:
            print("Error: No planes selected or generated for ensemble analysis."); return {}

        print("Running Ensemble SRM (Rotation: matrix)...")
        aggregated_results_by_group = {}
        group_iterator_outer = tqdm(grouped_vectors.items(), desc=f"Groups", leave=True, unit="group")

        for group_name, vector_list in group_iterator_outer:
            group_iterator_outer.set_postfix_str(f"Group: {group_name[:20]}...")
            if not vector_list: print(f"\nSkipping empty group '{group_name}'."); continue
            try:
                 data_vectors = np.stack([np.asarray(v, dtype=np.float32) for v in vector_list], axis=0)
            except ValueError as e: print(f"\nSkipping group '{group_name}': Error stacking vectors - {e}"); continue
            except Exception as e: print(f"\nSkipping group '{group_name}': Unexpected error preparing vectors - {e}"); continue

            if data_vectors.ndim != 2 or data_vectors.shape[1] != DIMENSION: print(f"\nSkipping group '{group_name}': Invalid vector shape {data_vectors.shape}."); continue
            normalized_data_vectors = normalise(data_vectors, axis=1); N = data_vectors.shape[0]

            # Store results per angle across all planes for this group
            # List of dictionaries, where each dict holds aggregated results for one angle
            aggregated_angle_results = defaultdict(lambda: {'sum_similarity': 0.0, 'count': 0})
            for thresh in thresholds:
                aggregated_angle_results[0.0][f'sum_count_{thresh}'] = 0.0 # Initialize sum columns using float angle 0.0
                if signed:
                     aggregated_angle_results[0.0][f'sum_signed_count_{thresh}'] = 0.0

            num_planes_processed_for_group = 0
            plane_iterator_inner = tqdm(plane_index_pairs, desc=f"Planes", leave=False, total=num_planes_to_run)
            for i, (idx1, idx2) in enumerate(plane_iterator_inner):
                norm_basis_1 = ensemble_basis_norm[idx1]; norm_basis_2 = ensemble_basis_norm[idx2]
                plane_id_str = f"{idx1}-{idx2}"
                if ensemble_labels is not None and idx1 < len(ensemble_labels) and idx2 < len(ensemble_labels):
                    try: plane_id_str = f"{sanitize_label(ensemble_labels[idx1])}-{sanitize_label(ensemble_labels[idx2])}" # Sanitize labels
                    except IndexError: pass
                plane_iterator_inner.set_postfix_str(f"Plane: {plane_id_str[:20]}...")

                try:
                    rotation_generator = vectors_to_bivectors(norm_basis_1, norm_basis_2)
                    # If vectors are collinear, skip this plane
                    if np.linalg.norm(rotation_generator) < EPSILON: # Use constant
                        if i < 5: print(f"Info: Skipping plane {plane_id_str} for group '{group_name}' due to collinear basis.")
                        continue

                    rotation_matrices = generate_special_orthogonal_matrices(rotation_generator, angles_rad, debug_log_file=debug_log_file)
                    if rotation_matrices.shape[0] != num_angles:
                         print(f"Error generating rotation matrices for group '{group_name}', plane {plane_id_str}. Skipping plane.")
                         continue

                    probe_vector = norm_basis_1
                    rotated_probes = np.einsum('aij,j->ai', rotation_matrices, probe_vector, optimize='optimal')
                    all_similarities = np.einsum("bi,ai->ba", normalized_data_vectors, rotated_probes, optimize='optimal') # Shape (N, A)

                    # Aggregate results for this plane into the group's aggregate
                    for a in range(num_angles):
                        angle_deg_key = angles_deg[a] # Use float angle as key
                        similarities_at_angle = all_similarities[:, a]

                        # Get the defaultdict for this angle
                        agg_data = aggregated_angle_results[angle_deg_key]
                        agg_data['sum_similarity'] += np.sum(similarities_at_angle) # Sum over N vectors
                        agg_data['count'] += N # Increment total vector count for this angle

                        for thresh in thresholds:
                            positive_mask = similarities_at_angle >= thresh
                            positive_count = np.sum(positive_mask)
                            agg_data[f'sum_count_{thresh}'] = agg_data.get(f'sum_count_{thresh}', 0.0) + positive_count
                            if signed:
                                negative_mask = similarities_at_angle <= -thresh
                                negative_count = np.sum(negative_mask)
                                agg_data[f'sum_signed_count_{thresh}'] = agg_data.get(f'sum_signed_count_{thresh}', 0.0) + (positive_count - negative_count)

                    num_planes_processed_for_group += 1

                except np.linalg.LinAlgError as e: print(f"\nLinAlgError for group '{group_name}' plane {plane_id_str}: {e}. Skipping plane.");
                except Exception as e: print(f"\nError for group '{group_name}' plane {plane_id_str}: {e}. Skipping plane."); traceback.print_exc();

            # --- Finalize aggregation for this group ---
            if num_planes_processed_for_group > 0:
                final_results_list = []
                sorted_angles = sorted(aggregated_angle_results.keys())
                for angle_deg_key in sorted_angles:
                    agg_data = aggregated_angle_results[angle_deg_key]
                    total_N_for_angle = agg_data['count']
                    if total_N_for_angle == 0: continue # Should not happen if planes were processed

                    mean_sim = agg_data['sum_similarity'] / total_N_for_angle
                    angle_output = {"angle_deg": angle_deg_key, "mean_similarity": mean_sim}
                    for thresh in thresholds:
                         mean_count = agg_data.get(f'sum_count_{thresh}', 0.0) / num_planes_processed_for_group
                         angle_output[f"count_thresh_{thresh}"] = mean_count
                         if signed:
                             mean_signed_count = agg_data.get(f'sum_signed_count_{thresh}', 0.0) / num_planes_processed_for_group
                             angle_output[f"signed_count_thresh_{thresh}"] = mean_signed_count
                    final_results_list.append(angle_output)

                if final_results_list:
                     aggregated_df = pd.DataFrame(final_results_list)
                     # Add count of planes averaged
                     aggregated_df['planes_averaged'] = num_planes_processed_for_group
                     aggregated_results_by_group[group_name] = aggregated_df
                else:
                      print(f"\nWarning: No results generated for group '{group_name}' after aggregation attempt.")
            else:
                print(f"\nWarning: SRM failed for ALL planes processed for group '{group_name}'. Skipping group.")

        results_by_group = aggregated_results_by_group

    # Return only the results_by_group dict in the original version
    return results_by_group


# === Plotting ===

def plot_srm_results_grouped(
    grouped_results_dfs: dict[str, pd.DataFrame],
    group_by_key: str | None,
    basis_id_str: str,
    analysis_mode: str,
    rotation_mode: str | None, # Relevant only for single_plane display
    signed_mode: bool,
    save_dir: Path,
    plot_threshold: float | None = None,
    self_srm_df: pd.DataFrame | None = None,
    # REMOVED grey_vector_summary argument
    ):
    """
    Generates and saves a plot for grouped SRM results.
    Handles both single plane and ensemble averaged results.
    (Grey vector plotting is NOT included in this original version).
    """
    if not grouped_results_dfs:
        print("Plotting Error: No grouped results provided to plot.")
        return

    fig, ax1 = plt.subplots(figsize=(14, 7))
    num_groups = len(grouped_results_dfs)
    colors = cm.get_cmap('tab20')(np.linspace(0, 1, num_groups)) if num_groups > 10 else cm.get_cmap('viridis')(np.linspace(0, 0.95, num_groups))

    group_by_title = str(group_by_key).replace('_',' ').title() if group_by_key else "All Vectors"
    mode_title = f"Mode: {rotation_mode.capitalize()}" if analysis_mode == 'single_plane' else "Mode: Ensemble (Matrix Avg)"
    signed_title_comp = " (Signed)" if signed_mode and plot_threshold is not None else ""
    ensemble_planes_comp = ""
    if analysis_mode == 'ensemble':
         # Try to get plane count from first df
         first_df = next(iter(grouped_results_dfs.values()), pd.DataFrame())
         planes_avg = first_df['planes_averaged'].iloc[0] if 'planes_averaged' in first_df.columns and not first_df.empty else 'N/A'
         ensemble_planes_comp = f" ({planes_avg} Planes Avg)"

    ax1.set_xlabel('Angle (degrees)')
    ax1.grid(True, axis='x', linestyle=':')

    ax2 = None # Secondary axis for mean similarity if counts are plotted
    count_lines = []
    count_col = None
    plot_type_title = "(Mean Similarity Only)" # Default title component

    if plot_threshold is not None:
        count_col_prefix = "signed_count" if signed_mode else "count"
        count_col = f"{count_col_prefix}_thresh_{plot_threshold}"
        col_exists = any(count_col in df.columns for df in grouped_results_dfs.values())

        if col_exists:
            ax2 = ax1.twinx() # Create secondary axis
            count_label = f"Signed Count (Thr: {plot_threshold})" if signed_mode else f"Count (Sim >= {plot_threshold})"
            if analysis_mode == 'ensemble': count_label = f"Avg {count_label}"
            count_legend_title = f"Threshold {plot_threshold}{signed_title_comp}"; plot_type_title = f"({count_label})"
            ax1.set_ylabel(count_label, color='tab:blue'); ax1.tick_params(axis='y', labelcolor='tab:blue'); ax1.grid(True, axis='y', linestyle=':', color='tab:blue', alpha=0.5)

            sorted_items = sorted(grouped_results_dfs.items(), key=lambda item: str(item[0]))
            for i, (group_name, df) in enumerate(sorted_items):
                if count_col in df.columns: line, = ax1.plot(df['angle_deg'], df[count_col], color=colors[i % len(colors)], marker='.', markersize=3, linestyle=':', label=f'{group_name} ({count_label.split(" ")[0]})'); count_lines.append(line)
            if count_lines: ax1.legend(handles=count_lines, loc='upper left', bbox_to_anchor=(0.0, 1.0), title=count_legend_title, fontsize='small')
        else: print(f"Warning: Count column '{count_col}' not found. Plotting mean similarity only."); plot_threshold = None; ax1.set_ylabel('Mean Cosine Similarity'); ax1.tick_params(axis='y', labelcolor='black'); ax1.grid(True, axis='y', linestyle=':', color='grey', alpha=0.7)
    else: ax1.set_ylabel('Mean Cosine Similarity'); ax1.tick_params(axis='y', labelcolor='black'); ax1.grid(True, axis='y', linestyle=':', color='grey', alpha=0.7)

    target_ax = ax2 if ax2 is not None else ax1
    mean_sim_label = 'Mean Cosine Similarity'; mean_sim_color = 'tab:red' if ax2 is not None else 'black'
    target_ax.set_ylabel(mean_sim_label, color=mean_sim_color); target_ax.tick_params(axis='y', labelcolor=mean_sim_color)
    if ax2 is not None: target_ax.grid(True, axis='y', linestyle='-.', color=mean_sim_color, alpha=0.5)

    mean_lines = []; sorted_items = sorted(grouped_results_dfs.items(), key=lambda item: str(item[0]))
    for i, (group_name, df) in enumerate(sorted_items):
         if 'mean_similarity' in df.columns: line, = target_ax.plot(df['angle_deg'], df['mean_similarity'], color=colors[i % len(colors)], marker=None, linestyle='-', linewidth=2, label=f'{group_name} (Mean Sim)'); mean_lines.append(line)

    # Grey vector marker is NOT plotted in this version
    # Self-SRM reference line IS plotted
    if analysis_mode == 'single_plane' and self_srm_df is not None and not self_srm_df.empty:
        if 'mean_similarity' in self_srm_df.columns and 'angle_deg' in self_srm_df.columns: line, = target_ax.plot(self_srm_df['angle_deg'], self_srm_df['mean_similarity'], color='black', linestyle='--', linewidth=1.5, label='Self-SRM Ref.'); mean_lines.append(line)

    if mean_lines:
        legend_title = f"Mean Similarity ({group_by_title})"; mean_legend_loc = 'lower right' if ax2 is not None else 'upper right'; mean_legend_anchor = (1.0, 0.0) if ax2 is not None else (1.0, 1.0)
        target_ax.legend(handles=mean_lines, loc=mean_legend_loc, bbox_to_anchor=mean_legend_anchor, title=legend_title, fontsize='small')

    fig.suptitle(f'Grouped SRM Sweep Analysis: {group_by_title}', fontsize=16, y=1.02)
    ax1.set_title(f'Basis ({basis_id_str}){ensemble_planes_comp}, {mode_title} {plot_type_title}', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); plt.xticks(np.arange(0, 361, 45))
    thresh_tag = f"thresh{plot_threshold}" if plot_threshold is not None else "meansim"; plot_base_filename = f"srm_plot_{thresh_tag}"; plot_filename = save_dir / (plot_base_filename + ".png")
    try:
        save_dir.mkdir(parents=True, exist_ok=True); plot_filename_str = str(plot_filename.resolve()); print(f"Attempting to save plot to: {plot_filename_str}")
        plt.savefig(plot_filename_str, bbox_inches='tight'); print(f"Saved grouped SRM plot: {plot_filename.name}")
    except Exception as e: print(f"Error saving plot {plot_filename.name}: {e}"); traceback.print_exc()
    finally: plt.close(fig)

# --- END OF FILE utils.py ---