# --- START OF FILE compare_srm_runs.py ---

import argparse
import datetime
import json
import sys
import traceback
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import re # Added import

# Assume utils.py is in the same directory or accessible via PYTHONPATH
try:
    import utils # Use updated utils
    # --- Define Constants using utils if available ---
    DEFAULT_EXPERIMENT_BASE_DIR = getattr(utils, 'DEFAULT_EXPERIMENT_BASE_DIR', "experiments")
    ANALYSES_SUBFOLDER_NAME = getattr(utils, 'ANALYSES_SUBFOLDER', "analyses")
    METADATA_SUBFOLDER_NAME = getattr(utils, 'METADATA_SUBFOLDER', "metadata")
    DATA_SUBFOLDER_NAME = getattr(utils, 'DATA_SUBFOLDER', "data") # Use for output data dir
    BASELINE_GROUP_NAME_HEURISTIC = getattr(utils, 'BASELINE_GROUP_NAME_HEURISTIC', "all")
    # Use analysis prefix constant if defined in utils, otherwise fallback
    try:
        ANALYSIS_PREFIX = getattr(utils, 'DEFAULT_ANALYSIS_LABEL_PREFIX', "srm_analysis")
    except AttributeError:
        ANALYSIS_PREFIX = "srm_analysis"
    # --- End Constants ---
except ImportError:
    print("Error: Could not import 'utils.py'. Make sure it's in the same directory or Python path.")
    # --- Define Fallback Constants ---
    DEFAULT_EXPERIMENT_BASE_DIR = "experiments"
    ANALYSES_SUBFOLDER_NAME = "analyses"
    METADATA_SUBFOLDER_NAME = "metadata"
    DATA_SUBFOLDER_NAME = "data"
    BASELINE_GROUP_NAME_HEURISTIC = "all" # Fallback definition
    ANALYSIS_PREFIX = "srm_analysis"
    # --- End Fallback Constants ---
    sys.exit(1)


# --- Constants ---
COMPARISON_SUBFOLDER = "comparisons"
# DATA_SUBFOLDER redefined locally below for clarity in output path construction
PLOTS_SUBFOLDER = "plots"
ANALYSIS_DATA_SUBFOLDER = DATA_SUBFOLDER_NAME # Input data subfolder uses name from utils/fallback


# --- Helper Functions ---

# --- MODIFIED find_srm_data_files function ---
def find_srm_data_files(analysis_data_dir: Path, grouping_key: str | None) -> dict[str, Path]:
    """
    Finds SRM data CSVs in an analysis data directory.
    Handles --group_by sweep by finding all matching files.
    Handles grouping_key 'all_vectors' or None.
    """
    srm_files = {}
    if not analysis_data_dir.is_dir():
        print(f"Warning: Analysis data directory not found: {analysis_data_dir}")
        return srm_files

    base_pattern = "srm_results_group_"
    baseline_heuristic = BASELINE_GROUP_NAME_HEURISTIC # e.g., 'all'

    # --- NEW LOGIC for 'sweep' ---
    if grouping_key == 'sweep':
        print(f"DEBUG: Searching for sweep group files in {analysis_data_dir}...")
        # Find all files matching the base pattern for groups
        pattern = f"{base_pattern}*.csv"
        files = list(analysis_data_dir.glob(pattern))
        if not files:
            print(f"Warning: No SRM result files found with pattern '{pattern}' in {analysis_data_dir}")
            return srm_files

        for f in files:
            try:
                # Extract group name after the last '_group_'
                # Handles names like 'baseline', '0', '10', '-10' etc.
                group_name = f.stem.split('_group_')[-1]
                # We store the path against the extracted group name
                srm_files[group_name] = f
                print(f"DEBUG:   Found sweep group file: {f.name} -> group '{group_name}'")
            except IndexError:
                print(f"Warning: Could not parse group name from filename: {f.name}")
        if not srm_files: # Double check if parsing failed for all
             print(f"Warning: Found files matching pattern, but failed to parse group names.")

    # --- Logic for specific group (non-sweep, non-all) ---
    elif grouping_key and grouping_key not in [baseline_heuristic, "all_vectors", None]:
        safe_group_name = utils.sanitize_label(grouping_key) # Sanitize the specific key
        pattern = f"{base_pattern}{safe_group_name}.csv"
        files = list(analysis_data_dir.glob(pattern))
        if files:
            srm_files[grouping_key] = files[0] # Use original group key
            if len(files) > 1: print(f"Warning: Multiple files for group '{grouping_key}' found. Using {files[0].name}")
        else:
             # Optional: Add fallback search here if needed
             print(f"Warning: No data file found for specific group '{grouping_key}' in {analysis_data_dir}")

    # --- Logic for 'all' group ---
    else: # Handle 'all' group request (grouping_key is None, 'all', or 'all_vectors')
        group_name_to_find = baseline_heuristic # Use the consistent heuristic name ('all')
        safe_group_name = utils.sanitize_label(group_name_to_find)
        pattern = f"{base_pattern}{safe_group_name}.csv"
        files = list(analysis_data_dir.glob(pattern))
        if files:
            srm_files[group_name_to_find] = files[0] # Use the standard heuristic name 'all' as the key
            if len(files) > 1: print(f"Warning: Multiple '{group_name_to_find}' files found. Using {files[0].name}")
        else:
             pattern_fallback = f"{base_pattern}*all*.csv" # Broader search
             files_fallback = list(analysis_data_dir.glob(pattern_fallback))
             if files_fallback:
                 srm_files[group_name_to_find] = files_fallback[0]
                 print(f"Warning: Found '{group_name_to_find}' file using fallback pattern: {files_fallback[0].name}")
             else:
                 print(f"Warning: No '{group_name_to_find}' data file found in {analysis_data_dir}")

    return srm_files
# --- END MODIFIED find_srm_data_files function ---


def plot_srm_deltas_grouped(
    delta_dfs_by_group: dict[str, pd.DataFrame],
    baseline_label: str, # Label for the baseline run (e.g., analysis dir name)
    intervention_label: str, # Label for the intervention run
    basis_id_str: str,
    plot_thresholds: list[float] | None, # User request (e.g., [0.5, 0.7])
    common_thresholds: list[float], # Actual common thresholds found
    signed_deltas_calculated: bool,
    save_dir: Path
    ):
    """Generates and saves plots for grouped SRM delta results."""
    # (Plotting function content remains the same as original)
    if not delta_dfs_by_group: print("Plotting Error: No delta results provided."); return

    groups = sorted(delta_dfs_by_group.keys()); num_groups = len(groups)
    try: # Define colors
        if num_groups > 20: colors = cm.get_cmap('turbo')(np.linspace(0, 1, num_groups))
        elif num_groups > 10: colors = cm.get_cmap('tab20')(np.linspace(0, 1, num_groups))
        else: colors = cm.get_cmap('viridis')(np.linspace(0, 0.95, max(num_groups, 2)))
    except ValueError: colors = cm.get_cmap('viridis')(np.linspace(0, 0.95, max(num_groups, 2)))

    # --- Plot Delta Mean Similarity ---
    fig_mean, ax_mean = plt.subplots(figsize=(14, 7)); lines_mean = []
    plot_title = f'SRM Mean Sim Change: {intervention_label} vs {baseline_label}\nBasis: {basis_id_str}'
    if baseline_label == 'all (as baseline)': plot_title = f'SRM Mean Sim Change from Baseline\nIntervention: {intervention_label}, Basis: {basis_id_str}'
    for i, group_name in enumerate(groups):
        df = delta_dfs_by_group[group_name]
        if 'delta_mean_similarity' in df.columns:
            line, = ax_mean.plot(df['angle_deg'], df['delta_mean_similarity'], color=colors[i % len(colors)], marker='.', markersize=3, linestyle='-', label=group_name); lines_mean.append(line)
    if not lines_mean: plt.close(fig_mean)
    else:
        ax_mean.axhline(0, color='grey', linestyle='--', linewidth=1); ax_mean.set_xlabel('Angle (degrees)')
        ax_mean.set_ylabel('Delta Mean Similarity (Intervention - Baseline)'); ax_mean.set_title(plot_title)
        ax_mean.legend(handles=lines_mean, loc='best', fontsize='small', title="Intervention Group (vs Baseline)")
        ax_mean.grid(True, axis='both', linestyle=':'); plt.xticks(np.arange(0, 361, 45)); fig_mean.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_mean_path = save_dir / 'delta_mean_sim_comparison.png'
        try: plt.savefig(plot_mean_path, bbox_inches='tight'); print(f"Saved delta mean plot: {plot_mean_path.name}")
        except Exception as e: print(f"Error saving delta mean plot: {e}")
        plt.close(fig_mean)

    # --- Plot Delta Counts ---
    thresholds_to_plot_final = []
    if plot_thresholds is not None:
        thresholds_to_plot_final = [t for t in plot_thresholds if t in common_thresholds]
        if len(thresholds_to_plot_final) < len(plot_thresholds): missing_req = set(plot_thresholds) - set(common_thresholds); print(f"Warning: Requested thresholds {missing_req} not common. Plotting only for {thresholds_to_plot_final}.")
        if not thresholds_to_plot_final: print("Info: No requested plot thresholds common.")
    for thresh in thresholds_to_plot_final:
        # Delta Count Plot
        delta_count_col = f'delta_count_thresh_{thresh}'; fig_count, ax_count = plt.subplots(figsize=(14, 7)); lines_count = []
        col_exists = any(delta_count_col in df.columns for df in delta_dfs_by_group.values())
        plot_title_count = f'SRM Count Change (Thr={thresh}): {intervention_label} vs {baseline_label}\nBasis: {basis_id_str}'
        if baseline_label == 'all (as baseline)': plot_title_count = f'SRM Count Change from Baseline (Thr={thresh})\nIntervention: {intervention_label}, Basis: {basis_id_str}'
        if col_exists:
            for i, group_name in enumerate(groups):
                 df = delta_dfs_by_group[group_name]
                 if delta_count_col in df.columns: line, = ax_count.plot(df['angle_deg'], df[delta_count_col], color=colors[i % len(colors)], marker='.', markersize=3, linestyle='-', label=group_name); lines_count.append(line)
            if lines_count:
                 ax_count.axhline(0, color='grey', linestyle='--', linewidth=1); ax_count.set_xlabel('Angle (degrees)')
                 ax_count.set_ylabel(f'Delta Count (Thr: {thresh})'); ax_count.set_title(plot_title_count)
                 ax_count.legend(handles=lines_count, loc='best', fontsize='small', title="Intervention Group (vs Baseline)"); ax_count.grid(True, axis='both', linestyle=':'); plt.xticks(np.arange(0, 361, 45)); fig_count.tight_layout(rect=[0, 0.03, 1, 0.95])
                 plot_count_path = save_dir / f'delta_count_thresh{thresh}_comparison.png'
                 try: plt.savefig(plot_count_path, bbox_inches='tight'); print(f"Saved delta count plot (Thr {thresh}): {plot_count_path.name}")
                 except Exception as e: print(f"Error saving delta count plot {plot_count_path.name}: {e}")
            else: print(f"Plotting Warning: Delta count column '{delta_count_col}' exists but no valid data found.")
        plt.close(fig_count)
        # Delta Signed Count Plot
        if signed_deltas_calculated:
            delta_signed_col = f'delta_signed_count_thresh_{thresh}'; fig_signed, ax_signed = plt.subplots(figsize=(14, 7)); lines_signed = []
            signed_col_exists = any(delta_signed_col in df.columns for df in delta_dfs_by_group.values())
            plot_title_signed = f'SRM Signed Count Change (Thr={thresh}): {intervention_label} vs {baseline_label}\nBasis: {basis_id_str}'
            if baseline_label == 'all (as baseline)': plot_title_signed = f'SRM Signed Count Change from Baseline (Thr={thresh})\nIntervention: {intervention_label}, Basis: {basis_id_str}'
            if signed_col_exists:
                for i, group_name in enumerate(groups):
                    df = delta_dfs_by_group[group_name]
                    if delta_signed_col in df.columns: line, = ax_signed.plot(df['angle_deg'], df[delta_signed_col], color=colors[i % len(colors)], marker='.', markersize=3, linestyle='-', label=group_name); lines_signed.append(line)
                if lines_signed:
                    ax_signed.axhline(0, color='grey', linestyle='--', linewidth=1); ax_signed.set_xlabel('Angle (degrees)')
                    ax_signed.set_ylabel(f'Delta Signed Count (Thr: {thresh})'); ax_signed.set_title(plot_title_signed)
                    ax_signed.legend(handles=lines_signed, loc='best', fontsize='small', title="Intervention Group (vs Baseline)"); ax_signed.grid(True, axis='both', linestyle=':'); plt.xticks(np.arange(0, 361, 45)); fig_signed.tight_layout(rect=[0, 0.03, 1, 0.95])
                    plot_signed_path = save_dir / f'delta_signed_count_thresh{thresh}_comparison.png'
                    try: plt.savefig(plot_signed_path, bbox_inches='tight'); print(f"Saved delta signed count plot (Thr {thresh}): {plot_signed_path.name}")
                    except Exception as e: print(f"Error saving delta signed plot {plot_signed_path.name}: {e}")
                else: print(f"Plotting Warning: Delta signed count column '{delta_signed_col}' exists but no valid data found.")
            plt.close(fig_signed)


# --- Helper: Function to Select Analysis Directory ---
def select_analysis_directory(base_dir: Path, prompt_text: str) -> Path | None:
    """Finds and interactively selects an SRM analysis directory."""
    print(f"\n{prompt_text}")
    print(f"Searching for analysis folders in '{base_dir}'...")
    analysis_dirs = []
    try: # Use the corrected search logic
        analyses_subdirs = list(base_dir.rglob(ANALYSES_SUBFOLDER_NAME))
        for analyses_path in analyses_subdirs:
            if analyses_path.is_dir() and analyses_path.name == ANALYSES_SUBFOLDER_NAME and analyses_path.parent.is_dir():
                for potential_dir in analyses_path.iterdir():
                    if potential_dir.is_dir(): analysis_dirs.append(potential_dir)
        analysis_dirs = sorted(analysis_dirs)
    except Exception as e: print(f"Error searching: {e}"); return None

    if not analysis_dirs: print(f"Error: No analysis directories found within {base_dir}."); return None

    selected_dir = None
    if len(analysis_dirs) == 1:
        selected_dir = analysis_dirs[0]; print(f"Auto-selected: {selected_dir.relative_to(base_dir)}")
    else:
        print("\nMultiple analysis directories found:")
        for i, dir_path in enumerate(analysis_dirs):
            run_dir_name = "[Unknown]"
            try: run_dir_name = dir_path.parent.parent.name
            except IndexError: pass
            print(f"  {i+1}: {dir_path.relative_to(base_dir)}  [Run: {run_dir_name}]")
        while selected_dir is None:
            try:
                choice = input(f"Enter the number (1-{len(analysis_dirs)}), or 0 to cancel: ")
                if choice == '0': print("Selection cancelled."); return None
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(analysis_dirs): selected_dir = analysis_dirs[choice_idx]; print(f"Selected: {selected_dir.relative_to(base_dir)}")
                else: print("Invalid choice.")
            except ValueError: print("Invalid input.")
            except (EOFError, KeyboardInterrupt): print("\nSelection cancelled."); return None
    return selected_dir.resolve()


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compares SRM analysis results between a baseline and an intervention run.")
    parser.add_argument("--experiment_base_dir", type=Path, default=Path(DEFAULT_EXPERIMENT_BASE_DIR), help="Base directory containing experiment run folders.")
    parser.add_argument("--baseline_dir", type=Path, default=None, help="[Optional] Path to the specific baseline srm_analysis_* directory.")
    parser.add_argument("--intervention_dir", type=Path, default=None, help="[Optional] Path to the specific intervention srm_analysis_* directory.")
    parser.add_argument("--output_dir", type=Path, default=None, help="Directory to save comparison results. If omitted, defaults to {intervention_dir}/comparisons/compare_{base_label}_vs_{int_label}/")
    parser.add_argument("--plot_thresholds", type=float, nargs='+', default=None, help="[Optional] List of specific epsilon thresholds for delta plots.")
    args = parser.parse_args()

    print("--- Starting SRM Comparison ---")
    base_dir = args.experiment_base_dir.resolve()
    if not base_dir.is_dir(): print(f"Error: Base directory not found: {base_dir}"); sys.exit(1)

    baseline_analysis_dir = args.baseline_dir
    if baseline_analysis_dir: baseline_analysis_dir = baseline_analysis_dir.resolve(); print(f"Using provided baseline dir: {baseline_analysis_dir.name}")
    else: baseline_analysis_dir = select_analysis_directory(base_dir, "Select the BASELINE analysis directory:");
    if not baseline_analysis_dir: sys.exit(1)

    intervention_analysis_dir = args.intervention_dir
    if intervention_analysis_dir: intervention_analysis_dir = intervention_analysis_dir.resolve(); print(f"Using provided intervention dir: {intervention_analysis_dir.name}")
    else: intervention_analysis_dir = select_analysis_directory(base_dir, "Select the INTERVENTION analysis directory:");
    if not intervention_analysis_dir: sys.exit(1)

    if baseline_analysis_dir == intervention_analysis_dir: print("Error: Baseline and Intervention dirs cannot be the same."); sys.exit(1)

    try: # Resolve again to ensure existence
        baseline_analysis_dir = baseline_analysis_dir.resolve(strict=True)
        intervention_analysis_dir = intervention_analysis_dir.resolve(strict=True)
    except FileNotFoundError as e: print(f"Error: Selected analysis dir not found: {e}."); sys.exit(1)
    except Exception as e: print(f"Error resolving dirs: {e}."); sys.exit(1)

    # Load Metadata
    meta_base_path = baseline_analysis_dir / METADATA_SUBFOLDER_NAME / "analysis_metadata.json"
    meta_int_path = intervention_analysis_dir / METADATA_SUBFOLDER_NAME / "analysis_metadata.json"
    print(f"Loading baseline metadata from: {meta_base_path.relative_to(base_dir)}")
    meta_base = utils.load_metadata(meta_base_path)
    print(f"Loading intervention metadata from: {meta_int_path.relative_to(base_dir)}")
    meta_int = utils.load_metadata(meta_int_path)
    if meta_base is None or meta_int is None: print("Error: Could not load metadata."); sys.exit(1)

    # Extract IDs
    base_capture_run_id = meta_base.get("source_capture_run_id"); int_capture_run_id = meta_int.get("source_capture_run_id")
    base_basis_run_id = meta_base.get("source_basis_run_id"); int_basis_run_id = meta_int.get("source_basis_run_id")
    print(f"Baseline Capture RunID: {base_capture_run_id or 'Unknown'}"); print(f"Intervention Capture RunID: {int_capture_run_id or 'Unknown'}")
    print(f"Baseline Basis Source RunID: {base_basis_run_id or 'NA'}"); print(f"Intervention Basis Source RunID: {int_basis_run_id or 'NA'}")

    # Determine Output Dir
    output_dir = args.output_dir
    base_analysis_label = utils.sanitize_label(meta_base.get("analysis_label", baseline_analysis_dir.name))
    int_analysis_label = utils.sanitize_label(meta_int.get("analysis_label", intervention_analysis_dir.name))
    comparison_label_auto = f"compare_{base_analysis_label}_vs_{int_analysis_label}"
    if output_dir is None: output_dir = intervention_analysis_dir / COMPARISON_SUBFOLDER / comparison_label_auto; print(f"Output dir defaulting to: {output_dir.relative_to(base_dir)}")
    else: output_dir = output_dir.resolve(); print(f"Using specified output dir: {output_dir}"); comparison_label_auto = output_dir.name

    # Use local constants for output subdirs
    output_data_dir = output_dir / "data"
    output_plots_dir = output_dir / "plots"
    output_meta_dir = output_dir / METADATA_SUBFOLDER_NAME # Use constant from top
    try: output_meta_dir.mkdir(parents=True, exist_ok=True); output_data_dir.mkdir(exist_ok=True); output_plots_dir.mkdir(exist_ok=True)
    except OSError as e: print(f"Error creating output dirs: {e}"); traceback.print_exc(); sys.exit(1)

    # Comparability Validation
    print("\n--- Validating Analysis Comparability ---")
    errors = []; warnings = []; validation_passed = True; special_comparison_mode = False
    basis_id_base = meta_base.get("basis_id_string", "UnknownBase"); basis_id_int = meta_int.get("basis_id_string", "UnknownInt")
    if basis_id_base != basis_id_int: errors.append(f"Basis ID mismatch: '{basis_id_base}' vs '{basis_id_int}'")
    if meta_base.get("num_angles") != meta_int.get("num_angles"): errors.append(f"Num angles mismatch")
    if meta_base.get("analysis_mode") != meta_int.get("analysis_mode"): errors.append(f"Analysis mode mismatch")
    if meta_base.get("dimension") != meta_int.get("dimension"): errors.append(f"Dimension mismatch")

    # Grouping Key Validation
    base_group_key = meta_base.get("grouping_key", BASELINE_GROUP_NAME_HEURISTIC)
    int_group_key = meta_int.get("grouping_key", BASELINE_GROUP_NAME_HEURISTIC)
    if base_group_key in [None, "all_vectors"]: base_group_key = BASELINE_GROUP_NAME_HEURISTIC
    if int_group_key in [None, "all_vectors"]: int_group_key = BASELINE_GROUP_NAME_HEURISTIC
    if base_group_key != int_group_key:
        if base_group_key == BASELINE_GROUP_NAME_HEURISTIC and int_group_key == 'sweep': warnings.append(f"Special Comparison Mode: Baseline '{base_group_key}' vs Intervention 'sweep'."); special_comparison_mode = True
        else: errors.append(f"Grouping key mismatch: '{base_group_key}' vs '{int_group_key}'.")

    if errors: print("Error: Analyses not comparable:"); [print(f"  - {err}") for err in errors]; sys.exit(1)

    common_thresholds = []; thresh_base = set(meta_base.get("tested_thresholds", [])); thresh_int = set(meta_int.get("tested_thresholds", []))
    if thresh_base != thresh_int:
        warnings.append(f"Thresholds differ. Base: {sorted(list(thresh_base))}, Int: {sorted(list(thresh_int))}")
        common_thresholds = sorted(list(thresh_base.intersection(thresh_int)))
        if not common_thresholds: errors.append("No common thresholds."); validation_passed = False
        else: warnings.append(f"-> Using common: {common_thresholds}")
    else: common_thresholds = sorted(list(thresh_base))

    signed_base = meta_base.get("signed_mode_enabled", False); signed_int = meta_int.get("signed_mode_enabled", False)
    calculate_signed_deltas = False
    if signed_base != signed_int: warnings.append(f"Signed mode mismatch. Delta signed counts NOT calculated.")
    elif signed_base and signed_int: calculate_signed_deltas = True; print("Info: Signed mode enabled. Delta signed counts calculated.")

    if not validation_passed: print("Error: Analyses not comparable:"); [print(f"  - {err}") for err in errors]; sys.exit(1)
    if warnings: print("Warnings:"); [print(f"  - {warn}") for warn in warnings]
    print("Validation passed. Proceeding.")

    # File Identification & Loading
    print("\n--- Locating and Loading SRM Data Files ---")
    baseline_data_dir = baseline_analysis_dir / ANALYSIS_DATA_SUBFOLDER
    intervention_data_dir = intervention_analysis_dir / ANALYSIS_DATA_SUBFOLDER

    baseline_group_to_load = BASELINE_GROUP_NAME_HEURISTIC if special_comparison_mode else base_group_key
    base_csvs = find_srm_data_files(baseline_data_dir, baseline_group_to_load)
    if baseline_group_to_load not in base_csvs: print(f"Error: Baseline '{baseline_group_to_load}' data file not found in {baseline_data_dir}."); sys.exit(1)
    baseline_data_path = base_csvs[baseline_group_to_load]
    print(f"Found baseline '{baseline_group_to_load}' data: {baseline_data_path.name}")

    # --- Load intervention data using corrected find_srm_data_files ---
    int_csvs = find_srm_data_files(intervention_data_dir, int_group_key) # Pass 'sweep' if that's the key
    if not int_csvs: print(f"Error: No intervention data files found for group key '{int_group_key}' in {intervention_data_dir}."); sys.exit(1)
    print(f"Found {len(int_csvs)} intervention group(s): {sorted(list(int_csvs.keys()))}") # Should now list 'baseline', '0', '10', '-10'

    # Data Alignment & Delta Calculation
    print("\n--- Calculating Deltas ---")
    delta_dfs = {}; output_csv_paths = {}
    try: df_base = pd.read_csv(baseline_data_path); print(f"  Loaded baseline '{baseline_group_to_load}' ({len(df_base)} rows).")
    except Exception as e: print(f"Error loading baseline {baseline_data_path.name}: {e}"); sys.exit(1)

    for group_name, int_csv_path in sorted(int_csvs.items()): # group_name will now be 'baseline', '0', etc.
        print(f"  Processing intervention group: {group_name}")
        try:
            df_int = pd.read_csv(int_csv_path)
            df_baseline_for_merge = df_base # Use the already loaded baseline df
            if not np.allclose(df_baseline_for_merge['angle_deg'], df_int['angle_deg']): print(f"    Error: Angle mismatch. Skipping."); continue

            merged_df = pd.merge(df_baseline_for_merge, df_int, on='angle_deg', how='inner', suffixes=('_base', '_int'))
            merged_df.sort_values(by='angle_deg', inplace=True)
            if len(merged_df) != meta_base.get("num_angles"): print(f"    Warning: Merged data length mismatch.")

            if 'mean_similarity_base' in merged_df.columns and 'mean_similarity_int' in merged_df.columns: merged_df['delta_mean_similarity'] = merged_df['mean_similarity_int'] - merged_df['mean_similarity_base']

            for t in common_thresholds:
                base_col=f'count_thresh_{t}_base';int_col=f'count_thresh_{t}_int';delta_col=f'delta_count_thresh_{t}'
                if base_col in merged_df.columns and int_col in merged_df.columns: merged_df[delta_col] = merged_df[int_col] - merged_df[base_col]
                if calculate_signed_deltas:
                    base_scol=f'signed_count_thresh_{t}_base';int_scol=f'signed_count_thresh_{t}_int';delta_scol=f'delta_signed_count_thresh_{t}'
                    if base_scol in merged_df.columns and int_scol in merged_df.columns: merged_df[delta_scol] = merged_df[int_scol] - merged_df[base_scol]

            delta_dfs[group_name] = merged_df
            output_csv_filename = f'delta_srm_group_{utils.sanitize_label(group_name)}_vs_{utils.sanitize_label(baseline_group_to_load)}.csv'
            output_csv_path = output_data_dir / output_csv_filename
            merged_df.to_csv(output_csv_path, index=False)
            output_csv_paths[group_name] = str(output_csv_path.relative_to(output_dir))

        except Exception as e: print(f"    Error processing group '{group_name}': {e}"); traceback.print_exc()

    if not delta_dfs: print("\nError: No delta DataFrames generated."); sys.exit(1)

    # Plotting
    print("\n--- Generating Delta Plots ---")
    try:
        plot_baseline_label = base_analysis_label if not special_comparison_mode else f"{baseline_group_to_load} (as baseline)"
        plot_srm_deltas_grouped(
            delta_dfs_by_group=delta_dfs, baseline_label=plot_baseline_label, intervention_label=int_analysis_label,
            basis_id_str=basis_id_base, plot_thresholds=args.plot_thresholds, common_thresholds=common_thresholds,
            signed_deltas_calculated=calculate_signed_deltas, save_dir=output_plots_dir )
    except Exception as e: print(f"Error plotting: {e}"); traceback.print_exc()
    generated_plots = [p.relative_to(output_dir).as_posix() for p in output_plots_dir.glob("*.png")]

    # Save Metadata
    print("\n--- Saving Comparison Metadata ---")
    comparison_metadata = {
        "comparison_timestamp": datetime.datetime.now().isoformat(), "comparison_script": Path(__file__).name,
        "input_baseline_analysis_directory_arg": str(args.baseline_dir) if args.baseline_dir else "Selected Interactively",
        "input_intervention_analysis_directory_arg": str(args.intervention_dir) if args.intervention_dir else "Selected Interactively",
        "baseline_analysis_directory_resolved": str(baseline_analysis_dir), "intervention_analysis_directory_resolved": str(intervention_analysis_dir),
        "baseline_capture_run_id": base_capture_run_id, "intervention_capture_run_id": int_capture_run_id,
        "baseline_basis_source_run_id": base_basis_run_id, "intervention_basis_source_run_id": int_basis_run_id,
        "comparison_output_directory": str(output_dir), "comparison_label": comparison_label_auto,
        "cli_args": vars(args),
        "validation": { "basis_id_compared": basis_id_base, "num_angles": meta_base.get("num_angles"), "analysis_mode": meta_base.get("analysis_mode"),
                       "baseline_grouping_key": base_group_key, "intervention_grouping_key": int_group_key,
                       "special_comparison_mode_all_vs_sweep": special_comparison_mode,
                       "common_thresholds": common_thresholds, "signed_deltas_calculated": calculate_signed_deltas, "warnings": warnings },
        "results": { "baseline_group_used": baseline_group_to_load, "intervention_groups_compared": sorted(list(delta_dfs.keys())),
                    "output_data_files_relative": output_csv_paths, "output_plot_files_relative": generated_plots }
    }
    meta_save_path = output_meta_dir / "comparison_metadata.json"
    if utils.save_json_metadata(meta_save_path, comparison_metadata): print(f"Comparison metadata saved to: {meta_save_path.relative_to(base_dir)}")
    else: print(f"Error saving comparison metadata.")

    # Console Summary
    print("\n--- Comparison Summary ---")
    print(f"Baseline Analysis Dir:     {baseline_analysis_dir.name}")
    print(f"Intervention Analysis Dir: {intervention_analysis_dir.name}")
    print(f"Baseline Capture Run ID:   {base_capture_run_id or 'Unknown'}"); print(f"Intervention Capture Run ID: {int_capture_run_id or 'Unknown'}")
    print(f"Basis Compared:            {comparison_metadata['validation']['basis_id_compared']}")
    print(f"Comparison Mode:           {'Special (All vs Sweep)' if special_comparison_mode else f'Standard ({base_group_key} vs {int_group_key})'}")
    print(f"Intervention Groups Compared: {comparison_metadata['results']['intervention_groups_compared']}")
    print(f"Output written to:         {output_dir.relative_to(base_dir)}"); print("-------------------------")
    print("\nComparison script finished successfully.")

# --- END OF FILE compare_srm_runs.py ---