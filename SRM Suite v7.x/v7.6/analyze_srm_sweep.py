# --- START OF FILE analyze_srm_sweep.py ---

import argparse
import traceback
import datetime
import re
import json
from pathlib import Path
import numpy as np
import pandas as pd
import sys # For exit

# Local application imports
import utils # Import our updated utility functions

# --- Constants ---
DEFAULT_EXPERIMENT_BASE_DIR = "experiments"
# Use analysis prefix constant from utils
DEFAULT_ANALYSIS_LABEL_PREFIX = getattr(utils, 'DEFAULT_ANALYSIS_LABEL_PREFIX', "srm_analysis")
VALID_ANALYSIS_MODES = ['single_plane', 'ensemble']
VALID_PLANE_SELECTIONS = ['comb', 'perm'] # For ensemble mode
VALID_GROUP_KEYS = utils.VALID_GROUP_KEYS # Use keys defined in utils
DEBUG_LOG_FILENAME = "srm_debug_log.txt"
# Get default onehot dir from utils if possible, otherwise use local default
DEFAULT_ONEHOT_OUTPUT_DIR = getattr(utils, 'DEFAULT_ONEHOT_OUTPUT_DIR', Path("./generated_basis"))

# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"Run grouped SRM sweep analysis. Selects a run directory (containing vectors), loads vectors, finds/loads basis vectors (interactively if needed), performs analysis (including Grey Vector if applicable), and saves results to a new analysis subfolder."
    )
    # --- Run Selection Args ---
    parser.add_argument("--experiment_base_dir", type=str, default=DEFAULT_EXPERIMENT_BASE_DIR, help="Base directory containing experiment run folders.")

    # --- Analysis Config Args ---
    parser.add_argument("--analysis_label", type=str, default=None, help="[Optional] Custom name for the analysis subfolder within 'analyses/'. If None, a default name with mode and timestamp is generated.")
    parser.add_argument("--enable_debug_logging", action='store_true', help="If set, enables detailed debug logging to a file within the analysis metadata folder.")

    # --- Basis Specification Args ---
    parser.add_argument("--basis_file", type=str, default=None, help=f"[Optional] Path to a specific basis file (.npz). Can be absolute or relative. If not specified or not found, interactive search is triggered.")
    parser.add_argument("--basis_run_directory", type=str, default=None, help=f"[Optional] Path to the run directory containing the '{utils.BASIS_SUBFOLDER}/' subfolder for the basis file. Prioritized if provided.")
    # Ensemble Specific (Only relevant if basis_file is ensemble type)
    parser.add_argument("--ensemble_basis_key", type=str, default='basis_vectors', help="[Ensemble] Key for basis array within the ensemble .npz file.")

    # --- SRM Execution Args ---
    parser.add_argument("--analysis_mode", type=str, choices=VALID_ANALYSIS_MODES, required=True, help="Analysis mode: 'single_plane' or 'ensemble'.")
    parser.add_argument("--rotation_mode", type=str, choices=['linear', 'matrix'], default='matrix', help="[Single Plane] SRM rotation mode (linear or matrix). Ignored for ensemble mode (always matrix). Default: matrix.")
    parser.add_argument("--thresholds", type=float, nargs='+', required=True, help="List of similarity thresholds (epsilon) for counting.")
    parser.add_argument("--num_angles", type=int, default=72, help="Number of angles for the SRM sweep (e.g., 72 for 5-degree steps).")
    parser.add_argument("--signed", action='store_true', help="Calculate signed resonance counts (positive counts - negative counts).")

    # Ensemble Specific Execution Args
    parser.add_argument("--plane_selection", type=str, choices=VALID_PLANE_SELECTIONS, default='comb', help="[Ensemble] Use combinations ('comb') or permutations ('perm') of basis vectors.")
    parser.add_argument("--max_planes", type=int, default=None, help="[Ensemble] Maximum number of planes to sample randomly. If None, use all generated planes.")

    # --- Grouping/Plotting/Saving Args ---
    parser.add_argument("--group_by", type=str, choices=utils.VALID_GROUP_KEYS, default=None, help="Metadata key from vector filename/key to group results by. If omitted, analyzes all vectors together (group name 'all').") # MODIFIED choices and help text
    parser.add_argument("--plot_threshold", type=float, default=None, help="Single epsilon threshold value for plotting count/signed_count data. If None, only mean similarity is plotted.")
    parser.add_argument("--plot_all_thresholds", action='store_true', help="Generate separate plots for count data at ALL calculated thresholds, plus mean similarity. Overrides --plot_threshold.")
    parser.add_argument("--save_csv", action='store_true', help="Save detailed SRM results to CSV files.")

    args = parser.parse_args()

    # --- Input Validation ---
    if args.analysis_mode == 'single_plane':
         if args.rotation_mode == 'linear': print("Info: Single plane analysis requested with linear rotation.")
         else: args.rotation_mode = 'matrix'; print("Info: Single plane analysis using matrix rotation.")
    elif args.analysis_mode == 'ensemble':
         if args.rotation_mode != 'matrix': print("Info: Ensemble analysis mode selected. Forcing matrix rotation."); args.rotation_mode = 'matrix'

    if args.plot_threshold is not None and args.plot_threshold not in args.thresholds:
        print(f"Warning: Specified --plot_threshold {args.plot_threshold} is not in the list of calculated --thresholds {args.thresholds}. Count plot for this specific threshold will not be generated unless --plot_all_thresholds is also set.")
    if args.plot_all_thresholds and args.plot_threshold is not None:
        print("Info: --plot_all_thresholds is set, ignoring specific --plot_threshold value.")

    # --- Select Experiment Run Directory ---
    base_dir = Path(args.experiment_base_dir).resolve()
    selected_run_dir = utils.select_experiment_folder(
        base_dir, prompt_text="Select the experiment run directory containing vectors to analyze:", pattern="*"
    )
    if not selected_run_dir: print("No run directory selected. Exiting."); sys.exit(1)
    print(f"Analyzing run directory: {selected_run_dir.name}")
    run_identifier = selected_run_dir.name

    # --- Locate and Load Input Vectors ---
    input_vector_path = utils.find_vector_file(selected_run_dir)
    if not input_vector_path:
        found_simple = False
        for simple_name in ["baseline_vectors.npz", "intervened_vectors.npz"]:
            potential_path = selected_run_dir / utils.CAPTURE_SUBFOLDER / utils.VECTORS_SUBFOLDER / simple_name
            if potential_path.is_file():
                 print(f"Info: Auto-selected vector file with simplified name: {potential_path.name}")
                 input_vector_path = potential_path; found_simple = True; break
        if not found_simple:
            print(f"Error: Could not find required vector input file in {selected_run_dir / utils.CAPTURE_SUBFOLDER / utils.VECTORS_SUBFOLDER}. Exiting."); sys.exit(1)

    print(f"Using Input Vectors: {input_vector_path.relative_to(selected_run_dir)}")
    structured_data, source_capture_metadata = utils.load_vector_data(input_vector_path, expected_dim=utils.DIMENSION)

    capture_run_id = None
    if structured_data is None: print("Failed to load vectors. Cannot proceed."); sys.exit(1)
    elif not structured_data: print("Warning: No valid vectors found in the loaded file. Exiting."); sys.exit(0)
    else:
         capture_run_id = source_capture_metadata.get('run_id')
         if capture_run_id: print(f"Source capture RunID confirmed from vector metadata: {capture_run_id}")
         else:
             match = re.match(r"([A-Z]+-\d{3})_", selected_run_dir.name)
             if match: capture_run_id = match.group(1); print(f"Warning: 'run_id' missing in vector metadata. Parsed from directory name: {capture_run_id}")
             else: print("Warning: Could not determine source capture 'run_id' from metadata or directory name.")
    print(f"Total vectors loaded: {len(structured_data)}")


    # --- Resolve Basis File Path ---
    print("\nResolving basis file path...")
    basis_path = None; basis_source_info = "Not resolved"; basis_run_dir_used = None
    if args.basis_file:
        basis_file_p = Path(args.basis_file)
        try:
            resolved_direct = basis_file_p.resolve(strict=True)
            basis_path = resolved_direct; basis_source_info = f"direct_path_arg (--basis_file): {basis_path}"; print(f"Found basis file via direct path: {basis_path}")
        except FileNotFoundError:
             if args.basis_run_directory:
                 basis_run_p = Path(args.basis_run_directory).resolve()
                 if basis_run_p.is_dir():
                     basis_search_dir = basis_run_p / utils.BASIS_SUBFOLDER
                     basis_path_rel = utils.find_basis_file(basis_search_dir, specific_filename=basis_file_p.name)
                     if basis_path_rel: basis_path = basis_path_rel; basis_source_info = f"relative_in_basis_run_dir"; print(f"Found basis file via relative path in specified basis run dir: {basis_path}"); basis_run_dir_used = basis_run_p
                     else: print(f"Info: Basis file '{args.basis_file}' not found in specified --basis_run_directory's '{utils.BASIS_SUBFOLDER}' folder.")
                 else: print(f"Warning: Specified --basis_run_directory not found: {args.basis_run_directory}")
             else: print(f"Info: Specified --basis_file '{args.basis_file}' not found directly or relative to CWD.")
        except Exception as e: print(f"Warning: Error resolving --basis_file '{args.basis_file}': {e}")

    if basis_path is None and args.basis_run_directory:
        basis_run_p = Path(args.basis_run_directory).resolve()
        if basis_run_p.is_dir():
             print(f"Attempting to find basis file automatically in specified --basis_run_directory: {basis_run_p.name}")
             basis_search_dir = basis_run_p / utils.BASIS_SUBFOLDER
             basis_path_found = utils.find_basis_file(basis_search_dir)
             if basis_path_found: basis_path = basis_path_found; basis_source_info = f"auto_search_in_basis_run_dir"; basis_run_dir_used = basis_run_p
             else: print(f"Info: No basis files found in specified --basis_run_directory's '{utils.BASIS_SUBFOLDER}' folder.")
        else: print(f"Warning: Specified --basis_run_directory not found: {args.basis_run_directory}")

    if basis_path is None:
        print(f"No basis specified or found via arguments. Checking locally in '{selected_run_dir.name}/{utils.BASIS_SUBFOLDER}'...")
        local_basis_search_dir = selected_run_dir / utils.BASIS_SUBFOLDER
        basis_path_local = utils.find_basis_file(local_basis_search_dir)
        if basis_path_local: basis_path = basis_path_local; basis_source_info = "auto_search_local"; basis_run_dir_used = selected_run_dir
        else: print(f"Info: No basis files found locally.")

    if basis_path is None:
        print("\nBasis file not found via args or locally. Attempting interactive selection...");
        selected_baseline_run = utils.select_experiment_folder(base_dir, prompt_text="Select the RUN directory containing the required basis file:", pattern="*")
        if selected_baseline_run:
            baseline_basis_search_dir = selected_baseline_run / utils.BASIS_SUBFOLDER
            print(f"Searching for basis files in selected run: '{baseline_basis_search_dir.relative_to(base_dir)}'")
            basis_path_interactive = utils.find_basis_file(baseline_basis_search_dir)
            if basis_path_interactive: basis_path = basis_path_interactive; basis_source_info = f"auto_search_interactive ({selected_baseline_run.name})"; basis_run_dir_used = selected_baseline_run
            else: print(f"Info: No basis files found in the interactively selected run's basis folder.")
        else: print("Interactive selection of basis source run cancelled.")

    if not basis_path or not basis_path.is_file(): print("\nError: Could not find or select a suitable basis file (.npz). Exiting."); sys.exit(1)

    basis_path = basis_path.resolve()
    print(f"\nUsing basis file: {basis_path}")
    if basis_run_dir_used: print(f"(Basis sourced from run: {basis_run_dir_used.name})")
    else:
        potential_parent_dir = basis_path.parent
        if potential_parent_dir.name == utils.BASIS_SUBFOLDER: basis_run_dir_used = potential_parent_dir.parent; print(f"(Inferred basis source run: {basis_run_dir_used.name})")
        elif potential_parent_dir.name == DEFAULT_ONEHOT_OUTPUT_DIR.name: print("(Basis sourced from generated_basis directory)")
        else: print("(Basis source run directory could not be automatically inferred)")


    # --- Load Basis Metadata ---
    basis_metadata_path = basis_path.with_suffix('.json')
    basis_source_run_id = None; basis_source_metadata = None
    if basis_metadata_path.is_file():
        print(f"Attempting to load basis metadata from: {basis_metadata_path.name}")
        basis_source_metadata = utils.load_metadata(basis_metadata_path)
        if basis_source_metadata:
            if "source_run_id" in basis_source_metadata: basis_source_run_id = basis_source_metadata.get('source_run_id'); print(f"  Basis source RunID: {basis_source_run_id}" if basis_source_run_id else "  Warning: 'source_run_id' key present but empty.")
            elif basis_source_metadata.get("basis_generation_mode") == 'onehot': print("  Loaded onehot basis metadata.")
            else: print("  Warning: Loaded basis metadata, but missing 'source_run_id' and not 'onehot' mode.")
        else: print(f"  Warning: Failed to load or parse basis metadata file: {basis_metadata_path.name}")
    else: print(f"Info: Basis metadata file (.json) not found at {basis_metadata_path.name}. Basis source RunID will be unknown.")


    # --- Load Basis Vectors ---
    print(f"Loading basis vectors from: {basis_path.name}")
    basis_details = {'mode': args.analysis_mode, 'source_path': str(basis_path)}
    basis_id_str = f"{basis_path.stem}"
    try:
        basis_data = np.load(basis_path, allow_pickle=True)
        if args.analysis_mode == 'single_plane':
            if 'basis_1' not in basis_data or 'basis_2' not in basis_data: parser.error(f"Keys 'basis_1', 'basis_2' missing in basis file '{basis_path}' for single_plane mode.")
            basis_1 = basis_data['basis_1']; basis_2 = basis_data['basis_2']
            if basis_1.shape != (utils.DIMENSION,) or basis_2.shape != (utils.DIMENSION,): parser.error(f"Basis vector shape mismatch. Expected ({utils.DIMENSION},).")
            basis_details['basis_1'] = basis_1; basis_details['basis_2'] = basis_2
            basis_details['rotation_mode'] = args.rotation_mode
            if basis_source_metadata and basis_source_metadata.get("generated_filename_label"): basis_id_str = basis_source_metadata["generated_filename_label"]
            else: basis_id_str = f"SinglePlane_{basis_path.stem}"
            print(f"Loaded single plane basis (basis_1, basis_2)")
        elif args.analysis_mode == 'ensemble':
            ensemble_key = args.ensemble_basis_key
            if ensemble_key not in basis_data: parser.error(f"Ensemble key '{ensemble_key}' not found in {basis_path}.")
            ensemble_basis = basis_data[ensemble_key]
            if ensemble_basis.ndim != 2: parser.error(f"Ensemble basis must be 2D (M, D). Got shape {ensemble_basis.shape}.")
            m, d = ensemble_basis.shape
            if d != utils.DIMENSION: parser.error(f"Ensemble basis dimension mismatch. Got {d}, expected {utils.DIMENSION}.")
            if m < 2: parser.error(f"Ensemble basis needs at least 2 vectors, found {m}.")
            basis_details['ensemble_basis'] = ensemble_basis; basis_details['ensemble_labels'] = None
            if 'group_labels' in basis_data:
                labels = basis_data['group_labels']
                if len(labels) == m: basis_details['ensemble_labels'] = labels; print(f"Loaded ensemble basis (key: '{ensemble_key}', shape: {ensemble_basis.shape}) with {m} labels.")
                else: print(f"Warning: Label count ({len(labels)}) != vector count ({m}). Ignoring labels."); print(f"Loaded ensemble basis (key: '{ensemble_key}', shape: {ensemble_basis.shape}) WITHOUT matching labels.")
            else: print(f"Loaded ensemble basis (key: '{ensemble_key}', shape: {ensemble_basis.shape}). No labels found.")
            if basis_source_metadata and basis_source_metadata.get("generated_filename_label"): basis_id_str = basis_source_metadata["generated_filename_label"]
            else: basis_id_str = f"Ensemble_{basis_path.stem}_{m}vec"
    except Exception as e: print(f"Error loading basis file {basis_path}: {e}"); traceback.print_exc(); sys.exit(1)

    # --- Define Analysis Output Structure ---
    analysis_timestamp = datetime.datetime.now(); analysis_timestamp_str = analysis_timestamp.strftime("%Y%m%d_%H%M%S")
    if args.analysis_label: analysis_folder_name = utils.sanitize_label(args.analysis_label); print(f"Using provided analysis label: '{analysis_folder_name}'")
    else:
        group_tag = f"by_{args.group_by}" if args.group_by else utils.BASELINE_GROUP_NAME_HEURISTIC # Use constant
        basis_tag_sanitized = utils.sanitize_label(basis_id_str)
        analysis_folder_name = f"{DEFAULT_ANALYSIS_LABEL_PREFIX}_{args.analysis_mode}_{group_tag}_basis_{basis_tag_sanitized}_{analysis_timestamp_str}"
        print(f"Using generated analysis label: '{analysis_folder_name}'")

    analyses_base_path = selected_run_dir / utils.ANALYSES_SUBFOLDER
    analysis_path = analyses_base_path / analysis_folder_name
    analysis_plot_dir = analysis_path / utils.PLOTS_SUBFOLDER
    analysis_data_dir = analysis_path / utils.DATA_SUBFOLDER
    analysis_metadata_dir = analysis_path / utils.METADATA_SUBFOLDER
    try:
        analysis_metadata_dir.mkdir(parents=True, exist_ok=True)
        analysis_plot_dir.mkdir(parents=True, exist_ok=True)
        analysis_data_dir.mkdir(parents=True, exist_ok=True)
        print(f"Analysis outputs will be saved in: {analysis_path.relative_to(base_dir)}")
    except OSError as e: print(f"Error creating analysis output directories in '{analysis_path}': {e}"); traceback.print_exc(); sys.exit(1)

    # --- Setup Debug Logging ---
    debug_log_file = None; debug_log_path = None; debug_log_file_rel_path = None
    if args.enable_debug_logging:
        debug_log_path = analysis_metadata_dir / DEBUG_LOG_FILENAME
        try:
            debug_log_file = open(debug_log_path, 'w', encoding='utf-8')
            debug_log_file_rel_path = str(debug_log_path.relative_to(analysis_path))
            print(f"Debug logging enabled. Writing details to: {debug_log_path}")
            debug_log_file.write(f"--- SRM Analysis Debug Log ---\nRun: {analysis_folder_name}\nTimestamp: {analysis_timestamp.isoformat()}\n")
            debug_log_file.write(f"Args: {vars(args)}\nSource Run: {selected_run_dir}\nVectors: {input_vector_path}\nBasis: {basis_path}\n{'-'*30}\n\n"); debug_log_file.flush()
        except Exception as e: print(f"Warning: Could not open debug log file '{debug_log_path}': {e}. Debug logging disabled."); debug_log_file = None; debug_log_file_rel_path = None

    # --- Prepare Analysis Metadata ---
    all_run_metadata = {
        "script_name": Path(__file__).name, "analysis_timestamp": analysis_timestamp.isoformat(), "analysis_label": analysis_folder_name,
        "source_capture_run_id": capture_run_id, "source_capture_run_directory": str(selected_run_dir), "source_capture_run_directory_name": selected_run_dir.name,
        "source_capture_vector_file_relative": str(input_vector_path.relative_to(selected_run_dir)), "source_capture_metadata": source_capture_metadata,
        "basis_source_description": basis_source_info, "basis_file_path_resolved": str(basis_path),
        "basis_file_relative_to_analyzed_run": str(basis_path.relative_to(selected_run_dir)) if basis_path.is_relative_to(selected_run_dir) else "external_or_other_run",
        "basis_file_source_run_dir": str(basis_run_dir_used) if basis_run_dir_used else None,
        "basis_id_string": basis_id_str, "source_basis_run_id": basis_source_run_id, "source_basis_metadata": basis_source_metadata,
        "analysis_mode": args.analysis_mode, "dimension": utils.DIMENSION, "num_angles": args.num_angles, "signed_mode_enabled": args.signed,
        "tested_thresholds": sorted(list(set(args.thresholds))), "grouping_key": args.group_by if args.group_by else utils.BASELINE_GROUP_NAME_HEURISTIC, # Reflect 'all' if None
        "single_plane_params": None, "ensemble_params": None, "grey_vector_summary": None, # Initialize grey vector field
        "output_analysis_directory_relative": str(analysis_path.relative_to(selected_run_dir)),
        "output_plot_dir_relative": str(analysis_plot_dir.relative_to(analysis_path)), "output_data_dir_relative": str(analysis_data_dir.relative_to(analysis_path)),
        "output_metadata_dir_relative": str(analysis_metadata_dir.relative_to(analysis_path)),
        "debug_logging_enabled": args.enable_debug_logging, "debug_log_file_relative": debug_log_file_rel_path, "cli_args": vars(args),
        "analysis_results_summary": {},
    }
    if args.analysis_mode == 'single_plane': all_run_metadata["single_plane_params"] = {"rotation_mode": args.rotation_mode}
    else: all_run_metadata["ensemble_params"] = {"ensemble_basis_key": args.ensemble_basis_key, "plane_selection_method": args.plane_selection, "max_planes_setting": args.max_planes, "num_planes_analyzed": None}

    # --- Calculate Self-SRM (Optional Reference for Single Plane) ---
    self_srm_df = None
    if args.analysis_mode == 'single_plane':
        print("\nCalculating Self-SRM reference curve...")
        try:
            self_srm_structured_data = [{'key': 'self', 'key_components': {}, 'vector': basis_details['basis_1']}]
            self_srm_results, _ = utils.perform_srm_analysis(
                structured_data=self_srm_structured_data, basis_details=basis_details, group_by=None,
                signed=False, thresholds=[], num_angles=args.num_angles, debug_log_file=debug_log_file
            )
            baseline_key = utils.BASELINE_GROUP_NAME_HEURISTIC
            if self_srm_results and baseline_key in self_srm_results:
                 self_srm_df = self_srm_results[baseline_key]
                 print("Self-SRM reference calculated.")
                 if debug_log_file: debug_log_file.write("[INFO] Self-SRM reference calculated.\n")
            else: print("Warning: Failed to calculate Self-SRM reference line.");
            if debug_log_file and not self_srm_df: debug_log_file.write("[WARNING] Failed to calculate Self-SRM reference line.\n")
        except Exception as e: print(f"Warning: Error calculating Self-SRM reference: {e}")
        if debug_log_file: debug_log_file.write(f"[ERROR] Calculating Self-SRM reference: {e}\n{traceback.format_exc()}\n")


    # --- Execute SRM Analysis ---
    analysis_results_by_group = {}
    print(f"\nStarting SRM Analysis Run...")
    try:
        analysis_results_by_group, calculated_grey_vector_summary = utils.perform_srm_analysis(
            structured_data=structured_data,
            basis_details=basis_details,
            group_by=args.group_by,
            signed=args.signed,
            thresholds=args.thresholds,
            num_angles=args.num_angles,
            ensemble_max_planes=args.max_planes if args.analysis_mode == 'ensemble' else None,
            ensemble_plane_selection=args.plane_selection if args.analysis_mode == 'ensemble' else None,
            debug_log_file=debug_log_file
        )
        all_run_metadata["grey_vector_summary"] = calculated_grey_vector_summary

        if not analysis_results_by_group:
             print("\nWarning: SRM analysis returned no results. Check logs and input data.")
             if debug_log_file: debug_log_file.write("[WARNING] SRM analysis returned no results.\n")
             all_run_metadata["analysis_results_summary"]["status"] = "Completed (No Results)"
        else:
             print(f"\nSRM analysis completed for {len(analysis_results_by_group)} group(s).")
             if debug_log_file: debug_log_file.write(f"[INFO] SRM analysis completed for {len(analysis_results_by_group)} group(s).\n")
             all_run_metadata["analysis_results_summary"]["groups_analyzed"] = sorted(list(analysis_results_by_group.keys()))
             all_run_metadata["analysis_results_summary"]["num_groups_successful"] = len(analysis_results_by_group)
             if args.analysis_mode == 'ensemble' and analysis_results_by_group:
                  first_group_df = next(iter(analysis_results_by_group.values()), pd.DataFrame())
                  if 'planes_averaged' in first_group_df.columns and not first_group_df.empty:
                      try:
                          num_planes = int(first_group_df['planes_averaged'].iloc[0])
                          all_run_metadata["ensemble_params"]["num_planes_analyzed"] = num_planes
                          print(f"  (Ensemble analysis averaged over {num_planes} planes per group)")
                          if debug_log_file: debug_log_file.write(f"[INFO] Ensemble analysis averaged over {num_planes} planes per group.\n")
                      except (ValueError, TypeError): print("  Warning: Could not parse 'planes_averaged' value.");
                      if debug_log_file: debug_log_file.write("[WARNING] Could not parse 'planes_averaged' value.\n")
                  else: print("  Info: 'planes_averaged' column not found.");
                  if debug_log_file: debug_log_file.write("[INFO] 'planes_averaged' column not found.\n")

    except Exception as e:
        print("\n--- FATAL ERROR during SRM analysis execution ---"); traceback.print_exc()
        if debug_log_file: debug_log_file.write("\n--- FATAL ERROR during SRM analysis execution ---\n"); traceback.print_exc(file=debug_log_file); debug_log_file.flush()
        all_run_metadata["analysis_results_summary"]["status"] = "Failed"; all_run_metadata["analysis_results_summary"]["error"] = str(e)
        metadata_path_fail = analysis_metadata_dir / "analysis_metadata_FAILED.json"
        utils.save_json_metadata(metadata_path_fail, all_run_metadata)
        print(f"Attempted to save failure metadata to {metadata_path_fail}")
        if debug_log_file: debug_log_file.close()
        sys.exit(1)

    # --- Save CSV Data ---
    saved_csv_files = {}
    if args.save_csv and analysis_results_by_group:
        print("\nSaving detailed SRM results to CSV...")
        csv_base_filename = f"srm_results"
        for group_name, df in analysis_results_by_group.items():
            safe_group_name = utils.sanitize_label(group_name)
            csv_filename = f"{csv_base_filename}_group_{safe_group_name}.csv"
            csv_path = analysis_data_dir / csv_filename
            try:
                df.to_csv(csv_path, index=False)
                print(f"  Saved data for group '{group_name}' to: {csv_path.name}")
                saved_csv_files[group_name] = str(csv_path.relative_to(analysis_path))
            except Exception as e: print(f"Error saving data for group {group_name} to {csv_path.name}: {e}");
            if debug_log_file: debug_log_file.write(f"[ERROR] Saving CSV for group {group_name}: {e}\n")
        all_run_metadata["analysis_results_summary"]["output_data_files_relative"] = saved_csv_files
    elif not analysis_results_by_group: print("\nSkipping CSV saving: No analysis results were generated.")
    elif not args.save_csv: print("\nSkipping CSV saving: --save_csv not specified.")


    # --- Generate Plots ---
    generated_plot_files = []
    if analysis_results_by_group:
        print("\nGenerating plots...")
        plot_configs_run = []
        thresholds_to_plot_for_counts = []
        if args.plot_all_thresholds: thresholds_to_plot_for_counts = sorted(list(set(args.thresholds)))
        elif args.plot_threshold is not None and args.plot_threshold in args.thresholds: thresholds_to_plot_for_counts = [args.plot_threshold]
        plot_requests = thresholds_to_plot_for_counts + [None] # None signifies mean sim plot
        print(f"Requested plot thresholds (None = Mean Sim): {plot_requests}")

        for current_plot_thresh in plot_requests:
             is_redundant_mean_plot = (current_plot_thresh is None and
                                       len(analysis_results_by_group) == 1 and
                                       utils.BASELINE_GROUP_NAME_HEURISTIC in analysis_results_by_group and # Check specific key
                                       (self_srm_df is None or self_srm_df.empty))
             if is_redundant_mean_plot:
                 print(f"  Skipping mean similarity only plot for single '{utils.BASELINE_GROUP_NAME_HEURISTIC}' group as it's redundant without comparison or SelfSRM.")
                 continue

             try:
                 utils.plot_srm_results_grouped(
                     grouped_results_dfs=analysis_results_by_group,
                     group_by_key=args.group_by,
                     basis_id_str=basis_id_str,
                     analysis_mode=args.analysis_mode,
                     rotation_mode=args.rotation_mode if args.analysis_mode == 'single_plane' else None,
                     signed_mode=args.signed,
                     save_dir=analysis_plot_dir,
                     plot_threshold=current_plot_thresh,
                     self_srm_df=self_srm_df if args.analysis_mode == 'single_plane' else None,
                     grey_vector_summary=all_run_metadata["grey_vector_summary"] # Pass the summary here
                 )
                 thresh_tag = f"thresh{current_plot_thresh}" if current_plot_thresh is not None else "meansim"
                 plot_base_filename = f"srm_plot_{thresh_tag}"
                 plot_rel_path = Path(utils.PLOTS_SUBFOLDER) / f"{plot_base_filename}.png"
                 generated_plot_files.append(str(plot_rel_path))
                 plot_configs_run.append({"threshold": current_plot_thresh, "filename_relative": str(plot_rel_path)})

             except Exception as plot_e: print(f"Error generating plot for threshold {current_plot_thresh}: {plot_e}"); traceback.print_exc();
             if debug_log_file: debug_log_file.write(f"[ERROR] Generating plot for threshold {current_plot_thresh}: {plot_e}\n{traceback.format_exc()}\n")

        all_run_metadata["analysis_results_summary"]["output_plot_files_relative"] = generated_plot_files
        all_run_metadata["analysis_results_summary"]["plot_configurations"] = plot_configs_run
    else: print("\nSkipping plotting: No analysis results were generated.")

    # --- Finalize and Save Metadata ---
    print(f"\nSaving final analysis metadata...")
    analysis_metadata_file_path = analysis_metadata_dir / "analysis_metadata.json"
    if all_run_metadata["analysis_results_summary"].get("status") != "Failed":
         all_run_metadata["analysis_results_summary"]["status"] = "Completed"

    if utils.save_json_metadata(analysis_metadata_file_path, all_run_metadata):
        print(f"Analysis metadata saved successfully to: {analysis_metadata_file_path.relative_to(selected_run_dir)}")
    else: print(f"--- ERROR saving final analysis metadata ---")

    # --- Close Debug Log ---
    if debug_log_file:
        try: debug_log_file.write(f"\n--- Analysis End Time: {datetime.datetime.now().isoformat()} ---\n"); debug_log_file.close(); print(f"Closed debug log file: {debug_log_path.name}")
        except Exception as e: print(f"Warning: Error closing debug log file: {e}")

    print(f"\nScript finished. Analysis results are in directory: {analysis_path.relative_to(base_dir)}")


# --- END OF FILE analyze_srm_sweep.py ---