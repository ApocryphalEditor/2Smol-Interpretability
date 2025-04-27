# Updated gpt2_chat_gui.py (Version 10.6.3 - Fix if/else syntax and final errors)

import tkinter as tk
from tkinter import scrolledtext
from tkinter import ttk
from tkinter import Canvas, Scrollbar, PanedWindow
import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer
import threading
import time
import warnings
# import matplotlib.pyplot as plt # Keep commented unless used
from collections import deque
from functools import partial

# --- Configuration ---
MODEL_NAME = "gpt2-small"
HISTORY_LINES = 25 # Max lines to keep in the chat history

# --- Tag Constants for Coloring ---
TAG_USER = "user_tag"
TAG_GPT = "gpt_tag"
TAG_SYSTEM = "system_tag"

# --- Main Application Class ---
class GPT2ChatApp:
    def __init__(self, root):
        self.root = root
        self.root.title(f"{MODEL_NAME} Chat & Activation Clamper")
        self.root.geometry("1000x700")
        self.root.minsize(800, 500) # Set a minimum size

        # Model and clamping related attributes
        self.model = None
        self.device = None
        self.tokenizer = None
        self.d_mlp = None       # Dimension of MLP layers
        self.n_layers = None    # Number of layers
        self.clamp_rows = []    # Stores UI elements for each clamp row

        # --- UI Setup ---
        # Main PanedWindow (divides left chat area and right clamp controls)
        self.main_pane = PanedWindow(root, orient=tk.HORIZONTAL, sashrelief=tk.RAISED, sashwidth=5)
        self.main_pane.pack(fill=tk.BOTH, expand=True)

        # Left Frame (Chat interface)
        self.left_frame = tk.Frame(self.main_pane, relief=tk.FLAT)
        self.main_pane.add(self.left_frame, minsize=500, stretch="always") # Give left frame priority on resize

        # Status Label (top of left frame)
        self.status_label = tk.Label(self.left_frame, text="Initializing...", fg="blue")
        self.status_label.pack(pady=5, fill=tk.X, padx=10)

        # Parameter Control Frame (Temp, Top-K, Max Tokens)
        param_frame = tk.Frame(self.left_frame, pady=5)
        param_frame.pack(fill=tk.X, padx=10)

        # Temperature
        self.temp_var = tk.DoubleVar(value=0.7)
        tk.Label(param_frame, text="Temp:").pack(side=tk.LEFT, padx=5)
        self.temp_label_value = tk.Label(param_frame, text=f"{self.temp_var.get():.2f}", width=4)
        self.temp_label_value.pack(side=tk.LEFT, padx=(0, 5))
        ttk.Scale(param_frame, from_=0.1, to=2.0, orient=tk.HORIZONTAL, variable=self.temp_var, length=80, command=self._update_temp_label).pack(side=tk.LEFT, padx=5)

        # Top-K
        self.top_k_var = tk.IntVar(value=40)
        tk.Label(param_frame, text="Top-K:").pack(side=tk.LEFT, padx=(15, 5))
        self.top_k_label_value = tk.Label(param_frame, text=f"{self.top_k_var.get()}", width=3)
        self.top_k_label_value.pack(side=tk.LEFT, padx=(0, 5))
        ttk.Scale(param_frame, from_=0, to=100, orient=tk.HORIZONTAL, variable=self.top_k_var, length=80, command=self._update_top_k_label).pack(side=tk.LEFT, padx=5) # Top-K=0 means greedy

        # Max New Tokens
        self.max_tokens_var = tk.IntVar(value=50)
        tk.Label(param_frame, text="MaxTok:").pack(side=tk.LEFT, padx=(15, 5))
        self.max_tokens_label_value = tk.Label(param_frame, text=f"{self.max_tokens_var.get()}", width=3)
        self.max_tokens_label_value.pack(side=tk.LEFT, padx=(0, 5))
        ttk.Scale(param_frame, from_=10, to=100, orient=tk.HORIZONTAL, variable=self.max_tokens_var, length=80, command=self._update_max_tokens_label).pack(side=tk.LEFT, padx=5)

        # Left Control Frame (Developer Mode Toggle, Token Counts)
        self.left_control_frame = tk.Frame(self.left_frame)
        self.left_control_frame.pack(fill=tk.X, padx=10, pady=(5,0))
        self.dev_mode_var = tk.BooleanVar(value=False)
        dev_toggle = tk.Checkbutton(self.left_control_frame, text="Developer Mode", variable=self.dev_mode_var, command=self.toggle_dev_mode)
        dev_toggle.pack(side=tk.LEFT)
        self.cumulative_token_label = tk.Label(self.left_control_frame, text="Total Tok: N/A")
        self.cumulative_token_label.pack(side=tk.RIGHT, padx=5)
        self.context_token_label = tk.Label(self.left_control_frame, text="Prompt Tok: N/A")
        self.context_token_label.pack(side=tk.RIGHT, padx=5)

        # Output Text Area (Chat History)
        self.output_text = scrolledtext.ScrolledText(self.left_frame, wrap=tk.WORD, state='disabled', height=20)
        self.output_text.tag_configure(TAG_USER, foreground="blue")
        self.output_text.tag_configure(TAG_GPT, foreground="#006400") # Dark Green
        self.output_text.tag_configure(TAG_SYSTEM, foreground="gray50") # Gray
        self.output_text.pack(pady=5, padx=10, fill=tk.BOTH, expand=True)

        # Input Frame (Text Entry, Submit, Clear)
        input_frame = tk.Frame(self.left_frame, pady=5)
        input_frame.pack(fill=tk.X, padx=10)
        self.input_entry = tk.Entry(input_frame, width=60)
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.input_entry.bind("<Return>", self.submit_prompt_event) # Bind Enter key
        self.submit_button = tk.Button(input_frame, text="Submit", command=self.submit_prompt, state=tk.DISABLED)
        self.submit_button.pack(side=tk.LEFT)
        self.clear_button = tk.Button(input_frame, text="Clear Chat", command=self.clear_chat, state=tk.DISABLED)
        self.clear_button.pack(side=tk.LEFT, padx=(5,0))

        # Right Frame (Clamp Controls - initially hidden or added later)
        self.right_frame = tk.Frame(self.main_pane, relief=tk.FLAT, width=300)
        # Don't add it to main_pane here; add it in toggle_dev_mode if needed

        # --- Clamp Controls (within right_frame) ---
        # Clamp Control Buttons (Add Clamp)
        clamp_control_frame = tk.Frame(self.right_frame)
        clamp_control_frame.pack(fill=tk.X, pady=(10, 5), padx=10)
        tk.Label(clamp_control_frame, text="MLP Clamps:").pack(side=tk.LEFT, padx=(0, 10))
        self.add_clamp_button_widget = tk.Button(clamp_control_frame, text="Add", command=self.add_clamp_row, state=tk.DISABLED)
        self.add_clamp_button_widget.pack(side=tk.LEFT)

        # Quick Add Frame (Add same clamp to all layers)
        quick_add_frame = tk.Frame(self.right_frame)
        quick_add_frame.pack(fill=tk.X, padx=10, pady=(0, 5))
        tk.Label(quick_add_frame, text="Quick Add:").pack(side=tk.LEFT, padx=(0,5))
        tk.Label(quick_add_frame, text="N:").pack(side=tk.LEFT, padx=(5,0))
        self.all_neuron_var = tk.StringVar()
        validate_neuron_cmd = self.root.register(self.validate_neuron_clamp) # Register validation func
        quick_neuron_entry = tk.Entry(quick_add_frame, textvariable=self.all_neuron_var, width=5, validate="key", validatecommand=(validate_neuron_cmd, '%P'))
        quick_neuron_entry.pack(side=tk.LEFT, padx=(0,5))
        tk.Label(quick_add_frame, text="V:").pack(side=tk.LEFT, padx=(5,0))
        self.all_value_var = tk.StringVar()
        validate_value_cmd = self.root.register(self.validate_value_clamp) # Register validation func
        quick_value_entry = tk.Entry(quick_add_frame, textvariable=self.all_value_var, width=5, validate="key", validatecommand=(validate_value_cmd, '%P'))
        quick_value_entry.pack(side=tk.LEFT, padx=(0,5))
        self.add_all_button = tk.Button(quick_add_frame, text="All Layers", command=self.add_all_layers_clamp, state=tk.DISABLED)
        self.add_all_button.pack(side=tk.LEFT)

        # Scrollable Clamp List Area
        clamp_canvas_container = tk.Frame(self.right_frame)
        clamp_canvas_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 5))
        self.clamp_canvas = Canvas(clamp_canvas_container, borderwidth=0, background="#ffffff")
        self.clamp_scrollbar = Scrollbar(clamp_canvas_container, orient="vertical", command=self.clamp_canvas.yview)
        self.clamp_canvas.configure(yscrollcommand=self.clamp_scrollbar.set)
        self.clamp_scrollbar.pack(side="right", fill="y")
        self.clamp_canvas.pack(side="left", fill="both", expand=True)
        # Frame inside the canvas to hold the actual clamp rows
        self.clamp_list_frame = tk.Frame(self.clamp_canvas, background=self.clamp_canvas.cget('background'))
        self.clamp_canvas_window = self.clamp_canvas.create_window((0, 0), window=self.clamp_list_frame, anchor="nw")
        # Bind events for scrolling configuration
        self.clamp_list_frame.bind("<Configure>", self._on_clamp_frame_configure) # Update scroll region when frame size changes
        self.clamp_canvas.bind("<Configure>", self._on_clamp_canvas_configure)   # Update window width when canvas size changes
        # Label shown when no clamps are added
        self.no_clamps_label = tk.Label(self.clamp_list_frame, text="No clamps added.", background=self.clamp_list_frame.cget('background'))
        # Don't pack no_clamps_label here, pack/unpack it as needed

        # --- Initialization ---
        self.root.update_idletasks() # Ensure UI is drawn before starting model load
        self.status_label.config(text="Loading model...")
        # Start model loading in a separate thread to keep UI responsive
        threading.Thread(target=self.setup_model, daemon=True).start()

    # --- Canvas/Scrolling Helpers ---
    def _on_clamp_frame_configure(self, event=None):
        """Update scroll region when the frame inside the canvas changes size."""
        self.clamp_canvas.configure(scrollregion=self.clamp_canvas.bbox("all"))

    def _on_clamp_canvas_configure(self, event):
        """Update the width of the frame window inside the canvas when the canvas resizes."""
        canvas_width = event.width
        self.clamp_canvas.itemconfig(self.clamp_canvas_window, width=canvas_width)

    # --- Input Validation ---
    def validate_neuron_clamp(self, P):
        """Validation function for neuron entry (allows empty or digits)."""
        if P == "" or P.isdigit():
            return True
        else:
            return False

    def validate_value_clamp(self, P):
        """Validation function for value entry (allows empty, '-', '.', or valid float)."""
        if P == "" or P == "-" or P == ".":
            return True
        # Allow partial float entry like "1." or "-0."
        if P.count('.') <= 1 and (P.startswith('-') or P[0].isdigit() or P == '.'):
             try:
                 if P != '-': float(P) # Check if convertible, except for just '-'
                 return True
             except ValueError:
                 return False # If it's not a valid float representation
        return False


    # --- UI Control ---
    def toggle_dev_mode(self):
        """Shows or hides the right panel for developer/clamping controls."""
        if self.dev_mode_var.get():
            try:
                # Check if right frame is already added before trying to configure/add
                if self.right_frame not in self.main_pane.panes():
                    self.main_pane.add(self.right_frame, minsize=250, stretch="never") # Add if not present
            except tk.TclError:
                # Fallback if adding fails for some reason
                 print("[Warning] Could not add developer panel.")
            # Enable clamp buttons only if model is loaded
            if self.model and self.n_layers is not None:
                self.add_clamp_button_widget.config(state=tk.NORMAL)
                self.add_all_button.config(state=tk.NORMAL)
            # Show "No clamps" label if needed
            if not self.clamp_rows:
                self.no_clamps_label.pack(pady=5) # Use pack here
        else:
            # Check if right frame is currently managed by the pane before forgetting
            if self.right_frame in self.main_pane.panes():
                 try:
                     self.main_pane.forget(self.right_frame)
                 except tk.TclError:
                      print("[Warning] Could not properly hide developer panel.")
            # Hide "No clamps" label if it's visible
            if self.no_clamps_label.winfo_ismapped():
                 self.no_clamps_label.pack_forget() # Use pack_forget

    def add_clamp_row(self, layer=None, neuron="", value=""):
        """Adds a new row of controls for defining an MLP clamp."""
        if not self.model or self.n_layers is None: # Ensure model is ready
             print("[Error] Cannot add clamp row: Model not loaded or layers not determined.")
             return

        if self.no_clamps_label.winfo_ismapped(): # Hide the "No clamps" label
            self.no_clamps_label.pack_forget()

        # Create frame for the row
        row_frame = tk.Frame(self.clamp_list_frame, background=self.clamp_list_frame.cget('background'))
        row_frame.pack(fill=tk.X, pady=1, padx=1) # Pack row frame into the list frame

        # Layer Selection
        tk.Label(row_frame, text="L:", background=row_frame.cget('background'), width=2).pack(side=tk.LEFT, padx=(2, 0))
        layer_var = tk.StringVar(value=layer if layer is not None else "0")
        layer_values = [str(i) for i in range(self.n_layers)]
        layer_combo = ttk.Combobox(row_frame, textvariable=layer_var, width=3, values=layer_values, state='readonly')
        layer_combo.pack(side=tk.LEFT, padx=(0, 3))

        # Neuron Index Input
        tk.Label(row_frame, text="N:", background=row_frame.cget('background'), width=2).pack(side=tk.LEFT, padx=(3, 0))
        neuron_var = tk.StringVar(value=neuron)
        validate_neuron_cmd = self.root.register(self.validate_neuron_clamp)
        neuron_entry = tk.Entry(row_frame, textvariable=neuron_var, width=5, validate="key", validatecommand=(validate_neuron_cmd, '%P'))
        neuron_entry.pack(side=tk.LEFT, padx=(0, 3))

        # Clamp Value Input
        tk.Label(row_frame, text="V:", background=row_frame.cget('background'), width=2).pack(side=tk.LEFT, padx=(3, 0))
        value_var = tk.StringVar(value=value)
        validate_value_cmd = self.root.register(self.validate_value_clamp)
        value_entry = tk.Entry(row_frame, textvariable=value_var, width=5, validate="key", validatecommand=(validate_value_cmd, '%P'))
        value_entry.pack(side=tk.LEFT, padx=(0, 3))

        # Store references to widgets for this row
        row_data = {
            "frame": row_frame,
            "layer_var": layer_var, "layer_combo": layer_combo,
            "neuron_var": neuron_var, "neuron_entry": neuron_entry,
            "value_var": value_var, "value_entry": value_entry
        }

        # Clone Button (appears last on the right)
        clone_button = tk.Button(row_frame, text="C", width=2, command=partial(self.clone_clamp_row, row_data), relief=tk.FLAT, bd=0, fg="blue", font=("Segoe UI", 7))
        clone_button.pack(side=tk.RIGHT, padx=1)
        row_data["clone_button"] = clone_button

        # Remove Button (appears before clone)
        remove_button = tk.Button(row_frame, text="X", fg="red", width=2, command=partial(self.remove_clamp_row, row_data), relief=tk.FLAT, bd=0, font=("Segoe UI", 7, "bold"))
        remove_button.pack(side=tk.RIGHT, padx=1)
        row_data["remove_button"] = remove_button

        self.clamp_rows.append(row_data)
        # Schedule canvas update slightly later to ensure frame is packed
        self.root.after(10, self._on_clamp_frame_configure)

    def clone_clamp_row(self, original_row_data):
        """Creates a duplicate of an existing clamp row."""
        try:
            # Get values from the original row's variables
            layer = original_row_data["layer_var"].get()
            neuron = original_row_data["neuron_var"].get()
            value = original_row_data["value_var"].get()
            # Add a new row with these values
            self.add_clamp_row(layer=layer, neuron=neuron, value=value)
        except Exception as e:
            print(f"[Error] Could not clone clamp row: {e}")

    def add_all_layers_clamp(self):
        """Adds a clamp with the specified neuron/value to all MLP layers."""
        if not self.model or self.d_mlp is None or self.n_layers is None:
            print("[Error] Quick Add: Model not fully loaded yet.")
            return
        try:
            neuron_str = self.all_neuron_var.get()
            value_str = self.all_value_var.get()

            if not neuron_str or not value_str:
                print("[Error] Quick Add: Neuron and Value cannot be empty.")
                return
            # Validate inputs
            if not neuron_str.isdigit():
                print("[Error] Quick Add: Neuron must be a number.")
                return
            neuron_idx = int(neuron_str)
            value = float(value_str) # Relies on validatecommand, but catch ValueError

            if not (0 <= neuron_idx < self.d_mlp):
                print(f"[Error] Quick Add: Neuron index {neuron_idx} out of range (0-{self.d_mlp-1}).")
                return

            print(f"[Info] Quick Add: Adding N{neuron_idx} V{value:.2f} for all {self.n_layers} layers.")
            for layer_idx in range(self.n_layers):
                self.add_clamp_row(layer=str(layer_idx), neuron=neuron_str, value=value_str)

            # Clear the quick add entries
            self.all_neuron_var.set("")
            self.all_value_var.set("")
        except ValueError:
            print("[Error] Quick Add: Invalid value format.")
        except Exception as e:
            print(f"[Error] Quick Add failed: {e}")

    def remove_clamp_row(self, row_data):
        """Removes a specific clamp row from the UI and internal list."""
        try:
            row_data["frame"].destroy() # Remove UI elements
            if row_data in self.clamp_rows:
                self.clamp_rows.remove(row_data) # Remove from internal list
        except Exception as e:
             print(f"[Error] Failed to remove clamp row: {e}")

        # Show "No clamps" label if the list is now empty and dev mode is on
        if not self.clamp_rows and self.dev_mode_var.get():
             if not self.no_clamps_label.winfo_ismapped(): # Check before packing
                 self.no_clamps_label.pack(pady=5)

        # Update scroll region
        self.root.after(10, self._on_clamp_frame_configure)

    # --- Parameter UI Updates ---
    def _update_temp_label(self, value): self.temp_label_value.config(text=f"{float(value):.2f}")
    def _update_top_k_label(self, value): self.top_k_label_value.config(text=f"{int(float(value))}") # Convert via float first
    def _update_max_tokens_label(self, value): self.max_tokens_label_value.config(text=f"{int(float(value))}") # Convert via float first
    def update_prompt_token_display(self, count): self.context_token_label.config(text=f"Prompt Tok: {count}")
    def update_cumulative_token_display(self, count): self.cumulative_token_label.config(text=f"Total Tok: {count}")

    # --- Model Loading ---
    def setup_model(self):
        try:
            t_start = time.time()
            # *** FIXED: Standard if/else block ***
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
            # **************************************
            print(f"Moving model to device: {self.device}")

            # Use torch.device context manager for loading
            with torch.device(self.device):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore") # Ignore harmless warnings during download/load
                    self.model = HookedTransformer.from_pretrained(MODEL_NAME)

                # Explicitly move model to device after loading (safer)
                self.model.to(self.device)
                self.tokenizer = self.model.tokenizer
                # Set pad token if not present (common for GPT-2)
                if self.tokenizer.pad_token_id is None:
                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            # Store model config details
            self.d_mlp = self.model.cfg.d_mlp
            self.n_layers = self.model.cfg.n_layers
            t_end = time.time()

            # Schedule UI updates on the main thread
            self.root.after(0, self.update_status, f"Model loaded on {self.device.upper()} in {t_end - t_start:.2f}s. Ready.", "green")
            self.root.after(0, self.enable_controls)
            self.root.after(0, self._finalize_setup) # Final setup steps (like enabling clamp controls)

        except Exception as e:
            # Schedule error status update on the main thread
            self.root.after(0, self.update_status, f"Error loading model: {e}", "red")
            print(f"[Error] Model Loading Failed: {e}")
            import traceback
            traceback.print_exc()
            # Keep controls disabled if loading fails

    def _finalize_setup(self):
        """Post-model-load setup steps."""
        if self.model and hasattr(self, 'n_layers') and self.n_layers is not None:
            print(f"[Info] Model setup finalized. Layers: {self.n_layers}, MLP Dim: {self.d_mlp}")
            # Update combobox values for any existing clamp rows (unlikely but safe)
            for row_data in self.clamp_rows:
                try:
                    row_data["layer_combo"]['values'] = [str(i) for i in range(self.n_layers)]
                    if row_data["layer_combo"]['state'] == 'disabled':
                        row_data["layer_combo"].config(state='readonly')
                except tk.TclError: pass # Widget might be gone
                except Exception as e: print(f"[Warning] Error updating existing combobox: {e}")

            # Enable Add buttons if in dev mode
            if self.dev_mode_var.get():
                self.add_clamp_button_widget.config(state=tk.NORMAL)
                self.add_all_button.config(state=tk.NORMAL)
                # Ensure "No clamps" label is correctly shown/hidden
                if not self.clamp_rows and not self.no_clamps_label.winfo_ismapped():
                     self.no_clamps_label.pack(pady=5)
                elif self.clamp_rows and self.no_clamps_label.winfo_ismapped():
                     self.no_clamps_label.pack_forget()
        else:
            print("[Warning] _finalize_setup called but model/config not fully ready.")


    def update_status(self, message, color):
        """Updates the status label text and color."""
        self.status_label.config(text=message, fg=color)

    def enable_controls(self):
        """Enables chat input controls after model is loaded."""
        self.submit_button.config(state=tk.NORMAL)
        self.clear_button.config(state=tk.NORMAL)
        self.input_entry.config(state=tk.NORMAL)
        self.input_entry.focus()
        # Dev mode controls are enabled in _finalize_setup after model details are confirmed


    # --- Chat History Management ---
    def _add_message_to_history(self, message, tag=None, add_newline=True):
        """Adds a message to the chat history text area with optional tagging."""
        if not self.tokenizer: return # Should not happen if controls are enabled, but safe check

        self.output_text.config(state='normal') # Enable editing
        try:
            insert_point = tk.END # Insert at the end
            current_content = self.output_text.get('1.0', tk.END).strip()

            # Add newline separator if needed
            if add_newline and len(current_content) > 0:
                # Avoid adding newline if the last char is already a newline
                if not self.output_text.get(f"{tk.END}-2c", f"{tk.END}-1c") == '\n':
                    self.output_text.insert(insert_point, "\n")

            # Get start index *before* inserting the new message
            start_index = self.output_text.index(f"{insert_point}-1c linestart") # Use linestart for tag consistency

            # Insert the message
            self.output_text.insert(insert_point, message)

            # Apply tag if provided
            if tag:
                # Get end index *after* inserting
                end_index = self.output_text.index(f"{insert_point}-1c lineend") # Use lineend for tag consistency
                # Add tag from start to end index
                self.output_text.tag_add(tag, start_index, end_index)

            # --- History Limit ---
            # Get all lines after insertion
            lines = self.output_text.get('1.0', tk.END).splitlines()
            if len(lines) > HISTORY_LINES:
                lines_to_keep = lines[-HISTORY_LINES:]
                # Store current scroll position (optional, can be jumpy)
                # scroll_pos = self.output_text.yview()
                self.output_text.delete('1.0', tk.END)
                self.output_text.insert('1.0', "\n".join(lines_to_keep))
                # Restore scroll position (optional)
                # self.output_text.yview_moveto(scroll_pos[0])
                # Only warn if non-empty lines were actually removed
                if any(line.strip() for line in lines[:-HISTORY_LINES]):
                     print(f"[Warning] History truncated to last {HISTORY_LINES} lines. Tags on truncated lines are lost.")

            # Auto-scroll to the end
            self.output_text.see(tk.END)

            # --- Cumulative Token Count ---
            # Run tokenization in a separate thread to avoid UI lag for long histories
            # Note: This means the count might update slightly after the message appears
            full_history_text = self.output_text.get('1.0', tk.END)
            threading.Thread(target=self._update_cumulative_token_count_thread, args=(full_history_text,), daemon=True).start()

        except Exception as e:
            print(f"[Error] Failed to add message to history: {e}")
            # Attempt to recover gracefully
        finally:
             # Ensure text widget is disabled even if errors occur
             self.output_text.config(state='disabled')

    def _update_cumulative_token_count_thread(self, text_to_tokenize):
        """Helper thread function to calculate token count."""
        try:
            history_token_ids = self.tokenizer.encode(text_to_tokenize)
            cumulative_count = len(history_token_ids)
            # Schedule UI update back on the main thread
            self.root.after(0, self.update_cumulative_token_display, cumulative_count)
        except Exception as e:
            print(f"[Error] Token count update failed: {e}")


    # --- Prompt Handling ---
    def submit_prompt_event(self, event):
        """Handles the Enter key press in the input entry."""
        self.submit_prompt()

    def submit_prompt(self):
        """Handles submitting the user's input prompt."""
        user_input = self.input_entry.get().strip()
        # Check if input is valid and model is ready
        if not user_input or self.model is None or self.tokenizer is None:
            print("[Info] Cannot submit prompt: No input or model not ready.")
            return

        # Disable controls during generation
        self._set_controls_state(tk.DISABLED)
        if self.dev_mode_var.get():
            self._set_clamp_controls_state(tk.DISABLED)

        # --- Prepare Prompt for Model ---
        # Note: We only send the user_input, not the whole history yet.
        # Context management would be needed for a real chatbot.
        prompt_text_for_model = user_input
        prompt_token_ids = self.tokenizer.encode(prompt_text_for_model)
        final_token_count = len(prompt_token_ids)
        max_context_tokens = self.model.cfg.n_ctx # Max tokens the model can handle
        max_allowed_prompt_tokens = max_context_tokens - self.max_tokens_var.get() - 1 # Reserve space for max_new_tokens + EOS

        # --- Prompt Truncation (if necessary) ---
        if final_token_count > max_allowed_prompt_tokens:
             print(f"[Warning] User input token count ({final_token_count}) plus max generation tokens exceeds context limit ({max_context_tokens}). Truncating input.")
             # Truncate from the *left* to keep the most recent part of the prompt
             prompt_token_ids = prompt_token_ids[final_token_count - max_allowed_prompt_tokens:]
             final_token_count = len(prompt_token_ids)
             prompt_text_for_model = self.tokenizer.decode(prompt_token_ids) # Decode the truncated IDs back to text
             print(f"  - New prompt token count: {final_token_count}")

        # Update UI *before* starting thread
        self.root.after(0, self.update_prompt_token_display, final_token_count)
        self._add_message_to_history(f"You: {user_input}", tag=TAG_USER)
        self.input_entry.delete(0, tk.END) # Clear input field

        # --- Process Clamps ---
        valid_clamps = []
        if self.dev_mode_var.get() and self.n_layers is not None and self.d_mlp is not None:
            print("[Info] Processing Clamps...")
            for row_data in self.clamp_rows:
                try:
                    layer_str = row_data["layer_var"].get()
                    neuron_str = row_data["neuron_var"].get()
                    value_str = row_data["value_var"].get()

                    # Skip if any field is empty
                    if not layer_str or not neuron_str or not value_str: continue

                    layer = int(layer_str)
                    neuron = int(neuron_str)
                    value = float(value_str)

                    # Validate indices and value
                    if 0 <= layer < self.n_layers and 0 <= neuron < self.d_mlp:
                        valid_clamps.append({"layer": layer, "neuron": neuron, "value": value})
                        # Don't print every valid clamp here, can be verbose. Maybe log level?
                        # print(f"  - Valid Clamp Added: L{layer} N{neuron} V{value:.2f}")
                    else:
                        print(f"  - Invalid Clamp Skipped: L{layer} N{neuron} (Out of range L:0-{self.n_layers-1}, N:0-{self.d_mlp-1})")
                except ValueError as e:
                    # Error parsing int/float
                    layer_val = row_data.get('layer_var','N/A').get() # Safe get
                    neuron_val = row_data.get('neuron_var','N/A').get()
                    value_val = row_data.get('value_var','N/A').get()
                    print(f"  - Invalid Clamp Skipped: Error parsing L{layer_val} N{neuron_val} V{value_val} ({e})")
                except Exception as e: # Catch other potential errors (e.g., accessing destroyed widgets)
                    print(f"  - Error processing clamp row: {e}")
            if valid_clamps:
                print(f"  - Found {len(valid_clamps)} valid clamps.")
            else:
                 print("  - No valid clamps found.")

        # Get generation parameters *before* starting thread
        temp = self.temp_var.get()
        top_k = self.top_k_var.get()
        max_new_tokens_param = self.max_tokens_var.get()

        # --- Start Generation Thread ---
        print(f"[Info] Starting generation (Temp: {temp:.2f}, Top-K: {top_k}, MaxNew: {max_new_tokens_param})")
        threading.Thread(
            target=self.generate_response,
            args=(prompt_text_for_model, temp, top_k, max_new_tokens_param, valid_clamps),
            daemon=True
        ).start()

    def _set_controls_state(self, state):
        """Sets the state of the main chat controls."""
        self.submit_button.config(state=state)
        self.clear_button.config(state=state)
        self.input_entry.config(state=state)

    def _set_clamp_controls_state(self, state):
        """Sets the state of all clamp-related controls."""
        effective_state = state
        # Readonly state for combobox when enabling
        combo_state = 'readonly' if state == tk.NORMAL else tk.DISABLED
        entry_state = 'normal' if state == tk.NORMAL else tk.DISABLED

        # Add/Quick Add Buttons
        # Don't enable add buttons if model isn't ready, even if requested state is NORMAL
        if state == tk.NORMAL and (not self.model or not self.n_layers or not self.d_mlp):
            effective_state = tk.DISABLED
            combo_state = tk.DISABLED
            entry_state = tk.DISABLED

        self.add_clamp_button_widget.config(state=effective_state)
        self.add_all_button.config(state=effective_state)

        # Individual Clamp Rows
        for row_data in self.clamp_rows:
             try:
                 row_data["layer_combo"].config(state=combo_state)
                 row_data["neuron_entry"].config(state=entry_state)
                 row_data["value_entry"].config(state=entry_state)
                 row_data["remove_button"].config(state=effective_state)
                 row_data["clone_button"].config(state=effective_state)
             except tk.TclError: pass # Ignore errors for destroyed widgets
             except Exception as e: print(f"[Warning] Error setting clamp row state: {e}")


    # --- Model Generation & Clamping ---
    def _mlp_clamp_hook(self, activation_batch, hook, neurons_to_clamp):
        """Hook function to clamp MLP activations at hook_post."""
        # activation_batch shape: [batch_size, sequence_length, d_mlp]
        # We clamp the activations for the *last token* in the sequence
        # as that's what influences the next token prediction.
        for neuron_idx, clamp_value in neurons_to_clamp.items():
             try:
                 activation_batch[:, -1, neuron_idx] = clamp_value
             except IndexError:
                 # This indicates a potential mismatch between expected d_mlp and actual activation shape
                 print(f"[Error] Hook error: Neuron index {neuron_idx} out of bounds for d_mlp={activation_batch.shape[-1]} at layer {hook.layer()}.")
             except Exception as e:
                 print(f"[Error] Unexpected error in clamp hook: {e}")
        return activation_batch # IMPORTANT: Always return the activations


    def _sample_next_token(self, logits, temperature, top_k):
        """Samples the next token ID from the logits."""
        # logits shape: [batch_size, vocab_size], expect batch_size=1 here
        logits = logits[0] # Reduce to [vocab_size]

        if top_k == 0: # Greedy sampling (deterministic)
            next_token_index = torch.argmax(logits, dim=-1)
            return next_token_index.view(1, 1) # Reshape to [batch_size=1, seq_len=1]
        else:
            # Apply temperature scaling (makes distribution sharper or flatter)
            if temperature > 0:
                logits = logits / temperature

            # Apply top-k filtering (set probability of non-top-k tokens to zero)
            if top_k > 0 and top_k < logits.shape[-1]: # Ensure k is valid
                top_k_logits, top_k_indices = torch.topk(logits, top_k)
                # Create a mask filled with negative infinity (log(0))
                mask = torch.full_like(logits, -float('inf'))
                # Scatter the actual top-k logits back into the mask
                mask.scatter_(dim=-1, index=top_k_indices, src=top_k_logits)
                logits = mask # Use the masked logits for probability calculation

            # Calculate probabilities using softmax
            probabilities = F.softmax(logits, dim=-1)
            # Sample from the distribution
            next_token = torch.multinomial(probabilities, num_samples=1) # Shape [1]
            return next_token.unsqueeze(0) # Reshape to [batch_size=1, seq_len=1]


    def generate_response(self, prompt_text, temperature, top_k, max_new_tokens, clamp_list):
        """Generates a response from the model, applying clamps if specified."""
        hooks_created = False
        generated_token_ids = []
        clamp_details_str = ""
        hook_handles = [] # Store hook handles to remove them later

        try:
            # --- Add Hooks if Clamping ---
            if clamp_list:
                hooks_created = True
                clamps_by_layer = {} # Dictionary: layer_idx -> {neuron_idx: value}
                clamp_log_parts = [] # For display string

                # Organize clamps by layer
                for clamp in clamp_list:
                    layer = clamp["layer"]
                    if layer not in clamps_by_layer:
                        clamps_by_layer[layer] = {}
                    clamps_by_layer[layer][clamp["neuron"]] = clamp["value"]
                    # Format consistently for logging
                    clamp_log_parts.append(f"L{clamp['layer']} N{clamp['neuron']} V{clamp['value']:.2f}")

                if clamp_log_parts: # Build display string and add hooks
                    clamp_details_str = "\n  Clamps: " + ", ".join(clamp_log_parts)
                    print(f"[Info] Adding {len(clamps_by_layer)} clamp hooks...")
                    for layer_idx, neurons_to_clamp in clamps_by_layer.items():
                        hook_point = f"blocks.{layer_idx}.mlp.hook_post" # Target the output of MLP layer
                        # Use functools.partial to pass the specific neurons_to_clamp for this layer's hook
                        hook_fn = partial(self._mlp_clamp_hook, neurons_to_clamp=neurons_to_clamp)
                        # Add the hook and store its handle for later removal
                        hook_handle = self.model.add_hook(hook_point, hook_fn)
                        hook_handles.append(hook_handle)
                        print(f"  - Hook added for layer {layer_idx}")

            # --- Prepare Input ---
            input_ids = self.model.to_tokens(prompt_text).to(self.device)
            current_sequence = input_ids # Start with the prompt tokens
            t_gen_start = time.time()

            # --- Generation Loop ---
            self.model.eval() # Set model to evaluation mode
            with torch.no_grad(): # Disable gradient calculations for inference
                for i in range(max_new_tokens):
                    # Get model output (logits) - hooks run during the forward pass
                    model_output_logits = self.model(current_sequence)
                    # Extract logits for the *last* token position -> shape [batch_size, vocab_size]
                    logits_for_last_token = model_output_logits[:, -1, :]
                    # Sample the next token ID based on the logits
                    next_token_id_tensor = self._sample_next_token(logits_for_last_token, temperature, top_k)
                    next_token_id = next_token_id_tensor.item()

                    # Append generated token ID
                    generated_token_ids.append(next_token_id)
                    # Update the sequence for the next iteration
                    current_sequence = torch.cat([current_sequence, next_token_id_tensor], dim=1)

                    # Check if EOS token was generated
                    if next_token_id == self.tokenizer.eos_token_id:
                        print("[Info] EOS token generated. Stopping generation.")
                        break # Exit the loop

                    # Optional: Add a small sleep to prevent potential UI lockup on very fast generation
                    # time.sleep(0.01)

            t_gen_end = time.time()
            # Decode the generated token IDs into text
            response_text = self.tokenizer.decode(generated_token_ids).strip()
            num_response_tokens = len(generated_token_ids)

            # --- Prepare Payload for UI ---
            response_display_payload = (
                f"{response_text}\n"
                f"[Tokens: {num_response_tokens} | GenTime: {t_gen_end - t_gen_start:.2f}s | "
                f"Temp: {temperature:.2f} | Top-K: {top_k} | Max: {max_new_tokens}]"
                f"{clamp_details_str}" # Append clamp details string if clamps were used
            )
            # Schedule UI update on the main thread
            self.root.after(0, self.finalize_generation, response_display_payload, False) # is_error=False

        except Exception as e:
            error_message = f"Error during generation/clamping: {e}"
            print(f"[Error] {error_message}")
            import traceback
            traceback.print_exc()
            # Schedule error message display on the main thread
            self.root.after(0, self.finalize_generation, f"System Error: {e}", True) # is_error=True
        finally:
             # --- Remove Hooks ---
             # Ensure hooks are removed *after* generation or error handling
             if hooks_created and hook_handles:
                 print(f"[Info] Removing {len(hook_handles)} hooks.")
                 for handle in hook_handles:
                     try:
                         handle.remove()
                     except Exception as remove_e:
                          print(f"[Warning] Error removing hook: {remove_e}")
             elif hooks_created:
                 # Fallback if handle list wasn't populated correctly (shouldn't happen)
                 print("[Warning] Hooks were created but handles list is empty. Attempting model.reset_hooks().")
                 try:
                     self.model.reset_hooks(including_permanent=False)
                 except Exception as reset_e:
                     print(f"[Error] Failed to reset hooks: {reset_e}")


    def finalize_generation(self, response_payload, is_error=False):
        """Updates the UI after generation finishes or an error occurs."""
        # Add response/error message to history
        if is_error:
            self._add_message_to_history(response_payload, tag=TAG_SYSTEM)
        else:
            # Split response text from stats/clamp info
            parts = response_payload.split('\n', 1)
            gpt_response = parts[0]
            stats_and_clamps = parts[1] if len(parts) > 1 else ""

            # Add GPT response first
            self._add_message_to_history(f"GPT-2: {gpt_response}", tag=TAG_GPT)
            # Add stats/clamp info on a new line if there was a response, otherwise append directly
            if stats_and_clamps:
                 self._add_message_to_history(stats_and_clamps, tag=TAG_SYSTEM, add_newline=(len(gpt_response)>0))

        # Re-enable controls only if the model is loaded
        if self.model:
             self._set_controls_state(tk.NORMAL)
             if self.dev_mode_var.get():
                 self._set_clamp_controls_state(tk.NORMAL)
             self.input_entry.focus() # Set focus back to input field
        else:
             # Keep controls disabled if something went wrong with the model during generation
             self._set_controls_state(tk.DISABLED)
             self._set_clamp_controls_state(tk.DISABLED)


    # --- Other Actions ---
    def clear_chat(self):
        """Clears the chat history."""
        self.output_text.config(state='normal')
        self.output_text.delete('1.0', tk.END)
        self.output_text.config(state='disabled')
        # Add cleared message, ensuring no leading newline if chat was empty
        self._add_message_to_history("System: Chat cleared.", tag=TAG_SYSTEM, add_newline=False)
        # Reset token displays
        self.root.after(0, self.update_prompt_token_display, "N/A")
        self.root.after(0, self.update_cumulative_token_display, 0)

    # --- Placeholder ---
    def plot_activation_trend(self):
        """Placeholder for future activation plotting functionality."""
        print("[Info] Activation plotting not implemented yet.")
        pass

# --- Run the Application ---
if __name__ == "__main__":
    # Set high DPI awareness on Windows (optional, improves text clarity)
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except ImportError: # Not Windows
        pass
    except AttributeError: # Older Windows version
        try:
           windll.user32.SetProcessDPIAware()
        except: pass # Ignore if fails

    root = tk.Tk()
    app = GPT2ChatApp(root)
    root.mainloop()