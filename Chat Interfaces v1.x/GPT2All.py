# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, simpledialog
import torch
import transformer_lens
from transformer_lens.hook_points import HookPoint
import threading
import queue # For thread-safe UI updates
import traceback # For detailed error logging

# --- Configuration ---
MODEL_NAME = "gpt2-small"
NUM_LAYERS = 12  # GPT-2 Small has 12 layers (0-11)
# GPT-2 Small MLP dimension (d_mlp = 4 * d_model = 4 * 768)
MLP_DIM = 3072
DEFAULT_TEMP = 0.7
DEFAULT_TOP_K = 40
DEFAULT_MAX_NEW_TOKENS = 50

# --- Main Application Class ---

class Gpt2AllTokenApp:
    def __init__(self, master):
        self.master = master
        self.master.title(f"GPT-2 Small All-Token Generation w/ Neuron Intervention")
        self.master.geometry("1000x700")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None # Loaded later in a separate thread
        self.tokenizer = None
        self.clamps = [] # List to store clamp dictionaries: {'layer': int, 'neuron': int, 'value': float}
        self.is_generating = False
        self.ui_update_queue = queue.Queue() # Queue for thread-safe UI updates
        self.active_clamps_for_generation = [] # Thread-local storage during generation

        # --- Add greedy sampling variable ---
        self.greedy_var = tk.BooleanVar(value=False)

        self._create_widgets()
        self._load_model_async()
        self.master.after(100, self._process_ui_queue) # Start checking the queue

    def _process_ui_queue(self):
        """Process updates from the generation thread."""
        try:
            while True:
                task, args = self.ui_update_queue.get_nowait()
                task(*args)
        except queue.Empty:
            pass
        finally:
            # Reschedule check
            self.master.after(100, self._process_ui_queue)

    def _queue_ui_update(self, task, *args):
        """Safely add a task to the UI update queue from another thread."""
        self.ui_update_queue.put((task, args))

    def _update_status(self, message):
        """Update status bar (thread-safe)."""
        self.status_label.config(text=message)

    def _append_to_chat(self, text, tag=None):
        """Append text to chat history (thread-safe)."""
        # Ensure chat history is accessible before modifying state
        if hasattr(self, 'chat_history') and self.chat_history:
            try:
                self.chat_history.config(state=tk.NORMAL)
                if tag:
                    self.chat_history.insert(tk.END, text, tag)
                else:
                    self.chat_history.insert(tk.END, text)
                self.chat_history.see(tk.END)
                self.chat_history.config(state=tk.DISABLED)
            except tk.TclError as e:
                print(f"Error updating chat history (widget might be destroyed): {e}")


    def _update_token_stats(self, input_tokens, output_tokens):
        """Update token count label (thread-safe)."""
         # Ensure label exists before configuring
        if hasattr(self, 'token_stats_label') and self.token_stats_label:
             try:
                self.token_stats_label.config(text=f"Input Tokens: {input_tokens} | Output Tokens: {output_tokens}")
             except tk.TclError as e:
                print(f"Error updating token stats label (widget might be destroyed): {e}")


    def _set_generate_button_state(self, state):
         """Enable/disable generate button (thread-safe)."""
         # Ensure button exists before configuring
         if hasattr(self, 'generate_button') and self.generate_button:
            try:
                self.generate_button.config(state=state)
            except tk.TclError as e:
                print(f"Error setting generate button state (widget might be destroyed): {e}")


    def _load_model_async(self):
        """Loads the model in a separate thread to avoid freezing the UI."""
        self.status_label.config(text=f"Loading {MODEL_NAME}...")
        self.generate_button.config(state=tk.DISABLED)

        def _load():
            try:
                model = transformer_lens.HookedTransformer.from_pretrained(
                    MODEL_NAME, device=self.device
                )
                model.eval() # Set to evaluation mode
                tokenizer = model.tokenizer
                # Add pad token if it doesn't exist (like in GPT-2)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                # --- Store model and tokenizer ONLY after successful load ---
                self.model = model
                self.tokenizer = tokenizer

                # --- Queue UI updates ---
                self._queue_ui_update(self._update_status, f"Model {MODEL_NAME} loaded on {self.device}.")
                self._queue_ui_update(self._set_generate_button_state, tk.NORMAL)
                # --- FIX: Re-evaluate Temp/Top-K controls state AFTER model is loaded ---
                self._queue_ui_update(self._toggle_sampling_params)

            except Exception as e:
                 # Log full error for debugging
                 tb_str = traceback.format_exc()
                 print(f"Model Load Error:\n{tb_str}")
                 # Ensure model is None if loading failed
                 self.model = None
                 self.tokenizer = None
                 self._queue_ui_update(self._update_status, f"Error loading model: {e}")
                 # Make sure generate button remains disabled on error
                 self._queue_ui_update(self._set_generate_button_state, tk.DISABLED)
                 # Keep sampling params disabled too
                 self._queue_ui_update(self._toggle_sampling_params)
                 self._queue_ui_update(messagebox.showerror, "Model Load Error", f"Failed to load model: {e}\n\nCheck console for details.")


        threading.Thread(target=_load, daemon=True).start()


    def _create_widgets(self):
        # Main layout panes
        self.paned_window = ttk.PanedWindow(self.master, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # --- Left Panel (Chat & Controls) ---
        self.left_frame = ttk.Frame(self.paned_window, padding=5)
        self.paned_window.add(self.left_frame, weight=3) # Give more weight to left panel

        # Input Area
        input_frame = ttk.Frame(self.left_frame)
        input_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(input_frame, text="Input Prompt:").pack(side=tk.LEFT, padx=(0, 5))
        self.input_text = ttk.Entry(input_frame, width=60)
        self.input_text.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.input_text.bind("<Return>", self._on_generate_enter) # Bind Enter key

        # Chat History
        self.chat_history = scrolledtext.ScrolledText(self.left_frame, wrap=tk.WORD, height=20, state=tk.DISABLED)
        self.chat_history.pack(fill=tk.BOTH, expand=True, pady=5)
        # Define tags for styling chat messages
        self.chat_history.tag_configure("user", foreground="blue")
        self.chat_history.tag_configure("model", foreground="green")
        self.chat_history.tag_configure("info", foreground="gray", font=("TkDefaultFont", 8))
        self.chat_history.tag_configure("error", foreground="red")

        # Controls Frame (Bottom Left)
        controls_frame = ttk.Frame(self.left_frame)
        controls_frame.pack(fill=tk.X, pady=(5, 0))

        # Generation Parameters
        param_frame = ttk.Frame(controls_frame)
        # Use grid layout for easier alignment with checkbox
        param_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        ttk.Label(param_frame, text="Temp:").grid(row=0, column=0, padx=2, pady=2, sticky="w")
        self.temp_var = tk.DoubleVar(value=DEFAULT_TEMP)
        self.temp_spinbox = ttk.Spinbox(param_frame, from_=0.01, to=2.0, increment=0.1, textvariable=self.temp_var, width=5)
        self.temp_spinbox.grid(row=0, column=1, padx=2, pady=2)

        ttk.Label(param_frame, text="Top-K:").grid(row=0, column=2, padx=(10, 2), pady=2, sticky="w")
        self.top_k_var = tk.IntVar(value=DEFAULT_TOP_K)
        self.top_k_spinbox = ttk.Spinbox(param_frame, from_=0, to=100, increment=1, textvariable=self.top_k_var, width=5) # 0 disables top-k
        self.top_k_spinbox.grid(row=0, column=3, padx=2, pady=2)

        ttk.Label(param_frame, text="Max New Tokens:").grid(row=1, column=0, padx=2, pady=2, sticky="w")
        self.max_tokens_var = tk.IntVar(value=DEFAULT_MAX_NEW_TOKENS)
        self.max_tokens_spinbox = ttk.Spinbox(param_frame, from_=1, to=1024, increment=1, textvariable=self.max_tokens_var, width=6)
        self.max_tokens_spinbox.grid(row=1, column=1, padx=2, pady=2)

        # --- Add Greedy Checkbox ---
        self.greedy_checkbox = ttk.Checkbutton(param_frame, text="Greedy (Deterministic)", variable=self.greedy_var, command=self._toggle_sampling_params)
        self.greedy_checkbox.grid(row=1, column=2, columnspan=2, padx=(10, 2), pady=2, sticky="w")

        # Generate Button (move to the right of param_frame)
        self.generate_button = ttk.Button(controls_frame, text="Generate", command=self._trigger_generation, state=tk.DISABLED)
        self.generate_button.pack(side=tk.RIGHT, padx=(10, 0), anchor="se") # Anchor south-east

        # Status Bar Area
        status_frame = ttk.Frame(self.left_frame)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=(5,0))
        self.token_stats_label = ttk.Label(status_frame, text="Input Tokens: 0 | Output Tokens: 0")
        self.token_stats_label.pack(side=tk.LEFT)
        self.status_label = ttk.Label(status_frame, text="App Initialized.")
        self.status_label.pack(side=tk.RIGHT)


        # --- Right Panel (Neuron Clamping) ---
        self.right_frame = ttk.Frame(self.paned_window, padding=5)
        self.paned_window.add(self.right_frame, weight=1)

        ttk.Label(self.right_frame, text="Neuron Clamping", font=("TkDefaultFont", 12, "bold")).pack(pady=(0, 10))

        # Add Clamp Frame
        add_clamp_frame = ttk.LabelFrame(self.right_frame, text="Add Clamp", padding=5)
        add_clamp_frame.pack(fill=tk.X, pady=5)

        ttk.Label(add_clamp_frame, text="Layer (0-11):").grid(row=0, column=0, padx=2, pady=2, sticky="w")
        self.clamp_layer_var = tk.IntVar(value=0)
        self.clamp_layer_spinbox = ttk.Spinbox(add_clamp_frame, from_=0, to=NUM_LAYERS - 1, increment=1, textvariable=self.clamp_layer_var, width=5)
        self.clamp_layer_spinbox.grid(row=0, column=1, padx=2, pady=2)

        ttk.Label(add_clamp_frame, text="Neuron (0-3071):").grid(row=1, column=0, padx=2, pady=2, sticky="w")
        self.clamp_neuron_var = tk.StringVar()
        self.clamp_neuron_entry = ttk.Entry(add_clamp_frame, textvariable=self.clamp_neuron_var, width=7)
        self.clamp_neuron_entry.grid(row=1, column=1, padx=2, pady=2)

        ttk.Label(add_clamp_frame, text="Value:").grid(row=2, column=0, padx=2, pady=2, sticky="w")
        self.clamp_value_var = tk.StringVar(value="1.0")
        self.clamp_value_entry = ttk.Entry(add_clamp_frame, textvariable=self.clamp_value_var, width=7)
        self.clamp_value_entry.grid(row=2, column=1, padx=2, pady=2)

        self.add_clamp_button = ttk.Button(add_clamp_frame, text="Add", command=self._add_clamp)
        self.add_clamp_button.grid(row=3, column=0, columnspan=2, pady=(5, 0))

        # Clamp List Frame
        clamp_list_frame = ttk.LabelFrame(self.right_frame, text="Active Clamps", padding=5)
        clamp_list_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.clamp_listbox = tk.Listbox(clamp_list_frame, height=10)
        self.clamp_listbox.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        # Consider adding scrollbar for very long lists
        # clamp_scrollbar = ttk.Scrollbar(clamp_list_frame, orient=tk.VERTICAL, command=self.clamp_listbox.yview)
        # self.clamp_listbox.configure(yscrollcommand=clamp_scrollbar.set)
        # clamp_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        clamp_list_buttons_frame = ttk.Frame(clamp_list_frame)
        clamp_list_buttons_frame.pack(fill=tk.X, pady=(5, 0))

        self.remove_clamp_button = ttk.Button(clamp_list_buttons_frame, text="Remove Selected", command=self._remove_selected_clamp)
        self.remove_clamp_button.pack(side=tk.LEFT, padx=(0, 5))

        self.clear_clamps_button = ttk.Button(clamp_list_buttons_frame, text="Clear All", command=self._clear_clamps)
        self.clear_clamps_button.pack(side=tk.RIGHT)

        # Quick Add Frame
        quick_add_frame = ttk.LabelFrame(self.right_frame, text="Quick Add to All MLP Layers", padding=5)
        quick_add_frame.pack(fill=tk.X, pady=5)

        ttk.Label(quick_add_frame, text="Neuron (0-3071):").grid(row=0, column=0, padx=2, pady=2, sticky="w")
        self.quick_neuron_var = tk.StringVar()
        self.quick_neuron_entry = ttk.Entry(quick_add_frame, textvariable=self.quick_neuron_var, width=7)
        self.quick_neuron_entry.grid(row=0, column=1, padx=2, pady=2)

        ttk.Label(quick_add_frame, text="Value:").grid(row=1, column=0, padx=2, pady=2, sticky="w")
        self.quick_value_var = tk.StringVar(value="1.0")
        self.quick_value_entry = ttk.Entry(quick_add_frame, textvariable=self.quick_value_var, width=7)
        self.quick_value_entry.grid(row=1, column=1, padx=2, pady=2)

        self.quick_add_button = ttk.Button(quick_add_frame, text="Quick Add", command=self._quick_add_clamps)
        self.quick_add_button.grid(row=2, column=0, columnspan=2, pady=(5, 0))

        # --- Set initial state of Temp/Top-K based on greedy checkbox ---
        # This will initially disable them because self.model is None
        self._toggle_sampling_params()


    # --- Add method to toggle sampling param controls ---
    def _toggle_sampling_params(self):
        """Enable/disable Temp and Top-K controls based on Greedy checkbox and model state."""
        # Ensure widgets exist before configuring
        if not hasattr(self, 'temp_spinbox') or not hasattr(self, 'top_k_spinbox'):
             return # Widgets not created yet

        try:
            if self.greedy_var.get():
                self.temp_spinbox.config(state=tk.DISABLED)
                self.top_k_spinbox.config(state=tk.DISABLED)
            else:
                # Enable only if NOT greedy AND model is loaded
                if self.model:
                    self.temp_spinbox.config(state=tk.NORMAL)
                    self.top_k_spinbox.config(state=tk.NORMAL)
                else: # Keep disabled if model not loaded or greedy is false but model is None
                     self.temp_spinbox.config(state=tk.DISABLED)
                     self.top_k_spinbox.config(state=tk.DISABLED)
        except tk.TclError as e:
             print(f"Error toggling sampling params (widget might be destroyed): {e}")


    # --- Clamp Management Methods ---

    def _add_clamp(self):
        try:
            layer = self.clamp_layer_var.get()
            neuron_str = self.clamp_neuron_var.get()
            value_str = self.clamp_value_var.get()

            if not neuron_str or not value_str:
                 messagebox.showwarning("Input Error", "Neuron index and Value cannot be empty.")
                 return

            neuron = int(neuron_str)
            value = float(value_str)

            if not (0 <= layer < NUM_LAYERS):
                messagebox.showwarning("Input Error", f"Layer must be between 0 and {NUM_LAYERS - 1}.")
                return
            if not (0 <= neuron < MLP_DIM):
                 messagebox.showwarning("Input Error", f"Neuron index must be between 0 and {MLP_DIM - 1}.")
                 return

            # Avoid duplicate clamps (optional, but good practice)
            new_clamp = {"layer": layer, "neuron": neuron, "value": value}
            if new_clamp not in self.clamps:
                self.clamps.append(new_clamp)
                self.clamps.sort(key=lambda c: (c['layer'], c['neuron'])) # Keep sorted
                self._update_clamp_list_display()
            else:
                 messagebox.showinfo("Info", "Clamp already exists.")

            # Clear inputs for next entry (optional)
            # self.clamp_neuron_var.set("")
            # self.clamp_value_var.set("1.0")

        except ValueError:
            messagebox.showwarning("Input Error", "Neuron index must be an integer and Value must be a float.")
        except Exception as e:
             messagebox.showerror("Error", f"An unexpected error occurred adding clamp: {e}")

    def _remove_selected_clamp(self):
        selected_indices = self.clamp_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("Selection Error", "Please select a clamp to remove.")
            return

        # Remove items in reverse order to avoid index shifting issues
        for index in sorted(selected_indices, reverse=True):
             if 0 <= index < len(self.clamps): # Bounds check
                 del self.clamps[index]

        self._update_clamp_list_display()

    def _clear_clamps(self):
        if not self.clamps:
            return
        if messagebox.askyesno("Confirm", "Are you sure you want to clear all clamps?"):
            self.clamps.clear()
            self._update_clamp_list_display()

    def _quick_add_clamps(self):
        try:
            neuron_str = self.quick_neuron_var.get()
            value_str = self.quick_value_var.get()

            if not neuron_str or not value_str:
                 messagebox.showwarning("Input Error", "Neuron index and Value cannot be empty for Quick Add.")
                 return

            neuron = int(neuron_str)
            value = float(value_str)

            if not (0 <= neuron < MLP_DIM):
                 messagebox.showwarning("Input Error", f"Neuron index must be between 0 and {MLP_DIM - 1}.")
                 return

            added_count = 0
            existing_count = 0
            for layer in range(NUM_LAYERS):
                # Check if a clamp for this layer/neuron already exists
                if not any(c['layer'] == layer and c['neuron'] == neuron for c in self.clamps):
                    new_clamp = {"layer": layer, "neuron": neuron, "value": value}
                    self.clamps.append(new_clamp)
                    added_count += 1
                else:
                    existing_count +=1 # Count how many existed

            if added_count > 0:
                self.clamps.sort(key=lambda c: (c['layer'], c['neuron'])) # Keep sorted
                self._update_clamp_list_display()
                info_msg = f"Added clamp for N{neuron} to {added_count} new MLP layers."
                if existing_count > 0:
                    info_msg += f"\n(Clamp already existed in {existing_count} layers)."
                messagebox.showinfo("Quick Add", info_msg)
            else:
                 # Only show message if clamps were attempted (i.e., neuron/value were valid)
                 if existing_count > 0:
                     messagebox.showinfo("Quick Add", f"Clamp for N{neuron} already exists in all {existing_count} layers. No new clamps added.")
                 # else: # Should not happen if validation passed, but maybe a safeguard
                 #    messagebox.showinfo("Quick Add", "No clamps were added.")


        except ValueError:
            messagebox.showwarning("Input Error", "Neuron index must be an integer and Value must be a float.")
        except Exception as e:
             messagebox.showerror("Error", f"An unexpected error occurred during Quick Add: {e}")

    def _update_clamp_list_display(self):
        # Ensure listbox exists
        if not hasattr(self, 'clamp_listbox') or not self.clamp_listbox:
            return
        try:
            self.clamp_listbox.delete(0, tk.END)
            for clamp in self.clamps:
                display_text = f"L{clamp['layer']} N{clamp['neuron']} = {clamp['value']:.4f}"
                self.clamp_listbox.insert(tk.END, display_text)
        except tk.TclError as e:
             print(f"Error updating clamp listbox (widget might be destroyed): {e}")


    # --- Generation Logic ---

    def _on_generate_enter(self, event=None):
        """Callback for pressing Enter in the input box."""
        self._trigger_generation()

    def _trigger_generation(self):
        """Starts the generation process in a separate thread."""
        if self.is_generating:
            messagebox.showwarning("Busy", "Generation is already in progress.")
            return
        if not self.model or not self.tokenizer:
            messagebox.showerror("Error", "Model is not loaded yet.")
            return

        prompt_text = self.input_text.get().strip()
        if not prompt_text:
            messagebox.showwarning("Input Error", "Please enter a prompt.")
            return

        try:
            temperature = self.temp_var.get()
            top_k = self.top_k_var.get()
            max_new_tokens = self.max_tokens_var.get()
            use_greedy = self.greedy_var.get() # Get greedy state

            # Validation only needed if not greedy
            if not use_greedy:
                if not (0.0 < temperature): # Use 0.0 strictly, 0.01 lower bound on spinbox handles UI
                    messagebox.showwarning("Input Error", "Temperature must be positive.")
                    return
                if not (0 <= top_k): # Allow 0 for "disable"
                    messagebox.showwarning("Input Error", "Top-K must be non-negative (0 to disable).")
                    return
            if not (0 < max_new_tokens <= 1024): # Added upper bound for safety
                messagebox.showwarning("Input Error", "Max New Tokens must be between 1 and 1024.")
                return

        except ValueError:
            messagebox.showwarning("Input Error", "Invalid generation parameter value.")
            return
        except tk.TclError:
             messagebox.showwarning("Input Error", "Could not read generation parameter value (UI error).")
             return


        self.is_generating = True
        self._queue_ui_update(self._set_generate_button_state, tk.DISABLED)
        # Also disable sampling param controls during generation
        self._queue_ui_update(lambda: self.temp_spinbox.config(state=tk.DISABLED))
        self._queue_ui_update(lambda: self.top_k_spinbox.config(state=tk.DISABLED))
        self._queue_ui_update(lambda: self.greedy_checkbox.config(state=tk.DISABLED))

        self._queue_ui_update(self._update_status, "Generating...")
        self._queue_ui_update(self._append_to_chat, f"You: {prompt_text}\n", "user")

        # Display active clamps in chat
        # Make a copy of self.clamps for the thread
        current_clamps_copy = list(self.clamps)
        if current_clamps_copy:
            clamp_info = "Active Clamps:\n" + "\n".join([f"  L{c['layer']} N{c['neuron']} = {c['value']:.4f}" for c in current_clamps_copy]) + "\n"
            self._queue_ui_update(self._append_to_chat, clamp_info, "info")
        else:
             self._queue_ui_update(self._append_to_chat, "Active Clamps: None\n", "info")

        # Display generation settings info
        gen_info = f"Generation Settings: Max={max_new_tokens}"
        if use_greedy:
            gen_info += ", Method=Greedy"
        else:
            gen_info += f", Temp={temperature:.2f}, Top-K={top_k if top_k > 0 else 'Off'}" # Show 'Off' if top_k=0
        self._queue_ui_update(self._append_to_chat, gen_info + "\n", "info")


        # Run generation in a separate thread
        threading.Thread(
            target=self._generate_response_thread,
            args=(prompt_text, temperature, top_k, max_new_tokens, current_clamps_copy, use_greedy), # Pass the copy of clamps
            daemon=True
        ).start()

    def _mlp_clamp_hook(self, activation: torch.Tensor, hook: HookPoint) -> torch.Tensor:
        """
        Hook function to clamp neuron activations in MLP layers.
        This version clamps across ALL sequence positions.
        activation shape: [batch, seq_pos, d_mlp]
        """
        current_layer = hook.layer()
        # Modify in place as we are in torch.no_grad()
        clamped_activation = activation

        # Use the copy of clamps stored for this specific generation run
        for clamp in self.active_clamps_for_generation: # This is set at the start of _generate_response_thread
            if clamp['layer'] == current_layer:
                neuron_idx = clamp['neuron']
                clamp_value = clamp['value']
                # Clamp across all sequence positions for the specified neuron
                try:
                    # Ensure neuron_idx is within bounds for safety, though validation should catch this
                     if 0 <= neuron_idx < clamped_activation.shape[-1]:
                        clamped_activation[:, :, neuron_idx] = clamp_value
                     else:
                         print(f"[Warning] Neuron index {neuron_idx} out of bounds for layer {current_layer} activation shape {clamped_activation.shape}. Skipping clamp.")
                except IndexError:
                     # This should theoretically not happen with the bounds check, but defensive coding
                     print(f"[Error] IndexError during clamp hook: L{current_layer}, N{neuron_idx}, Shape{clamped_activation.shape}. Skipping.")


        return clamped_activation

    def _generate_response_thread(self, prompt_text, temperature, top_k, max_new_tokens, current_clamps, use_greedy):
        """The actual generation logic running in a background thread."""
        # Store the passed clamps copy for the hook to use (acts like thread-local storage for this run)
        self.active_clamps_for_generation = current_clamps
        fwd_hooks = [] # Initialize hooks list for this run

        try:
            if self.active_clamps_for_generation:
                # Define hook points based on current clamps
                hook_points = sorted(list(set(f"blocks.{c['layer']}.mlp.hook_post" for c in self.active_clamps_for_generation)))
                fwd_hooks = [(hp, self._mlp_clamp_hook) for hp in hook_points]
                # print(f"[Debug] Applying hooks: {hook_points}") # Debugging

            input_ids = self.tokenizer.encode(prompt_text, return_tensors="pt").to(self.device)
            input_token_count = input_ids.shape[1]
            generated_ids = input_ids # Start with the prompt

            self.model.eval() # Ensure model is in eval mode
            with torch.no_grad(): # Disable gradient calculations
                for i in range(max_new_tokens):
                    # --- Run forward pass with hooks ---
                    # Pass the entire sequence each time.
                    try:
                        # print(f"[Debug] Step {i+1}, Input shape: {generated_ids.shape}") # Debug
                        logits = self.model.run_with_hooks(
                            generated_ids,
                            fwd_hooks=fwd_hooks,
                            reset_hooks_end=False # Keep hooks active for the loop
                        )
                        # print(f"[Debug] Step {i+1}, Logits shape: {logits.shape}") # Debug
                    except Exception as hook_err:
                         tb_str = traceback.format_exc()
                         print(f"Error during hooked forward pass:\n{tb_str}") # Debug
                         # Try to inform user via chat, might fail if UI is unresponsive
                         self._queue_ui_update(self._append_to_chat, f"\n[Error during generation (hook call): {hook_err}]\n", "error")
                         raise # Re-raise to be caught by the outer try/except

                    # Get logits for the *next* token prediction (at the last sequence position)
                    # Logits shape: [batch, seq_len, vocab_size]
                    next_token_logits = logits[0, -1, :] # Shape: [vocab_size]

                    # --- Select Next Token: Greedy or Sampling ---
                    if use_greedy:
                        # Greedy sampling: pick the token with the highest logit
                        next_token_id = torch.argmax(next_token_logits, dim=-1) # Result is 0-dim tensor (scalar index)
                        # print(f"[Debug] Greedy choice: ID={next_token_id.item()}") # Debug
                    else:
                        # Probabilistic sampling (Temp & Top-K)
                        # Apply temperature scaling (Validation ensures temperature > 0)
                        next_token_logits = next_token_logits / temperature

                        # Apply Top-K filtering
                        if top_k > 0:
                            # Ensure k is not larger than vocab size
                            k_actual = min(top_k, next_token_logits.shape[-1])
                            if k_actual < 1 : k_actual = 1 # Safety check

                            top_k_logits, top_k_indices = torch.topk(next_token_logits, k_actual)
                            # Create a mask, set everything initially to negative infinity
                            mask = torch.full_like(next_token_logits, -float('inf'))
                            # Copy the top-k logits to their original positions in the mask
                            mask.scatter_(dim=-1, index=top_k_indices, src=top_k_logits)
                            next_token_logits = mask # Use the masked logits

                        # Sample the next token from the filtered and scaled distribution
                        probs = torch.softmax(next_token_logits, dim=-1)
                        # Handle potential numerical issues in softmax/multinomial
                        if torch.isnan(probs).any():
                             print("[Warning] NaNs detected in probabilities before multinomial sampling. Falling back to argmax.")
                             next_token_id = torch.argmax(next_token_logits, dim=-1) # Use argmax on logits as fallback
                        else:
                             next_token_id = torch.multinomial(probs, num_samples=1).squeeze() # multinomial returns [1], squeeze to 0-dim
                        # print(f"[Debug] Sampled choice: ID={next_token_id.item()}") # Debug

                    # --- Append the new token ---
                    # `next_token_id` is now a 0-dim tensor (scalar). Unsqueeze twice to make it [1, 1]
                    # Shape needs to be [batch_size, sequence_pos] = [1, 1] to concatenate with generated_ids [1, seq_len]
                    next_token_id_tensor = next_token_id.unsqueeze(0).unsqueeze(0)
                    generated_ids = torch.cat([generated_ids, next_token_id_tensor], dim=1)

                    # Check for EOS token
                    if next_token_id.item() == self.tokenizer.eos_token_id:
                        self._queue_ui_update(self._update_status, "Generation complete (EOS).")
                        break
                else: # Loop finished without hitting EOS or error
                    self._queue_ui_update(self._update_status, "Generation complete (max tokens).")


            # Decode the generated part (excluding the prompt)
            output_ids = generated_ids[0, input_token_count:]
            # Use skip_special_tokens=True to avoid printing EOS, etc.
            output_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            output_token_count = len(output_ids)

            # Ensure update happens even if output_text is empty
            self._queue_ui_update(self._append_to_chat, f"Model: {output_text}\n" if output_text else "Model: [No output generated]\n", "model")
            self._queue_ui_update(self._update_token_stats, input_token_count, output_token_count)

        except Exception as e:
            # Log the full error for debugging
            traceback_str = traceback.format_exc()
            print(f"Generation Error:\n{traceback_str}") # Print full traceback to console
            self._queue_ui_update(self._update_status, f"Generation failed: {e}")
            # Append error details to chat
            self._queue_ui_update(self._append_to_chat, f"\n[Error during generation: {e}]\nCheck console for full traceback.\n", "error")
            # Don't show full traceback in messagebox, keep it simple
            self._queue_ui_update(messagebox.showerror, "Generation Error", f"An error occurred during generation:\n{e}\n\nCheck console for details.")

        finally:
            # --- Crucial: Reset hooks regardless of success or failure ---
            # print("[Debug] Resetting hooks.") # Debug
            # Check if model exists before resetting hooks (e.g., if loading failed)
            if self.model:
                self.model.reset_hooks(including_permanent=False) # Only remove temporary hooks added by run_with_hooks

            self.active_clamps_for_generation = [] # Clear thread-local clamps for this run
            self.is_generating = False
            # Re-enable generate button and sampling parameter controls via UI queue
            self._queue_ui_update(self._set_generate_button_state, tk.NORMAL)
            self._queue_ui_update(self._toggle_sampling_params) # This will correctly set state based on greedy_var
            self._queue_ui_update(lambda: self.greedy_checkbox.config(state=tk.NORMAL)) # Also re-enable greedy checkbox


# --- Main Execution ---
if __name__ == "__main__":
    root = tk.Tk()
    app = Gpt2AllTokenApp(root)

    # Optional: Add a protocol handler for safe closing
    def on_closing():
        # Potentially add cleanup here if needed (e.g., stopping threads)
        # For now, just destroy the window
        print("Closing application.")
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)

    root.mainloop()