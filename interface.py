import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import json
from pathlib import Path

from local_llm import LocalLLM


class LLMApp:

    def __init__(self, model_folder="./llm_model"):
        self.model_folder = Path(model_folder)
        self.llm = LocalLLM()
        self.models = []
        self.loaded_model_name = None

        self.root = tk.Tk()
        self.root.title("Local LLM")
        self.root.geometry("1200x700")

        self.create_ui()
        self.find_models()


    def create_ui(self):

        # -------------------------
        # Top controls
        # -------------------------

        top_frame = ttk.Frame(self.root, padding=10)
        top_frame.pack(fill="x")

        ttk.Label(top_frame, text="Model:").pack(side="left")

        self.model_var = tk.StringVar()

        self.model_selector = ttk.Combobox(top_frame, textvariable=self.model_var, state="readonly", width=35)
        self.model_selector.pack(side="left", padx=10)

        self.load_button = ttk.Button(top_frame, text="Load Model", command=self.load_selected_model)
        self.load_button.pack(side="left")

        self.unload_button = ttk.Button(top_frame, text="Unload Model", command=self.unload_model)
        self.unload_button.pack(side="left", padx=(10, 0))

        self.refresh_button = ttk.Button(top_frame, text="Refresh Models", command=self.find_models)
        self.refresh_button.pack(side="left", padx=(10, 0))

        self.clear_button = ttk.Button(top_frame, text="Clear Chat", command=self.clear_chat)
        self.clear_button.pack(side="left", padx=(10, 0))

        ttk.Label(top_frame, text="Mode:").pack(side="left", padx=(15, 5))

        self.mode_var = tk.StringVar(value="Stream")

        self.mode_selector = ttk.Combobox(top_frame, textvariable=self.mode_var, values=["Stream", "Chat"], state="readonly", width=8)
        self.mode_selector.pack(side="left")


        # -------------------------
        # Status
        # -------------------------

        self.status_var = tk.StringVar(value="No model loaded")

        ttk.Label(self.root, textvariable=self.status_var, padding=(10, 0)).pack(fill="x")


        # -------------------------
        # Chat + LLM Output
        # -------------------------

        content_frame = ttk.Frame(self.root)
        content_frame.pack(fill="both", expand=True, padx=10, pady=10)

        chat_frame = ttk.LabelFrame(content_frame, text="Chat", padding=5)
        chat_frame.pack(side="left", fill="both", expand=True)

        self.chat = scrolledtext.ScrolledText(chat_frame, wrap=tk.WORD, font=("Segoe UI", 11), state="disabled")
        self.chat.pack(fill="both", expand=True)

        output_frame = ttk.LabelFrame(content_frame, text="LLM Output", padding=5)
        output_frame.pack(side="right", fill="both", padx=(10, 0))

        self.llm_output = scrolledtext.ScrolledText(output_frame, width=50, wrap=tk.WORD, font=("Consolas", 9), state="disabled")
        self.llm_output.pack(fill="both", expand=True)


        # -------------------------
        # Input
        # -------------------------

        input_frame = ttk.Frame(self.root, padding=10)
        input_frame.pack(fill="x")

        self.input_box = tk.Text(input_frame, height=3, font=("Segoe UI", 11))
        self.input_box.pack(side="left", fill="x", expand=True)

        self.send_button = ttk.Button(input_frame, text="Send", command=self.send_message)
        self.send_button.pack(side="left", padx=(10, 0))

        self.input_box.bind("<Return>", self.on_enter)


    # ------------------------------------------------
    # Models
    # ------------------------------------------------

    def find_models(self):
        self.model_folder.mkdir(parents=True, exist_ok=True)

        current_model = self.model_var.get()

        self.models = sorted(self.model_folder.glob("*.gguf"))
        model_names = [model.name for model in self.models]

        self.model_selector["values"] = model_names

        if model_names:
            if current_model in model_names:
                self.model_var.set(current_model)
            else:
                self.model_selector.current(0)

            self.status_var.set(f"{len(model_names)} model(s) found")

        else:
            self.model_var.set("")
            self.status_var.set("No .gguf models found")


    def load_selected_model(self):
        selected = self.model_var.get()

        if not selected:
            messagebox.showwarning("Model", "Select a model first.")
            return

        model_path = self.model_folder / selected

        self.status_var.set(f"Loading {selected}...")
        self.load_button.config(state="disabled")

        thread = threading.Thread(target=self._load_model_worker, args=(model_path, selected), daemon=True)
        thread.start()


    def _load_model_worker(self, model_path, model_name):
        try:
            if self.llm.llm is not None:
                old_model_name = self.loaded_model_name
                self.llm.unload_model()

                if old_model_name:
                    self.root.after(0, lambda name=old_model_name: self.add_chat_text(f"\nModel unloaded: {name}\n\n"))

            self.llm.load_model(str(model_path), model_name)

            self.root.after(0, lambda name=model_name: self._model_loaded(name))

        except Exception as error:
            error_message = str(error)
            print("REAL MODEL ERROR:", repr(error))
            self.root.after(0, lambda message=error_message: self._show_error(message))


    def _model_loaded(self, model_name):
        self.loaded_model_name = model_name
        self.status_var.set(f"Loaded: {model_name} | GPU")
        self.load_button.config(state="normal")
        self.add_chat_text(f"\nModel loaded: {model_name}\n\n")


    # ------------------------------------------------
    # Unload Model
    # ------------------------------------------------

    def unload_model(self):
        if self.llm.llm is None:
            return

        model_name = self.loaded_model_name

        self.status_var.set("Unloading model...")
        self.load_button.config(state="disabled")

        thread = threading.Thread(target=self._unload_model_worker, args=(model_name,), daemon=True)
        thread.start()


    def _unload_model_worker(self, model_name):
        try:
            self.llm.unload_model()
            self.root.after(0, lambda name=model_name: self._model_unloaded(name))

        except Exception as error:
            error_message = str(error)
            print("REAL UNLOAD ERROR:", repr(error))
            self.root.after(0, lambda message=error_message: self._show_error(message))


    def _model_unloaded(self, model_name):
        self.loaded_model_name = None
        self.status_var.set("No model loaded")
        self.load_button.config(state="normal")
        self.add_chat_text(f"\nModel unloaded: {model_name}\n\n")


    # ------------------------------------------------
    # Chat
    # ------------------------------------------------

    def on_enter(self, event):
        self.send_message()
        return "break"


    def send_message(self):
        if self.llm.llm is None:
            messagebox.showwarning("LLM", "Load a model first.")
            return

        message = self.input_box.get("1.0", "end").strip()

        if not message:
            return

        self.input_box.delete("1.0", "end")

        mode = self.mode_var.get()

        self.add_chat_text(f"You: {message}\n\n")
        self.add_chat_text(f"LLM ({mode}): ")

        self.clear_llm_output()

        self.send_button.config(state="disabled")
        self.mode_selector.config(state="disabled")

        thread = threading.Thread(target=self._chat_worker, args=(message, mode), daemon=True)
        thread.start()


    def _chat_worker(self, message, mode):
        try:
            if mode == "Stream":
                self._stream_worker(message)

            else:
                self._normal_chat_worker(message)

        except Exception as error:
            error_message = str(error)
            print("REAL CHAT ERROR:", repr(error))
            self.root.after(0, lambda message=error_message: self._show_error(message))


    # ------------------------------------------------
    # Streaming Mode
    # ------------------------------------------------

    def _stream_worker(self, message):
        for event in self.llm.stream_chat(message):

            if event["type"] == "token":
                token = event["content"]
                self.root.after(0, lambda text=token: self.add_chat_text(text))

            elif event["type"] == "stats":
                tokens = event["tokens"]
                speed = event["tokens_per_second"]
                self.root.after(0, lambda t=tokens, s=speed: self.finish_response(t, s))

            elif event["type"] == "raw_output":
                raw_output = json.dumps(event["data"], indent=2, ensure_ascii=False)
                self.root.after(0, lambda data=raw_output: self.show_llm_output(data))


    # ------------------------------------------------
    # Normal Chat Mode
    # ------------------------------------------------

    def _normal_chat_worker(self, message):
        result = self.llm.chat(message)

        assistant_message = result["content"]
        response = result["response"]
        tokens = result["tokens"]
        speed = result["tokens_per_second"]

        raw_output = json.dumps(response, indent=2, ensure_ascii=False)

        self.root.after(0, lambda text=assistant_message: self.add_chat_text(text))
        self.root.after(0, lambda data=raw_output: self.show_llm_output(data))
        self.root.after(0, lambda t=tokens, s=speed: self.finish_response(t, s))


    # ------------------------------------------------
    # Finish Response
    # ------------------------------------------------

    def finish_response(self, tokens, speed):
        self.add_chat_text(f"\n\n{tokens} tokens | {speed:.2f} tok/s\n\n")

        self.send_button.config(state="normal")
        self.mode_selector.config(state="readonly")

        self.input_box.focus()


    # ------------------------------------------------
    # LLM Output
    # ------------------------------------------------

    def show_llm_output(self, data):
        self.llm_output.config(state="normal")
        self.llm_output.delete("1.0", "end")
        self.llm_output.insert("end", data)
        self.llm_output.see("1.0")
        self.llm_output.config(state="disabled")


    def clear_llm_output(self):
        self.llm_output.config(state="normal")
        self.llm_output.delete("1.0", "end")
        self.llm_output.config(state="disabled")


    # ------------------------------------------------
    # Helpers
    # ------------------------------------------------

    def add_chat_text(self, text):
        self.chat.config(state="normal")
        self.chat.insert("end", text)
        self.chat.see("end")
        self.chat.config(state="disabled")


    def clear_chat(self):
        self.llm.clear_history()

        self.chat.config(state="normal")
        self.chat.delete("1.0", "end")
        self.chat.config(state="disabled")

        self.clear_llm_output()


    def _show_error(self, error):
        self.load_button.config(state="normal")
        self.send_button.config(state="normal")
        self.mode_selector.config(state="readonly")

        self.status_var.set("Error")

        messagebox.showerror("Error", error)


    # ------------------------------------------------
    # Start Application
    # ------------------------------------------------

    def run(self):
        self.root.mainloop()