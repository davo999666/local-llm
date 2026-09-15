import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
from pathlib import Path

from local_llm import LocalLLM


class LLMApp:
    def __init__(self, model_folder="./llm_model"):
        self.model_folder = Path(model_folder)

        self.llm = LocalLLM()
        self.models = []

        self.root = tk.Tk()
        self.root.title("Local LLM")
        self.root.geometry("900x700")

        self.create_ui()
        self.find_models()

    def create_ui(self):
        # -------------------------
        # Model selection
        # -------------------------

        top_frame = ttk.Frame(self.root, padding=10)
        top_frame.pack(fill="x")

        ttk.Label(
            top_frame,
            text="Model:"
        ).pack(side="left")

        self.model_var = tk.StringVar()

        self.model_selector = ttk.Combobox(
            top_frame,
            textvariable=self.model_var,
            state="readonly",
            width=45,
        )

        self.model_selector.pack(
            side="left",
            padx=10
        )

        self.load_button = ttk.Button(
            top_frame,
            text="Load Model",
            command=self.load_selected_model,
        )
        self.refresh_button = ttk.Button(
            top_frame,
            text="Refresh Models",
            command=self.find_models,
        )

        self.refresh_button.pack(
            side="left",
            padx=(10, 0)
        )

        self.load_button.pack(side="left")

        self.clear_button = ttk.Button(
            top_frame,
            text="Clear Chat",
            command=self.clear_chat,
        )

        self.clear_button.pack(
            side="left",
            padx=10
        )

        # -------------------------
        # Status
        # -------------------------

        self.status_var = tk.StringVar(
            value="No model loaded"
        )

        ttk.Label(
            self.root,
            textvariable=self.status_var,
            padding=(10, 0),
        ).pack(
            fill="x"
        )

        # -------------------------
        # Chat
        # -------------------------

        self.chat = scrolledtext.ScrolledText(
            self.root,
            wrap=tk.WORD,
            font=("Segoe UI", 11),
            state="disabled",
        )

        self.chat.pack(
            fill="both",
            expand=True,
            padx=10,
            pady=10,
        )

        # -------------------------
        # Input
        # -------------------------

        input_frame = ttk.Frame(
            self.root,
            padding=10
        )

        input_frame.pack(fill="x")

        self.input_box = tk.Text(
            input_frame,
            height=3,
            font=("Segoe UI", 11),
        )

        self.input_box.pack(
            side="left",
            fill="x",
            expand=True,
        )

        self.send_button = ttk.Button(
            input_frame,
            text="Send",
            command=self.send_message,
        )

        self.send_button.pack(
            side="left",
            padx=(10, 0)
        )

        self.input_box.bind(
            "<Return>",
            self.on_enter
        )

    # ------------------------------------------------
    # Models
    # ------------------------------------------------

    def find_models(self):
        self.model_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        # Remember currently selected model
        current_model = self.model_var.get()

        # Scan folder again
        self.models = sorted(
            self.model_folder.glob("*.gguf")
        )

        model_names = [
            model.name
            for model in self.models
        ]

        # Update dropdown
        self.model_selector["values"] = model_names

        if model_names:

            # Keep current selection if it still exists
            if current_model in model_names:
                self.model_var.set(current_model)
            else:
                self.model_selector.current(0)

            self.status_var.set(
                f"{len(model_names)} model(s) found"
            )

        else:
            self.model_var.set("")

            self.status_var.set(
                "No .gguf models found"
            )

    def load_selected_model(self):
        selected = self.model_var.get()

        if not selected:
            messagebox.showwarning(
                "Model",
                "Select a model first."
            )
            return

        model_path = self.model_folder / selected

        self.status_var.set(
            f"Loading {selected}..."
        )

        self.load_button.config(
            state="disabled"
        )

        thread = threading.Thread(
            target=self._load_model_worker,
            args=(model_path, selected),
            daemon=True,
        )

        thread.start()

    def _load_model_worker(
        self,
        model_path,
        model_name
    ):
        try:
            self.llm.load_model(
                str(model_path),
                model_name
            )

            self.root.after(
                0,
                lambda: self._model_loaded(
                    model_name
                )
            )

        except Exception as error:
            self.root.after(
                0,
                lambda: self._show_error(
                    str(error)
                )
            )

    def _model_loaded(self, model_name):
        self.status_var.set(
            f"Loaded: {model_name} | GPU"
        )

        self.load_button.config(
            state="normal"
        )

        self.add_chat_text(
            f"\nModel loaded: {model_name}\n\n"
        )

    # ------------------------------------------------
    # Chat
    # ------------------------------------------------

    def on_enter(self, event):
        self.send_message()
        return "break"

    def send_message(self):
        if self.llm.llm is None:
            messagebox.showwarning(
                "LLM",
                "Load a model first."
            )
            return

        message = self.input_box.get(
            "1.0",
            "end"
        ).strip()

        if not message:
            return

        self.input_box.delete(
            "1.0",
            "end"
        )

        self.add_chat_text(
            f"You: {message}\n\n"
        )

        self.add_chat_text(
            "LLM: "
        )

        self.send_button.config(
            state="disabled"
        )

        thread = threading.Thread(
            target=self._chat_worker,
            args=(message,),
            daemon=True,
        )

        thread.start()

    def _chat_worker(self, message):
        try:
            for event in self.llm.stream_chat(message):

                if event["type"] == "token":
                    token = event["content"]

                    self.root.after(
                        0,
                        lambda t=token:
                        self.add_chat_text(t)
                    )

                elif event["type"] == "stats":

                    tokens = event["tokens"]
                    speed = event[
                        "tokens_per_second"
                    ]

                    self.root.after(
                        0,
                        lambda:
                        self.finish_response(
                            tokens,
                            speed
                        )
                    )

        except Exception as error:
            self.root.after(
                0,
                lambda: self._show_error(
                    str(error)
                )
            )

    def finish_response(
        self,
        tokens,
        speed
    ):
        self.add_chat_text(
            f"\n\n"
            f"{tokens} tokens | "
            f"{speed:.2f} tok/s\n\n"
        )

        self.send_button.config(
            state="normal"
        )

        self.input_box.focus()

    # ------------------------------------------------
    # Helpers
    # ------------------------------------------------

    def add_chat_text(self, text):
        self.chat.config(
            state="normal"
        )

        self.chat.insert(
            "end",
            text
        )

        self.chat.see("end")

        self.chat.config(
            state="disabled"
        )

    def clear_chat(self):
        self.llm.clear_history()

        self.chat.config(
            state="normal"
        )

        self.chat.delete(
            "1.0",
            "end"
        )

        self.chat.config(
            state="disabled"
        )

    def _show_error(self, error):
        self.load_button.config(
            state="normal"
        )

        self.send_button.config(
            state="normal"
        )

        self.status_var.set(
            "Error"
        )

        messagebox.showerror(
            "Error",
            error
        )

    # ------------------------------------------------
    # Start application
    # ------------------------------------------------

    def run(self):
        self.root.mainloop()