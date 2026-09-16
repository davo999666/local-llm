from llama_cpp import Llama
import time
import gc


class LocalLLM:
    def __init__(self):
        self.llm = None
        self.model_name = None
        self.messages = []

    def load_model(self, model_path: str, model_name: str):
        self.unload_model()

        self.model_name = model_name

        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1,
            n_ctx=4096,
            flash_attn=True,
            n_batch=512,
            n_ubatch=256,
            verbose=False,
        )

        self.messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            }
        ]

    def stream_chat(self, user_input: str):
        if self.llm is None:
            raise RuntimeError("No model loaded")

        self.messages.append({
            "role": "user",
            "content": user_input
        })

        assistant_message = ""
        generation_start = None

        stream = self.llm.create_chat_completion(
            messages=self.messages,
            max_tokens=500,
            temperature=0.5,
            stream=True,
        )

        for chunk in stream:
            delta = chunk["choices"][0]["delta"]
            content = delta.get("content")

            if not content:
                continue

            if generation_start is None:
                generation_start = time.perf_counter()

            assistant_message += content

            # Send each generated piece back to GUI
            yield {
                "type": "token",
                "content": content
            }

        generation_end = time.perf_counter()

        output_tokens = len(
            self.llm.tokenize(
                assistant_message.encode("utf-8"),
                add_bos=False,
            )
        )

        generation_time = (
            generation_end - generation_start
            if generation_start is not None
            else 0
        )

        tokens_per_second = (
            output_tokens / generation_time
            if generation_time > 0
            else 0
        )

        self.messages.append({
            "role": "assistant",
            "content": assistant_message
        })

        yield {
            "type": "stats",
            "tokens": output_tokens,
            "tokens_per_second": tokens_per_second,
        }

    def clear_history(self):
        if self.llm:
            self.messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                }
            ]

    def unload_model(self):

        if self.llm is not None:
            del self.llm
            self.llm = None

            gc.collect()

        self.model_name = None
        self.messages = []

    def chat(self, user_input: str):
        if self.llm is None:
            raise RuntimeError("No model loaded")

        # Add user message to conversation history
        self.messages.append({
            "role": "user",
            "content": user_input
        })

        # Generate response
        response = self.llm.create_chat_completion(
            messages=self.messages,
            max_tokens=500,
            temperature=0.5,
            stream=False,
        )

        assistant_message = response["choices"][0]["message"]["content"]

        # Add assistant response to conversation history
        self.messages.append({
            "role": "assistant",
            "content": assistant_message
        })

        return assistant_message