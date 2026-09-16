# Local LLM

## Documentation

For more information, installation options, troubleshooting, and advanced configuration, see the official `llama-cpp-python` documentation:

- [llama-cpp-python Documentation](https://llama-cpp-python.readthedocs.io/en/latest/)
- [llama-cpp-python GitHub](https://github.com/abetlen/llama-cpp-python)
- [llama-cpp-python PyPI](https://pypi.org/project/llama-cpp-python/)

Run GGUF models locally with `llama-cpp-python` and NVIDIA CUDA.

## Requirements

- Python 3.10–3.12
- NVIDIA GPU + driver
- GGUF model

## 1. Create Virtual Environment

### Windows

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

## 2. Check CUDA

```bash
nvidia-smi
```

Available CUDA wheels:

- `cu118` → 11.8
- `cu121` → 12.1
- `cu122` → 12.2
- `cu123` → 12.3
- `cu124` → 12.4
- `cu125` → 12.5
- `cu130` → 13.0
- `cu132` → 13.2

## 3. Install llama-cpp-python

Example for CUDA 13.2:

```bash
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu132
```

Change `cu132` to your compatible CUDA wheel.

## 4. Verify CUDA

```bash
python -c "from llama_cpp import llama_cpp; print(llama_cpp.llama_print_system_info().decode())"
```

## 5. Add Models

Put `.gguf` models inside:

```text
llm_model/
├── Qwen3.5-9B-Q4_K_M.gguf
├── Ornith-1.5-9B-Q4_K_M.gguf
└── gemma-4-12b-it-Q4_K_S.gguf
```

## 6. GPU Configuration

```python
llm = Llama(
    model_path="./llm_model/model.gguf",
    n_gpu_layers=-1,
    n_ctx=4096,
    verbose=False,
)
```

- `n_gpu_layers=0` → CPU
- `n_gpu_layers=-1` → GPU

## 7. Run

### Windows

```powershell
python main.py
```

### Linux

```bash
python3 main.py
```

Select a model, click **Load Model**, and start chatting.
