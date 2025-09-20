# Hugging Face in Action

This repository contains _selected_ code examples and exercises from the book ["Hugging Face in Action"](https://www.manning.com/books/hugging-face-in-action) by Manning Publications.

![Hugging Face in Action(MEAP)](./images/hugging_face_ia.png)

## Setup Differences from the Book

While the book recommends using Anaconda, this project uses **`uv`** instead for several compelling advantages:

### Why `uv` over `conda`?

- **🚀 Speed**: `uv` is significantly faster than conda for package installation and dependency resolution
- **📦 Modern tooling**: Follows modern Python packaging standards with `pyproject.toml`
- **🔒 Better reproducibility**: The `uv.lock` file ensures exact dependency versions across environments
- **💾 Efficient storage**: More disk-efficient than conda's approach
- **🌐 Full PyPI access**: Direct access to the entire Python Package Index
- **⚡ Cargo-inspired workflow**: Familiar commands for developers coming from Rust/modern tooling

## Environment Setup

This project uses Nix for system dependencies and `uv` for Python package management.

### Prerequisite

**Nix**: Install from [nixos.org](https://nixos.org/download.html). Generally speaking, this is optional, but this document doesn't expand on the alternative (for example manual) set-up of all of the prerequisites.

### Getting Started

1. **Clone and enter the repository**:

   ```bash
   git clone https://github.com/vasigorc/hugging_face_ia
   cd hugging_face_ia
   ```

2. **Load the Nix environment**:
   `nix-shell ~/repos/bash-utils/nix/combined.nix`

   > **Note**: The Nix configuration is sourced from [bash-utils/nix](https://github.com/vasigorc/bash-utils/tree/main/nix) which provides Python, `uv`, and necessary C libraries.

3. **Install Python dependencies**:

   ```bash
   uv sync
   ```

4. **Activate the virtual environment**:

   ```bash
   source .venv/bin/activate
   ```

## Project Structure

The repository is organized by book chapters:

```
hugging_face_ia/
├── chapter_01/          # Chapter 1 examples
├── chapter_02/          # Chapter 2 examples
├── chapter_03/          # Chapter 3 examples
├── ...
├── pyproject.toml # Project configuration
├── uv.lock        # Locked dependencies
└── README.md
```

## Key Dependencies

### Core Libraries

- **`transformers>=4.52.4`**: The heart of this book! Hugging Face's transformers library provides state-of-the-art pre-trained models for NLP, computer vision, and audio tasks. This library is what makes working with models like BERT, GPT, and others accessible.

- **`torch>=2.7.1`**: PyTorch serves as the deep learning framework. Notably, Hugging Face has been consistently moving away from TensorFlow toward PyTorch as their primary backend, making PyTorch knowledge essential for modern Hugging Face workflows.

- **`jupyter>=1.1.1`**: Most examples in the book are presented as Jupyter notebooks, allowing for interactive exploration and experimentation with models and datasets.

- **`pandas>=2.3.0`**: Essential for data manipulation and analysis, particularly when working with datasets and model outputs.

- **`huggingface-hub>=0.33.1`**: Provides a way to interact with the Hugging Face Hub, allowing you to download models and datasets, as well as upload your own.

## Running Examples

### Jupyter Notebooks

Most book examples use Jupyter notebooks. To get started:

```bash
# Launch Jupyter Notebook server
jupyter notebook

# Or use JupyterLab for a more modern interface
jupyter lab
[I 2025-09-20 15:20:21.121 ServerApp] jupyterlab | extension was successfully linked.
...
[I 2025-09-20 15:20:21.144 ServerApp] Serving notebooks from local directory: /home/vasile/repos/hugging_face_ia
[I 2025-09-20 15:20:21.144 ServerApp] Jupyter Server 2.15.0 is running at:
[I 2025-09-20 15:20:21.144 ServerApp] http://localhost:8888/lab?token=302f507fddffa1c5eb1b04692acc1ae471b8ec8483d85328
[I 2025-09-20 15:20:21.144 ServerApp]     http://127.0.0.1:8888/lab?token=302f507fddffa1c5eb1b04692acc1ae471b8ec8483d85328
[I 2025-09-20 15:20:21.144 ServerApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
[C 2025-09-20 15:20:21.242 ServerApp]

    To access the server, open this file in a browser:
        file:///home/vasile/.local-nix/share/jupyter/runtime/jpserver-121592-open.html
    Or copy and paste one of these URLs:
        http://localhost:8888/lab?token=302f507fddffa1c5eb1b04692acc1ae471b8ec8483d85328
        http://127.0.0.1:8888/lab?token=302f507fddffa1c5eb1b04692acc1ae471b8ec8483d85328
```

Then navigate to the suggested address in your browser and open the relevant chapter notebook.

⚠️If you are seeing a warning similar to this when starting the lab:

```bash
[W 2025-09-20 19:38:17.256 ServerApp] 404 GET /api/kernels/566b13a2-9f98-4b94-8035-470ba3579650?1758411497255 (127.0.0.1): Kernel does not exist: 566b13a2-9f98-4b94-8035-470ba3579650
```

Consider providing the allow hidden configuration to Jupyter like so:

```bash
# Generate config file if it doesn't exist
# Should create the following file: ~/.jupyter/jupyter_lab_config.py
jupyter lab --generate-config
```

Then add the following line to the created file:

```python
c.FileContentsManager.allow_hidden = True
```

Finally, restart the lab.

### Python Scripts

For standalone Python files, you can run them directly with `uv`:

```bash
# Run a specific Python file
uv run python chapter_02/main.py
__CUDNN VERSION: 8902
__Number CUDA Devices 1
__CUDA Device Name NVIDIA GeForce RTX 4080 Laptop GPU
__CUDA Device Total Memory [GB] 12.462456832

# Or if your virtual environment is activated
python chapter_01/example.py
```

## License

Code examples are based on the "Hugging Face in Action" book by Manning Publications. Please refer to the book's license terms for usage rights.
