# Blafisk

Blåfisk is a chess bot package that uses [lichess](https://lichess.org/) as an interface and a fine-tuned [LLM](https://huggingface.co/lazy-guy12/chess-llama) to generate moves. It is better than most human players but worse than almost all chess computers.

## Installation

### Prerequisites

- Python 3.8 or higher
- pip

### Setup Instructions

1. **Clone the repository**

   ```bash
   git clone https://github.com/axel-nils/blafisk.git
   cd blafisk
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment**

   On Windows:

   ```bash
   .venv\Scripts\activate
   ```

   On Mac/Linux:

   ```bash
   source .venv/bin/activate
   ```

4. **Install PyTorch**

   With cuda:

   ```bash
   pip install torch==2.9.0+cu130 --index-url https://download.pytorch.org/whl/cu130
   ```

   Without cuda:

   ```bash
   pip install torch==2.9.0
   ```

5. **Install this package**

   ```bash
   pip install -e .
   ```

6. **Set up environment variables**

   Put your API keys in a `.env`-file:

   ```python
   BOT_USERNAME=
   LICHESS_TOKEN=
   ```

Note: On Windows, either run Python as administrator or activate developer mode in the settings to ensure the model is cached efficiently.

## Hardware Requirements

- **Minimum**: 4GB RAM, CPU
- **Recommended**: NVIDIA GPU with 4GB+ VRAM

The neural network uses less than 2GB of VRAM when running with float16 precision on CUDA.

## Acknowledgements

- [lazy-guy12/chess-llama](https://huggingface.co/lazy-guy12/chess-llama): The LLM/Chess Model used to generate moves
- python-chess: For chess board representation and move validation
- Hugging Face Transformers: For model loading and inference
