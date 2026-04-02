# NanoInstruct - Instruction-Tuned Small Language Model

A **19.8M parameter** instruction-tuned language model built from scratch with PyTorch. NanoInstruct is fine-tuned on the Alpaca dataset and features a modern transformer architecture with RoPE embeddings, SwiGLU activations, and a byte-level tokenizer.

![Model Architecture](https://img.shields.io/badge/Parameters-19.8M-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Framework](https://img.shields.io/badge/Framework-PyTorch-red)

## Features

- **Compact Architecture**: Only 19.8M parameters - lightweight enough for edge deployment
- **Modern Design**: RoPE positional embeddings, SwiGLU feed-forward, Pre-Norm transformer blocks
- **Byte-Level Tokenizer**: Vocab size of 256 (direct byte encoding), no BPE/merge tables
- **Instruction-Tuned**: Fine-tuned on the Alpaca dataset for following instructions
- **Interactive Chat UI**: Streamlit-based interface for real-time conversation

## Model Architecture

| Component | Value |
|-----------|-------|
| Parameters | 19.8M |
| Layers | 8 |
| Attention Heads | 6 |
| Embedding Dim | 396 |
| Hidden Dim (FFN) | 1536 |
| Max Sequence Length | 105 |
| Vocabulary Size | 256 (byte-level) |

### Architecture Highlights

```
SimpleSLM
├── Token Embeddings (256 × 396)
├── TransformerBlock × 8
│   ├── Pre-Norm (LayerNorm)
│   ├── Multi-Head Attention + RoPE
│   └── SwiGLU Feed-Forward Network
├── Final LayerNorm
└── Output Head (weight-tied with embeddings)
```

## Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- Streamlit (for the chat UI)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/Instruction_tuned-SLM.git
cd Instruction_tuned-SLM

# Install dependencies
pip install torch streamlit
```

### Running the Chat UI

```bash
streamlit run app.py
```

The web interface will open at `http://localhost:8501` with a modern, dark-themed chat UI.

## Usage Examples

### Programmatic Inference

```python
import torch
import json
from app import SimpleSLM, SimpleTokenizer

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open("nano_instruct_model/config.json") as f:
    config = json.load(f)

model = SimpleSLM(**config).to(device)
model.load_state_dict(torch.load("nano_instruct_model/model.pt", map_location=device))
model.eval()

tokenizer = SimpleTokenizer()

# Generate response
def generate(instruction, max_tokens=100, temperature=0.7):
    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    input_ids = torch.tensor([tokenizer.encode_prompt(prompt)]).to(device)
    
    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=max_tokens, temperature=temperature)
    
    return tokenizer.decode(output)

# Example
response = generate("Explain neural networks in simple terms")
print(response)
```

### Example Prompts

```
✍️ Writing:     "Write a short poem about the ocean"
🧠 Explain:     "Explain neural networks simply"
📋 List:        "Give me 5 tips for better coding"
🌍 Knowledge:   "What is machine learning?"
```

## Training

The model was trained in a Google Colab environment using the following setup:

### Dataset

- **Source**: [disham993/alpaca-train-validation-test-split](https://huggingface.co/datasets/disham993/alpaca-train-validation-test-split)
- **Train**: 36,401 examples
- **Validation**: 7,801 examples
- **Test**: 7,800 examples

### Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Epochs | 3 |
| Batch Size | 4 |
| Learning Rate | 2e-4 |
| Optimizer | AdamW (weight_decay=0.01) |
| Scheduler | Linear with 10% warmup |
| Max Sequence Length | 105 |

### Training Results

| Epoch | Train Loss | Test Loss |
|-------|------------|-----------|
| 1 | 0.9962 | 0.7127 |
| 2 | 0.6350 | 0.6072 |
| 3 | 0.5100 | 0.5670 |

### Reproduce Training

See `Slm-instruction-tuned.ipynb` for the complete training notebook. Key components:

1. **Data Preparation**: Load Alpaca dataset, format as instruction/response pairs
2. **Tokenization**: Custom byte-level tokenizer (vocab=256)
3. **Model Definition**: Full transformer architecture in PyTorch
4. **Training Loop**: Standard cross-entropy loss with gradient clipping

## Project Structure

```
Instruction_tuned-SLM/
├── app.py                          # Streamlit chat application
├── Slm-instruction-tuned.ipynb     # Training notebook
├── nano_instruct_model/
│   ├── model.pt                    # Model weights
│   ├── config.json                 # Model hyperparameters
│   └── tokenizer_config.json       # Tokenizer configuration
└── README.md
```

## Technical Details

### Rotary Position Embeddings (RoPE)

RoPE encodes positional information by rotating query and key vectors based on their positions:

```python
class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim):
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        # ...
```

### SwiGLU Activation

The feed-forward network uses SwiGLU gating for improved expressivity:

```python
def forward(self, x):
    return self.out(F.silu(self.gate(x)) * self.fc(x))
```

### Byte-Level Tokenization

Instead of BPE or WordPiece, the model uses direct byte encoding:
- Each byte (0-255) maps to a token ID
- No vocabulary lookup tables needed
- Handles any UTF-8 text without OOV tokens

## Limitations

- **Small Context**: 105 token limit restricts conversation length
- **Limited Knowledge**: Trained only on Alpaca data, no general pre-training
- **Basic Tokenization**: Byte-level encoding is simple but less efficient than BPE
- **No Multi-Turn**: Designed for single instruction-response pairs

## Future Improvements

- [ ] Extend context window with sparse attention
- [ ] Add multi-turn conversation training
- [ ] Experiment with larger model variants
- [ ] Quantization for mobile deployment
- [ ] RLHF fine-tuning for improved alignment

## License

MIT License - feel free to use, modify, and distribute.

## Acknowledgments

- **Alpaca Dataset**: Stanford NLP for the instruction-following data
- **PyTorch**: Foundation for the entire implementation
- **RoPE**: Rotary position embeddings from Meta AI
- **SwiGLU**: Gated FFN architecture from PaLM

---

Built with ❤️ using PyTorch and Streamlit
