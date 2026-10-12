# GPT-2 From Scratch

A hands-on reimplementation of GPT-2 in PyTorch, built from the ground up and then verified
against the original OpenAI GPT-2 weights.

The project covers the full loop:

1. Tokenizing raw text into token IDs
2. Building a sliding-window dataset over those IDs
3. Implementing the transformer architecture from scratch (no `nn.Transformer`)
4. Training the model from random initialization on a small corpus
5. Sampling text with greedy, temperature, and top-k decoding
6. Downloading the official GPT-2 124M checkpoint and loading it into the from-scratch model

## Contents

| File | Description |
| --- | --- |
| `gpt2.ipynb` | Main notebook: data pipeline, model, training, generation, weight loading |
| `gpt_download3.py` | Downloads the official GPT-2 checkpoint files and converts the TF checkpoint into a NumPy parameter dict |
| `the-verdict.txt` | Training corpus — *The Verdict* by Elizabeth Gaskell (~20k tokens) |
| `requirements.txt` | Python dependencies |
| `a219f185-211b-4569-9398-2e080e2c5619_1166x1400.jpg` | Reference image used in the project |

## Setup

```bash
pip install -r requirements.txt
```

Dependencies: `numpy`, `torch`, `tensorflow`, `tqdm`, `tiktoken`, `requests`.

TensorFlow is only needed to read the original GPT-2 checkpoint; it is not used by the model itself.
The notebook was developed with Python 3.11.

Open `gpt2.ipynb` and run the cells top to bottom.

## Notebook Walkthrough

### 1. Data loading

`GPTDatasetV1` tokenizes the whole text with the GPT-2 BPE tokenizer (`tiktoken.get_encoding("gpt2")`)
and then uses a sliding window with a configurable `stride` to build overlapping
`(input_chunk, target_chunk)` pairs. Targets are the inputs shifted by one, which is what makes
next-token prediction work.

`create_dataloader_v1` wraps this in a PyTorch `DataLoader`. The text is split 75/25 into
train and validation sets.

### 2. Model architecture

All components are written from scratch in `nn.Module` subclasses:

- **`LayerNorm`** — mean/variance normalization with learnable `scale` and `shift` parameters and
  `eps = 1e-5`.
- **`GELU`** — the tanh approximation of the GELU activation.
- **`MultiHeadAttention`** — scaled dot-product causal self-attention. Q, K, V projections are
  split into heads, the causal mask is stored as a registered buffer (upper triangular), attention
  weights are masked with `-inf`, scaled by `sqrt(head_dim)`, passed through softmax and dropout,
  and finally merged and projected through `out_proj`.
- **`FeedForward`** — `Linear(emb_dim -> 4 * emb_dim)` → GELU → `Linear(4 * emb_dim -> emb_dim)`.
- **`TransformerBlock`** — pre-norm architecture: LayerNorm → attention → dropout → residual,
  then LayerNorm → feed-forward → dropout → residual.
- **`GPTModel`** — token + learned position embeddings, dropout, `n_layers` transformer blocks,
  a final layer norm, and a linear output head (tied to the token embedding weights).

### 3. Configuration

```python
GPT_CONFIG_124M = {
    "vocab_size":    50257,
    "context_length": 1024,
    "emb_dim":        768,
    "n_heads":        12,
    "n_layers":       12,
    "drop_rate":      0.1,
    "qkv_bias":       True,
}
```

### 4. Training

- Loss: cross-entropy over flattened logits and targets.
- Optimizer: `AdamW` with `lr = 4e-4`, `weight_decay = 0.1`.
- `train_model_simple` tracks tokens seen and evaluates every `eval_freq` steps over `eval_iter`
  batches, printing a sample generated from `"Every effort moves you"` at the end of each epoch.
- Run settings: 10 epochs, `batch_size = 2`, sequence length 1024, on CPU.
- Observed run: loss drops from ~9.57 to ~6.81 over 6 epochs (~5.6 minutes on CPU).

### 5. Text generation

Two generators are provided:

- `generate_text_simple` — greedy decoding (`argmax`) used during training to print samples.
- `generate` — full sampler supporting **top-k** filtering (everything below the k-th highest logit
  is masked to `-inf`) and **temperature** scaling (logits divided by temperature, then
  `multinomial` sampling). Also supports an optional `eos_id` to stop early.

Example:

```python
token_ids = generate(
    model=gpt,
    idx=text_to_token_ids("Life is full of", tokenizer).to(device),
    max_new_tokens=25,
    context_size=GPT_CONFIG_124M["context_length"],
    top_k=50,
    temperatue=1.5,
)
```

### 6. Loading the official GPT-2 weights

`gpt_download3.py` provides:

- `download_and_load_gpt2(model_size, models_dir)` — validates the model size against
  `124M / 355M / 774M / 1558M`, downloads the checkpoint, `encoder.json`, `hparams.json` and
  `vocab.bpe` from OpenAI's public blob storage into `models_dir/model_size/`, skips files that
  are already up to date (by comparing sizes), then loads settings and parameters.
- `download_file(url, destination)` — streamed download with a `tqdm` progress bar.
- `load_gpt2_params_from_tf_ckpt(ckpt_path, settings)` — walks the TensorFlow checkpoint variables
  and rebuilds a nested dictionary keyed by layer index, so the OpenAI naming (`wte`, `wpe`,
  `blocks.N.attn.c_attn`, `blocks.N.mlp.c_fc`, `blocks.N.ln_1`, …) becomes an easy-to-navigate dict.

In the notebook, `assign` checks shape compatibility and `load_weights_into_gpt` copies every tensor
into the `GPTModel`:

- `wte` → token embedding and the (tied) output head; `wpe` → position embedding
- `attn/c_attn` weights and biases are split in three for Q, K, V and transposed to PyTorch layout
- `attn/c_proj` → attention output projection
- `mlp/c_fc` and `mlp/c_proj` → the two feed-forward layers
- `ln_1`, `ln_2`, and the final `g`/`b` → LayerNorm scale and shift

The result is that the from-scratch implementation reproduces GPT-2 124M exactly. Generating with
the pretrained weights gives coherent text, e.g.

```
Life is full of new problems at work, when I'm forced to choose the right word to describe
how one finds the person I love and work
```

compared to the near-gibberish produced by the same model trained from scratch for a few epochs.

## Notes

- The notebook imports `tqdm` and `tensorflow` directly in the first cell even though they are only
  used by `gpt_download3.py`.
- `download_file` uses `verify=False`, so `urllib3` prints an `InsecureRequestWarning` on each
  request. Enable certificate verification if that matters in your environment.
- Training runs on CPU (`device = torch.device("cpu")`); switch to `cuda` if available for faster
  epochs.