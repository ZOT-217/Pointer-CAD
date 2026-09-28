# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Pointer-CAD (CVPR 2026) is an autoregressive model that generates parametric CAD models from natural language descriptions. It combines a Qwen2.5 LLM backbone (LoRA fine-tuned) with a UV-Net B-Rep encoder to predict CAD operations (extrude, fillet, chamfer) while pointing to specific faces/edges on the intermediate geometry.

## Commands

```bash
# Training (DDP, auto-detects GPUs via nvidia-smi)
bash train.sh

# Testing — first terminal starts the server, then run workers
bash test.sh                              # starts test_server.py if port free
bash test.sh -h <server_host>             # starts GPU workers

# Web demo (Gradio)
python web.py
# or with custom port:
python web.py -p 7860

# Evaluation on saved results
python eval.py -i ./log

# Export JSON models to CAD formats
python preprocessing/json2step.py --input_dir ./data/raw_json --output_dir ./exports/step
python preprocessing/json2stl.py --input_dir ./data/raw_json --output_dir ./exports/stl
```

**Environment variables for distributed training:** `MASTER_ADDR`, `MASTER_PORT`, `WORLD_SIZE`, `NODE_RANK`, `GLOBAL_RANK_OFFSET` (defaults in `train.sh`).

## Architecture

### Three prediction heads on top of Qwen2.5

- **Value head** — predicts the next CAD command token (special tokens like `<|sketch_start|>`, `<|extrude_start|>` etc.) or 8-bit quantized numeric values (angles, distances)
- **Pointer head** — outputs a 128-dim embedding matched via cosine similarity to B-Rep face/edge embeddings for selecting geometric entities
- **Scale head** — predicts overall model scale factor after Softplus activation

### B-Rep encoder (UV-Net)

Defined in `models/brep_embed.py`. Processes the face-adjacency graph:
1. **Curve encoder** (1D conv): 12-channel U-grids of edges (points, tangents, derivatives) → 128-dim embeddings
2. **Surface encoder** (2D conv): 8-channel UV-grids of faces (points, normals, Gaussian curvature, trimming mask) → 128-dim embeddings
3. **Graph encoder**: Multiple rounds of edge/node message passing (`_NodeConv` / `_EdgeConv`) over the face-adjacency DGL graph, then decoded to output embeddings per face/edge

### Tokenizer/Processor (`models/processor.py`)

`PointerCADTokenizer` extends Qwen2Tokenizer with special placeholder tokens (`<|brep_edge_pad|>`, `<|brep_face_pad|>`, `<|cad_pad|>`). `PointerCADProcessor.__call__()` replaces these placeholders in the chat template with the correct number of tokens matching the B-Rep geometry size, then embeds B-Rep features directly into token positions via `masked_scatter`.

### CAD model construction (`cadmodel/`)

`CADModel` is the Python-side CAD representation. Operations are:
- **Extrude** (`cadmodel/extrude.py`) — sketch-based extrusion with profiles, loops, curves
- **Fillet** (`cadmodel/fillet.py`) — edge fillets on existing geometry
- **Chamfer** (`cadmodel/chamfer.py`) — edge chamfers

`CADModel.to_vector()` serializes an operation to a (value, pointer) token vector for training. `CADModel.from_vector()` deserializes model predictions back to CAD geometry using `pythonocc-core`.

### Token vocabulary (`misc.py`)

Special tokens defined in `misc.py:TOKEN`: covers operation types (extrude/chamfer/fillet start), structure tokens (model/part/sketch start), pointer enable/disable, direction specifiers, sketch orientation, and extrusion types. Numeric parameters are 8-bit quantized (values ≥ `len(TOKEN)`).

### Data flow

1. **Dataset** (`dataset/dataset.py`): Loads preprocessed `.pkl` vectors, `.bin` DGL graphs, and text prompts. Supports `abs` (absolute) and `exp` (expression) prompt templates. Augmentation via `dataset/augmentation.py`.
2. **Preprocessing** (`preprocessing/json2vec.py`): Converts raw JSON CAD definitions to token vectors and B-Rep graphs.
3. **Training** (`train.py`): DDP training loop. Chat template formats messages with system prompt + B-Rep tokens + user text + CAD tokens. Model receives text embeddings interleaved with B-Rep embeddings. Loss combines value cross-entropy, pointer cosine-similarity-based loss, and scale MSE.
4. **Inference** (`models/pointercad.py:PointerCAD.predict()`): Autoregressive loop up to `MAX_VECTOR_LENGTH` steps. At each step: if a CAD token is expected, predicts value (argmax or multinomial sampling), and if the previous value was `<|pointer_enable|>`, predicts a pointer via cosine similarity over face/edge embeddings.
5. **Evaluation** (`test.py` / `test_server.py`): Client-server architecture. Server (`test_server.py` FastAPI on port 32500) coordinates workers via `/apply_model_id`, `/report`, `/finish` endpoints. Workers run inference and compute geometry metrics (IoU, Chamfer, F1, watertightness).
6. **Web demo** (`web.py`): Gradio interface. Iteratively calls `model.predict()`, deserializes each step's (value, pointer) vector via `CADModel.from_vector()`, builds the intermediate solid, re-computes the B-Rep graph, and feeds it back for the next step (max 10 parts).

## Configuration

Three YAML config files in `config/`:
- `train.yaml` — model params (base model, quantization bits), dataset paths, training loop settings (batch size, gradient accumulation, LR, epochs), criterion weights, validation schedule
- `test.yaml` — checkpoint path, dataset settings, evaluation batch size
- `web.yaml` — checkpoint path, base model, log directory

## Key dependencies

- `torch` 2.12+, `transformers` 5.8+, `peft` (LoRA), `dgl` 2.5+ (graph neural networks)
- `pythonocc-core` + `occwl` (OpenCASCADE CAD kernel and B-Rep utilities)
- `flash-attn` (CUDA attention kernel)
- `gradio`, `fastapi` + `uvicorn` (web demo and test server)
- `wandb` (logging)
