import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

import torch
import json
import sentencepiece as spm
from src.model.TransformerModel import TransformerModel

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps:0" if torch.backends.mps.is_available()
    else "cpu"
)
config_dir     = project_root / "config"
tokenizer_file = project_root / "src" / "tokenizer" / "tokenizer.model"
model_dir      = project_root / "model"

_tokenizer = None

def _resolve_config(model_name: str) -> dict:
    stem     = Path(model_name).stem
    size_key = stem.rsplit("_", 1)[-1]

    base_file = config_dir / "base.json"
    size_file = config_dir / f"{size_key}.json"

    if not base_file.exists():
        raise FileNotFoundError(f"Config base không tìm thấy: {base_file}")
    if not size_file.exists():
        raise FileNotFoundError(
            f"Config '{size_key}.json' không tìm thấy: {size_file}\n"
            f"Tên model phải có dạng <name>_<size>.pt  (vd: sft_35M.pt)"
        )

    with open(base_file) as f:
        config = json.load(f)
    with open(size_file) as f:
        config.update(json.load(f))

    return config

def load_model(model_file: Path):
    global _tokenizer

    config = _resolve_config(model_file.name)

    if not tokenizer_file.exists():
        raise FileNotFoundError(f"Tokenizer không tìm thấy: {tokenizer_file}")

    _tokenizer = spm.SentencePieceProcessor()
    _tokenizer.load(str(tokenizer_file))

    model = TransformerModel(
        config["vocab_size"],
        config["d_model"],
        config["num_heads"],
        config["num_kv_heads"],
        config["num_layers"],
        config["ff_dim"],
        config["max_seq_len"],
        config["dropout"],
    )

    if not model_file.exists():
        raise FileNotFoundError(f"Model file không tìm thấy: {model_file}")

    state_dict = torch.load(model_file, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded ({model_file.name}). Params: {params:,}")

    return model, _tokenizer

def generate_response(model, tokenizer, user_input):
    tok = tokenizer if tokenizer is not None else _tokenizer
    return model.generate_response(user_input, tok)
