import json
import numpy as np
from model import Model
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
from data.processed.data_tokenized import combined_X, combined_Y

current_file = Path(__file__).resolve()

config_dir = current_file.parent.parent / "config" / "config.json"
with open(config_dir, 'r') as f:
    config = json.load(f)

vocab_size = config['vocab_size']
max_seq_len = config['max_seq_len']
d_model = config['d_model']
num_heads = config['num_heads']
num_layers = config['num_layers']
ff_dim = config['ff_dim']
dropout = config['dropout']
epochs = config['epochs']

# ================
#    Huấn luyện
# ================
model = Model(vocab_size, max_seq_len, d_model, num_heads, num_layers, ff_dim, dropout)
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")

for epoch in range(epochs):
    loss = model.train_on_batch(combined_X, combined_Y)
    if epoch == 0:
        print("╔════════════════════════════════════════════════════╗")
        print("║                 BẮT ĐẦU PRE-TRAIN                  ║")
        print("╠════════════════════════════════════════════════════╣")
    if epoch % 100 == 0 or epoch == epochs - 1:
        print(f"║  [Pretrain] Epoch: {epoch:4d}, Loss: {loss:.4f}              ║")
    if epoch == epochs - 1:
        print("╚════════════════════════════════════════════════════╝")

model_folder = project_root / "model"
model_folder.mkdir(parents=True, exist_ok=True)
model.save(model_folder / "s_a_i.keras")
