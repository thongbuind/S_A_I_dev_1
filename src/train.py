import json
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
batch_size = config['batch_size']

# ================
#    Huấn luyện
# ================
model = Model(vocab_size, max_seq_len, d_model, num_heads, num_layers, ff_dim, dropout)
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")

num_samples = combined_X.shape[0]
num_batches = (num_samples + batch_size - 1) // batch_size


for epoch in range(epochs):
    if epoch == 0:
        print("╔═════════════════════════════════════════╗")
        print("║            BẮT ĐẦU PRE-TRAIN            ║")
        print("╠═════════════════════════════════════════╣")
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, num_samples)
        batch_X = combined_X[start_idx:end_idx]
        batch_Y = combined_Y[start_idx:end_idx]
        
        loss = model.train_on_batch(batch_X, batch_Y)
        if i % 100 == 0 or i == num_batches - 1:
            print(f"║  Epoch: {epoch:4d}, Batch: {i+1}/{num_batches}, Loss: {loss:.4f} ║")
    
    if epoch == epochs - 1:
        print("╚═════════════════════════════════════════╝")

model_folder = project_root / "model"
model_folder.mkdir(parents=True, exist_ok=True)
model.save(model_folder / "s_a_i.keras")
