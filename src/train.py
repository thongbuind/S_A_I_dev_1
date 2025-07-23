import json
import numpy as np
import tensorflow as tf
from model import Model
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
from data.processed.data_tokenized import X, Y, lengths

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

def create_dynamic_batch(X, Y, lengths, batch_indices):
    batch_X = [X[i] for i in batch_indices]
    batch_Y = [Y[i] for i in batch_indices]
    batch_lengths = [lengths[i] for i in batch_indices]
    
    max_len_in_batch = max(batch_lengths)
    
    batch_X_padded = tf.keras.preprocessing.sequence.pad_sequences(
        batch_X, maxlen=max_len_in_batch, padding='post'
    )
    batch_Y_padded = tf.keras.preprocessing.sequence.pad_sequences(
        batch_Y, maxlen=max_len_in_batch, padding='post'
    )
    
    return batch_X_padded, batch_Y_padded, batch_lengths

# ================
#    Huấn luyện
# ================
model = Model(vocab_size, d_model, num_heads, num_layers, ff_dim, dropout)
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")

num_samples = len(X)
num_batches = (num_samples + batch_size - 1) // batch_size

for epoch in range(epochs):
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = list(range(start_idx, end_idx))
        
        # Tạo batch với dynamic padding
        batch_X, batch_Y, batch_lengths = create_dynamic_batch(X, Y, lengths, batch_indices)
        
        # Padding batch cuối nếu cần (để đảm bảo batch size cố định nếu yêu cầu)
        if batch_X.shape[0] < batch_size:
            pad_size = batch_size - batch_X.shape[0]
            current_seq_len = batch_X.shape[1]
            batch_X = np.pad(batch_X, [(0, pad_size), (0, 0)], mode='constant', constant_values=0)
            batch_Y = np.pad(batch_Y, [(0, pad_size), (0, 0)], mode='constant', constant_values=0)
        
        # Huấn luyện trên batch
        if epoch == 0 and i == 0:
            print("╔═════════════════════════════════════════╗")
            print("║            BẮT ĐẦU PRE-TRAIN            ║")
            print("╠═════════════════════════════════════════╣")
        loss = model.train_on_batch(batch_X, batch_Y)
        if i == 1 or i == num_batches - 1:
            print(f"║ Epoch: {epoch:2d}, Batch: {i+1:3d}/{num_batches}, Loss: {loss:.4f} ║")
    
    if epoch == epochs - 1:
        print("╚═════════════════════════════════════════╝")

model_folder = project_root / "model"
model_folder.mkdir(parents=True, exist_ok=True)
model.save(model_folder / "s_a_i.keras")
