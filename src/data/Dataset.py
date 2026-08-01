import torch
import numpy as np
import gc
import json
from pathlib import Path
from utils.utils import log_progress

class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, lengths, indices, loss_masks=None):
        self.X = X
        self.Y = Y
        self.lengths = lengths
        self.indices = indices
        self.loss_masks = loss_masks

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        return (
            self.X[real_idx],
            self.Y[real_idx],
            self.lengths[real_idx],
            self.loss_masks[real_idx] if self.loss_masks is not None else None
        )
    
    @classmethod
    def create_dataloader(cls, X, Y, lengths, batch_size, max_seq_len, num_workers, shuffle, loss_masks=None):
        log_progress(f"Đang tạo dataset từ {len(X)} samples...")
        indices = np.arange(len(X))
        dataset = cls(X, Y, lengths, indices, loss_masks)
        bucket_size = 20
        PAD_ID = 0

        def collate_fn(batch):
            X_batch = [item[0] for item in batch]
            Y_batch = [item[1] for item in batch]
            lm_batch = [item[3] for item in batch]

            max_len = max(len(x) for x in X_batch)
            if max_len > (max_seq_len - (max_seq_len%bucket_size)):
                bucket = max_seq_len
            else:
                bucket = int(np.ceil(max_len / bucket_size) * bucket_size)

            bsz = len(X_batch)
            X_padded = torch.zeros((bsz, bucket), dtype=torch.long)
            Y_padded = torch.zeros((bsz, bucket), dtype=torch.long)

            if loss_masks is not None:
                loss_mask_padded = torch.zeros((bsz, bucket), dtype=torch.float)

            for i, (x, y) in enumerate(zip(X_batch, Y_batch)):
                x_len = len(x)
                y_len = len(y)

                X_padded[i, :x_len] = torch.as_tensor(x, dtype=torch.long)
                Y_padded[i, :y_len] = torch.as_tensor(y, dtype=torch.long)

                if loss_masks is not None:
                    lm = lm_batch[i]
                    lm_len = len(lm)
                    loss_mask_padded[i, :lm_len] = torch.as_tensor(lm, dtype=torch.float)

            attention_mask = (X_padded != PAD_ID).float()

            if loss_masks is None:
                sample_weight = (Y_padded != PAD_ID).float()
            else:
                sample_weight = loss_mask_padded

            return X_padded, Y_padded, sample_weight, attention_mask

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers = True,
            drop_last=False
        )

        log_progress(f"Dataset được tạo với batch_size={batch_size}")
        return dataloader

def split_train_val_test(X, Y, loss_masks, lengths, train_ratio, val_ratio, seed=54):
    total_sample = len(X)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(total_sample)

    train_end = int(total_sample * train_ratio)
    val_end = int(total_sample * (train_ratio + val_ratio))

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    X_train, Y_train, lengths_train = X[train_idx], Y[train_idx], lengths[train_idx]
    X_val, Y_val, lengths_val = X[val_idx], Y[val_idx], lengths[val_idx]
    X_test, Y_test, lengths_test = X[test_idx], Y[test_idx], lengths[test_idx]
    
    if loss_masks is not None:
        mask_train = loss_masks[train_idx]
        mask_val = loss_masks[val_idx]
        mask_test = loss_masks[test_idx]
        
        return (X_train, Y_train, mask_train, lengths_train, 
                X_val, Y_val, mask_val, lengths_val, 
                X_test, Y_test, mask_test, lengths_test)
    else:
        return (X_train, Y_train, None, lengths_train,
                X_val, Y_val, None, lengths_val,
                X_test, Y_test, None, lengths_test)

def _load_single_npz(path, load_mask=False):
    """Bản gốc: đọc 1 file .npz duy nhất."""
    f = np.load(path)
    def reconstruct(name):
        flat, offsets = f[f"{name}_flat"], f[f"{name}_offsets"]
        return np.array([flat[offsets[i]:offsets[i+1]] for i in range(len(offsets)-1)], dtype=object)
    X = reconstruct("X")
    Y = reconstruct("Y")
    lengths = f["lengths"]
    if load_mask:
        M = reconstruct("M")
        f.close()
        return X, Y, M, lengths
    f.close()
    return X, Y, lengths

def _load_manifest_shards(manifest_path, load_mask=False):
    """Đọc manifest.json rồi ghép nhiều shard .npz lại thành 1 dataset logic."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    manifest_dir = Path(manifest_path).parent
    shards = manifest["shards"]

    X_list, Y_list, M_list, L_list = [], [], [], []
    for shard in shards:
        shard_path = manifest_dir / shard["file"]
        if load_mask:
            X_s, Y_s, M_s, L_s = _load_single_npz(shard_path, load_mask=True)
            M_list.append(M_s)
        else:
            X_s, Y_s, L_s = _load_single_npz(shard_path, load_mask=False)
        X_list.append(X_s)
        Y_list.append(Y_s)
        L_list.append(L_s)

    X = np.concatenate(X_list)
    Y = np.concatenate(Y_list)
    lengths = np.concatenate(L_list)

    if load_mask:
        M = np.concatenate(M_list)
        return X, Y, M, lengths
    return X, Y, lengths

def load_npz(path, load_mask=False):
    """
    - path là file .json -> manifest nhiều shard, tự đọc & ghép tất cả.
    - path là file .npz -> đọc 1 file như bản gốc.
    """
    path = Path(path)
    if path.suffix == ".json":
        return _load_manifest_shards(path, load_mask=load_mask)
    return _load_single_npz(path, load_mask=load_mask)

def load_data(data_type, main_data, sub_data=None, seed=54):
    if data_type == "pretrain":
        X, Y, lengths = load_npz(main_data)
        return X, Y, lengths

    elif data_type == "continued_pretrain":
        X_main, Y_main, L_main = load_npz(main_data)

        if sub_data is not None:
            X_sub, Y_sub, L_sub = load_npz(sub_data)

            n_continued = len(X_main)
            n_pretrain_needed = 3 * n_continued
            total_pretrain = len(X_sub)
            
            rng = np.random.default_rng(seed)
            shuffled_indices = rng.permutation(total_pretrain)
            
            n_samples = min(n_pretrain_needed, total_pretrain)
            selected_indices = shuffled_indices[:n_samples]
            
            X_sub_sampled = X_sub[selected_indices]
            Y_sub_sampled = Y_sub[selected_indices]
            L_sub_sampled = L_sub[selected_indices]
            
            del X_sub, Y_sub, L_sub, shuffled_indices, selected_indices
            gc.collect()
            
            X_combined = np.concatenate([X_main, X_sub_sampled])
            Y_combined = np.concatenate([Y_main, Y_sub_sampled])
            L_combined = np.concatenate([L_main, L_sub_sampled])
            
            combined_indices = rng.permutation(len(X_combined))
            X = X_combined[combined_indices]
            Y = Y_combined[combined_indices]
            lengths = L_combined[combined_indices]
            
        else:
            X, Y, lengths = X_main, Y_main, L_main

        return X, Y, lengths

    elif data_type == "sft1":
        X, Y, loss_mask, lengths = load_npz(main_data, load_mask=True)
        return X, Y, loss_mask, lengths
    
    elif data_type == "sft2":
        X_main, Y_main, M_main, L_main = load_npz(main_data, load_mask=True)

        if sub_data is not None:
            X_sub, Y_sub, M_sub, L_sub = load_npz(sub_data, load_mask=True)

            n_sub = len(X_main) * 3
            total_sub = len(X_sub)

            rng = np.random.default_rng(seed)
            shuffled_indices = rng.permutation(total_sub)

            n_samples = min(n_sub, total_sub)
            selected_indices = shuffled_indices[:n_samples]

            X_sub_sampled = X_sub[selected_indices]
            Y_sub_sampled = Y_sub[selected_indices]
            M_sub_sampled = M_sub[selected_indices]
            L_sub_sampled = L_sub[selected_indices]

            del X_sub, Y_sub, M_sub, L_sub, shuffled_indices, selected_indices
            gc.collect()

            X_combined = np.concatenate([X_main, X_sub_sampled])
            Y_combined = np.concatenate([Y_main, Y_sub_sampled])
            M_combined = np.concatenate([M_main, M_sub_sampled])
            L_combined = np.concatenate([L_main, L_sub_sampled])

            combined_indices = rng.permutation(len(X_combined))
            X = X_combined[combined_indices]
            Y = Y_combined[combined_indices]
            loss_mask = M_combined[combined_indices]
            lengths = L_combined[combined_indices]

        else:
            X, Y, loss_mask, lengths = X_main, Y_main, M_main, L_main

        return X, Y, loss_mask, lengths

    else:
        raise ValueError(f"Unknown data_type: {data_type}")
