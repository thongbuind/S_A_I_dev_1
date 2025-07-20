import tensorflow as tf
from keras import layers, models

class RelativePositionalEmbedding(layers.Layer):
    def __init__(self, d_model, max_relative_distance=128, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.max_relative_distance = max_relative_distance
        
        self.relative_embeddings = layers.Embedding(
            input_dim=2 * max_relative_distance + 1,
            output_dim=d_model,
            name="relative_embedding"
        )

    def call(self, inputs):
        # Giả sử inputs là tensor đầu vào với kích thước [batch_size, seq_len]
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        
        # Tạo ma trận vị trí tương đối
        positions = tf.range(seq_len, dtype=tf.int32)
        relative_positions = positions[:, None] - positions[None, :]
        relative_positions = tf.clip_by_value(
            relative_positions,
            -self.max_relative_distance,
            self.max_relative_distance
        )
        relative_positions = relative_positions + self.max_relative_distance
        
        # Tạo embedding vị trí tương đối
        relative_emb = self.relative_embeddings(relative_positions)  # [seq_len, seq_len, d_model]
        
        # Mở rộng để khớp với batch_size
        relative_emb = tf.expand_dims(relative_emb, 0)  # [1, seq_len, seq_len, d_model]
        relative_emb = tf.tile(relative_emb, [batch_size, 1, 1, 1])  # [batch_size, seq_len, seq_len, d_model]
        
        # Chuyển đổi để phù hợp với kích thước [batch_size, seq_len, d_model]
        # Lấy embedding vị trí tương ứng cho mỗi token
        relative_emb = tf.reduce_sum(relative_emb, axis=2)  # Sum hoặc lấy trung bình nếu cần
        # Hoặc có thể chọn cách khác để giảm chiều, tùy thuộc vào thiết kế mô hình
        
        return relative_emb

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "max_relative_distance": self.max_relative_distance,
        })
        return config

class DecoderBlock(layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, dropout, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout

        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(d_model),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)

    def call(self, x, training=False):
        attn_output = self.mha(x, x, x, use_causal_mask=True)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout": self.dropout
        })
        return config
    
class Model(models.Model):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, ff_dim, dropout, max_relative_distance=512, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.max_relative_distance = max_relative_distance

        self.token_embedding = layers.Embedding(input_dim=vocab_size, output_dim=d_model, mask_zero=True)
        self.pos_embedding = RelativePositionalEmbedding(d_model=d_model, max_relative_distance=max_relative_distance)

        self.decoder_blocks = [
            DecoderBlock(d_model, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ]
        self.dropout_layer = layers.Dropout(dropout)
        self.final_layer = layers.Dense(vocab_size, activation="softmax")

    def call(self, inputs, training=False):
        x = self.token_embedding(inputs)  # [batch_size, seq_len, d_model]
        x = x + self.pos_embedding(inputs)  # Truyền inputs thay vì seq_len
        x = self.dropout_layer(x, training=training)

        for block in self.decoder_blocks:
            x = block(x, training=training)

        return self.final_layer(x)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "ff_dim": self.ff_dim,
            "dropout": self.dropout,
            "max_relative_distance": self.max_relative_distance,
        })
        return config
    