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
    
class RotaryPositionEmbedding(layers.Layer):
    def __init__(self, d_model, max_seq_len=2048, **kwargs):
        super().__init__(**kwargs)
        
        # Validation checks
        if d_model % 2 != 0:
            raise ValueError('d_model must be even for RoPE')
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Pre-compute frequency matrix
        inv_freq = 1.0 / (10000 ** (tf.range(0, d_model, 2, dtype=tf.float32) / d_model))
        self.inv_freq = tf.Variable(inv_freq, trainable=False, name="inv_freq")
        
        # Pre-compute and cache cos/sin frequencies for max_seq_len
        position = tf.range(max_seq_len, dtype=tf.float32)
        freqs = tf.einsum('i,j->ij', position, inv_freq)  # [max_seq_len, d_model//2]
        
        cos_freqs = tf.cos(freqs)
        sin_freqs = tf.sin(freqs)
        
        # Expand and cache
        cos_freqs = tf.repeat(cos_freqs, 2, axis=-1)  # [max_seq_len, d_model]
        sin_freqs = tf.repeat(sin_freqs, 2, axis=-1)  # [max_seq_len, d_model]
        
        self.cos_cached = tf.Variable(cos_freqs, trainable=False, name="cos_freqs")
        self.sin_cached = tf.Variable(sin_freqs, trainable=False, name="sin_freqs")
    
    def call(self, seq_len):
        """
        Args:
            seq_len: Sequence length (scalar tensor or int)
        Returns:
            cos_freqs, sin_freqs: [1, seq_len, d_model]
        """
        # Validation
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len ({seq_len}) exceeds max_seq_len ({self.max_seq_len})")
        
        # Slice cached frequencies
        cos_freqs = self.cos_cached[:seq_len, :]  # [seq_len, d_model]
        sin_freqs = self.sin_cached[:seq_len, :]  # [seq_len, d_model]
        
        # Add batch dimension
        cos_freqs = tf.expand_dims(cos_freqs, 0)  # [1, seq_len, d_model]
        sin_freqs = tf.expand_dims(sin_freqs, 0)  # [1, seq_len, d_model]
        
        return cos_freqs, sin_freqs
    
    def apply_rope(self, x, cos_freqs, sin_freqs):
        """Apply RoPE to input tensor x"""
        # Reshape x để tách cặp dimensions
        x_even = x[..., ::2]  # [batch, seq_len, d_model//2]
        x_odd = x[..., 1::2]  # [batch, seq_len, d_model//2]
        
        cos_half = cos_freqs[..., ::2]  # [1, seq_len, d_model//2]
        sin_half = sin_freqs[..., ::2]  # [1, seq_len, d_model//2]
        
        # Apply rotation
        rotated_x_even = x_even * cos_half - x_odd * sin_half
        rotated_x_odd = x_even * sin_half + x_odd * cos_half
        
        # Interleave back
        rotated_x = tf.stack([rotated_x_even, rotated_x_odd], axis=-1)
        rotated_x = tf.reshape(rotated_x, tf.shape(x))
        
        return rotated_x

    
class MultiHeadAttention(layers.Layer):
    def __init__(self, d_model, num_heads, max_seq_len=2048, **kwargs):
        super().__init__(**kwargs)
        
        if d_model % num_heads != 0:
            raise ValueError('d_model must be divisible by num_heads')
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.max_seq_len = max_seq_len
        
        self.wq = layers.Dense(d_model, name="query")
        self.wk = layers.Dense(d_model, name="key")
        self.wv = layers.Dense(d_model, name="value")
        self.wo = layers.Dense(d_model, name="output")
        
        self.rope = RotaryPositionEmbedding(self.head_dim, max_seq_len)
        
        mask = tf.linalg.band_part(tf.ones((max_seq_len, max_seq_len)), -1, 0)
        mask = tf.where(mask == 0, -1e9, 0.0)
        self.causal_mask = tf.Variable(mask, trainable=False, name="causal_mask")
    
    def call(self, x, mask=None, training=False):
        """
        Args:
            x: Input tensor [batch, seq_len, d_model]
            mask: Custom attention mask (optional)
            training: Training mode flag
        """
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len ({seq_len}) exceeds max_seq_len ({self.max_seq_len})")
        
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        
        q = tf.reshape(q, (batch_size, seq_len, self.num_heads, self.head_dim))
        k = tf.reshape(k, (batch_size, seq_len, self.num_heads, self.head_dim))
        v = tf.reshape(v, (batch_size, seq_len, self.num_heads, self.head_dim))
        
        cos_freqs, sin_freqs = self.rope(seq_len)
        cos_freqs = tf.reshape(cos_freqs, (1, seq_len, 1, self.head_dim))
        sin_freqs = tf.reshape(sin_freqs, (1, seq_len, 1, self.head_dim))
        
        q = self.rope.apply_rope(q, cos_freqs, sin_freqs)
        k = self.rope.apply_rope(k, cos_freqs, sin_freqs)
        
        q = tf.transpose(q, [0, 2, 1, 3])
        k = tf.transpose(k, [0, 2, 1, 3])
        v = tf.transpose(v, [0, 2, 1, 3])
        
        scores = tf.matmul(q, k, transpose_b=True) / tf.sqrt(tf.cast(self.head_dim, tf.float32))
        
        if mask is not None:
            scores += mask
        else:
            causal_mask = self.causal_mask[:seq_len, :seq_len]
            scores += causal_mask
        
        attention_weights = tf.nn.softmax(scores, axis=-1)
        
        if training:
            attention_weights = tf.nn.dropout(attention_weights, rate=0.1)
        
        attention_output = tf.matmul(attention_weights, v)
        
        attention_output = tf.transpose(attention_output, [0, 2, 1, 3])
        attention_output = tf.reshape(attention_output, (batch_size, seq_len, self.d_model))
        
        return self.wo(attention_output)

class DecoderBlock(layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, dropout, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout

        self.mha = MultiHeadAttention(d_model, num_heads)
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
    