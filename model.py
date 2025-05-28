import tensorflow as tf
from keras import layers, models

class Model(models.Model):
    def __init__(self, vocab_size, max_len=64, d_model=64, num_heads=2, num_layers=2, ff_dim=128, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_dim = ff_dim
        self.dropout = dropout

        self.token_embedding = layers.Embedding(input_dim=vocab_size, output_dim=d_model, mask_zero=True)
        self.pos_embedding = layers.Embedding(input_dim=max_len, output_dim=d_model)

        self.decoder_blocks = [
            TransformerDecoderBlock(d_model, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ]
        self.dropout_layer = layers.Dropout(dropout)
        self.final_layer = layers.Dense(vocab_size, activation="softmax")

    def call(self, inputs, training=False):
        seq_len = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        x = self.token_embedding(inputs) + self.pos_embedding(positions)

        x = self.dropout_layer(x, training=training)

        for block in self.decoder_blocks:
            x = block(x, training=training)

        return self.final_layer(x)


class TransformerDecoderBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, dropout):
        super().__init__()
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
    