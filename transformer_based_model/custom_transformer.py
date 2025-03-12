import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model必须是n_heads的整数倍"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        # 线性变换并分头
        def transform(x, linear):
            x = linear(x)  # [B, L, d_model]
            x = x.view(batch_size, -1, self.n_heads, self.d_k)
            return x.transpose(1,2)  # [B, n_heads, L, d_k]

        q = transform(query, self.linear_q)
        k = transform(key, self.linear_k)
        v = transform(value, self.linear_v)

        # 缩放点积注意力
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, v)  # [B, n_heads, L, d_k]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.linear_out(context)
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, src, src_mask=None):
        # 自注意力模块
        attn_out = self.self_attn(src, src, src, src_mask)
        src = self.norm1(src + self.dropout1(attn_out))
        # 前馈模块
        ff_out = self.ff(src)
        src = self.norm2(src + self.dropout2(ff_out))
        return src

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.norm = nn.LayerNorm(encoder_layer.d_model)

    def forward(self, src, mask=None):
        for layer in self.layers:
            src = layer(src, src_mask=mask)
        return self.norm(src)

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # 自注意力模块
        self_attn_out = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = self.norm1(tgt + self.dropout1(self_attn_out))
        # 编码器-解码器注意力模块
        cross_attn_out = self.cross_attn(tgt, memory, memory, memory_mask)
        tgt = self.norm2(tgt + self.dropout2(cross_attn_out))
        # 前馈模块
        ff_out = self.ff(tgt)
        tgt = self.norm3(tgt + self.dropout3(ff_out))
        return tgt

class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])
        self.norm = nn.LayerNorm(decoder_layer.d_model)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask, memory_mask)
        return self.norm(tgt)

class CustomTransformer(nn.Module):
    def __init__(self, d_model=512, n_heads=8, d_ff=2048, num_encoder_layers=6, num_decoder_layers=6, dropout=0.1):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
        decoder_layer = TransformerDecoderLayer(d_model, n_heads, d_ff, dropout)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers)
        self.d_model = d_model

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        memory = self.encoder(src, mask=src_mask)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        return output

# if __name__ == "__main__":
#     batch_size = 2
#     seq_len_src = 10
#     seq_len_tgt = 8
#     d_model = 512
#     src = torch.rand(batch_size, seq_len_src, d_model)
#     tgt = torch.rand(batch_size, seq_len_tgt, d_model)
#     model = CustomTransformer(d_model=d_model)
#     out = model(src, tgt)
#     print("输出形状:", out.shape)
