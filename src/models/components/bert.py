import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch import einsum

def exists(val):
    return val is not None

class GlobalLinearSelfAttention(nn.Module):
    def __init__(self,
                 dim, 
                 dim_head, 
                 heads): 
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def rope_encoding(self, seq_len, dim_head):
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim_head, 2).float() / dim_head))
        positions = torch.arange(0, seq_len).float()
        sinusoid_inp = torch.einsum('i,j -> ij', positions, inv_freq)
        return torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)

    def forward(self, feats, mask=None):
        h = self.heads
        q, k, v = self.to_qkv(feats).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        seq_len = q.size(2) 
        rope = self.rope_encoding(seq_len, q.size(-1)).to(feats.device)
        q = q + rope[:seq_len, :]
        k = k + rope[:seq_len, :]

        if exists(mask):
            mask = rearrange(mask, 'b n -> b () n ()')
            k = k.masked_fill(~mask, -torch.finfo(k.dtype).max)

        q = q.softmax(dim=-1)
        k = k.softmax(dim=-2)

        q = q * self.scale

        if exists(mask):
            v = v.masked_fill(~mask, 0.)

        context = einsum('b h n d, b h n e -> b h d e', k, v)
        out = einsum('b h d e, b h n d -> b h n e', context, q)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out), q @ k.transpose(-2, -1) * self.scale


class Layer(nn.Module):
    def __init__(self,
                 dim, 
                 narrow_conv_kernel=9, 
                 wide_conv_kernel=9, 
                 wide_conv_dilation=5, 
                 attn_heads=8, 
                 attn_dim_head=64, 
                 local_self_attn=True):
        
        super().__init__()
        self.seq_self_attn = GlobalLinearSelfAttention(dim=dim, dim_head=attn_dim_head, heads=attn_heads) if local_self_attn else None

        conv_mult = 1
        self.narrow_conv = nn.Sequential(
            nn.Conv1d(dim, dim * conv_mult, narrow_conv_kernel, padding=narrow_conv_kernel // 2),
            nn.GELU()
        )

        wide_conv_padding = (wide_conv_kernel + (wide_conv_kernel - 1) * (wide_conv_dilation - 1)) // 2

        self.wide_conv = nn.Sequential(
            nn.Conv1d(dim, dim * conv_mult, wide_conv_kernel, dilation=wide_conv_dilation, padding=wide_conv_padding),
            nn.GELU()
        )

        self.local_norm = nn.LayerNorm(dim)

        self.local_feedforward = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.LayerNorm(dim)
        )

    def forward(self, tokens, mask=None, output_attentions=False):
        if exists(self.seq_self_attn):
            tokens, local_attn_weights = self.seq_self_attn(tokens, mask=mask)
        else:
            local_attn_weights = None
        conv_input = rearrange(tokens, 'b n d -> b d n')

        if exists(mask):
            conv_input_mask = rearrange(mask, 'b n -> b () n')
            conv_input = conv_input.masked_fill(~conv_input_mask, 0.)

        narrow_out = self.narrow_conv(conv_input)
        narrow_out = rearrange(narrow_out, 'b d n -> b n d')
        wide_out = self.wide_conv(conv_input)
        wide_out = rearrange(wide_out, 'b d n -> b n d')

        tokens = tokens + narrow_out + wide_out
        tokens = self.local_norm(tokens)
        tokens = self.local_feedforward(tokens)

        if output_attentions:
            return tokens, local_attn_weights
        return tokens, None

class ProteinBERT(nn.Module):
    def __init__(self,
                 num_tokens=23, 
                 dim=512, 
                 depth=6, 
                 narrow_conv_kernel=9, 
                 wide_conv_kernel=9, 
                 wide_conv_dilation=5, 
                 attn_heads=8, 
                 attn_dim_head=64, 
                 local_self_attn=True, 
                 output_attentions=True, 
                 output_hidden_states=False):
        
        super().__init__()
        self.num_tokens = num_tokens
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states

        self.layers = nn.ModuleList([
            Layer(
                dim=dim,
                narrow_conv_kernel=narrow_conv_kernel,
                wide_conv_dilation=wide_conv_dilation,
                wide_conv_kernel=wide_conv_kernel,
                attn_heads=attn_heads,
                attn_dim_head=attn_dim_head,
                local_self_attn=local_self_attn
            ) for _ in range(depth)
        ])

    def forward(self, seq, mask=None):
        if seq.dim() == 3:  # assuming one-hot input
            seq = torch.argmax(seq, dim=-1) 

        if mask is not None and mask.dtype == torch.float:
            mask = mask.bool()
            
        tokens = self.token_emb(seq)

        all_local_attn_weights = []
        all_hidden_states = []

        for layer in self.layers:
            if self.output_hidden_states:
                all_hidden_states.append(tokens)

            tokens, local_attn_weights = layer(tokens, mask=mask, output_attentions=self.output_attentions)
            
            if self.output_attentions:
                all_local_attn_weights.append(local_attn_weights)

        outputs = (tokens,)  # [batch_size, seq_len, num_tokens]
        
        if self.output_hidden_states:
            all_hidden_states.append(tokens)
            outputs = outputs + (torch.stack(all_hidden_states, dim=0),)  # [num_layers+1, batch_size, seq_len, hidden_dim]

        if self.output_attentions:
            outputs = outputs + (torch.stack(all_local_attn_weights, dim=0),)  # [num_layers, batch_size, num_heads, seq_len, seq_len]
            
        return outputs
