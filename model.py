import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

class args:
    dim: int = 1024
    n_layers: int = 8
    n_heads: int = 16 # should be in multiple of 8 for some reason
    hidden_dim: int = 14336
    n_kv_heads: int = 16 # should be in multiple of 8 for some reason
    vocab_size: int = 32000
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    theta: float = 10000.0
    context_size: int = None

    # Needed for KV cache
    max_batch_size: int = 32
    max_seq_len: int = 256

    num_experts: int = 8
    top_k_experts: int = 2

    device: str = "cuda"

class RMSNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.eplsilon = 1e-8
        self.gamma = nn.Parameter(torch.ones(emb_dim))

    def forward(self, x):
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = x / torch.sqrt(var + self.eplsilon)
        return self.gamma * norm_x

class RotaryEmbeddings(torch.nn.Module):
    def __init__(self, head_dim, seq_len, device="cuda", theta=10000):
        super().__init__()
        self.head_dim=head_dim
        self.seq_len=seq_len
        self.device=device
        self.theta=theta
        self.freqs_complex=self.precompute_theta_pos_freq(head_dim,seq_len,device,theta)

    def precompute_theta_pos_freq(self,head_dim,seq,device,theta):
        assert head_dim%2==0 #dimension should be divisible by 2
        #theta_i=10000^(-2(i-1)/dim) for i={1,2,...,dim/2}
        theta_num=torch.arange(0,head_dim,2).float()
        theta=1./(theta**(theta_num/head_dim)).to(device)
        #write the m positions
        m=torch.arange(seq,device=device)
        #multiply m with every theta
        freqs=torch.outer(m,theta).float()
        #convert this into complex form (polar)
        freqs_complex=torch.polar(torch.ones_like(freqs),freqs)
        return freqs_complex

    def rotary_pos_embeds(self, x,device):
        x_complex=torch.view_as_complex(x.float().reshape(*x.shape[:-1],-1,2))
        freqs_complex=self.freqs_complex.unsqueeze(0).unsqueeze(1)
        x_rotated=x_complex*freqs_complex
        x_out=torch.view_as_real(x_rotated)
        x_out=x_out.reshape(*x.shape)
        return x_out.type_as(x).to(device)

    def forward(self,x):
        return self.rotary_pos_embeds(x,self.device)

class MHA(nn.Module):
    def __init__(self, args, qkv_bias = False):
        super(MHA, self).__init__()

        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_heads_q = args.n_heads
        self.n_rep = self.n_heads_q // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.dim = args.dim
        self.device = args.device
        self.W_query = nn.Linear(self.dim, self.head_dim*self.n_heads_q , bias=False).to(args.device)
        self.W_key = nn.Linear(self.dim, self.n_kv_heads*self.head_dim, bias=False).to(args.device)
        self.W_value = nn.Linear(self.dim, self.n_kv_heads*self.head_dim, bias=False).to(args.device)
        self.out_proj = nn.Linear(args.n_heads*self.head_dim, self.dim).to(args.device)

        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim)).to(args.device)
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim)).to(args.device)

        # self.rope = RotaryEmbeddings(head_dim = args.dim, seq_len = args.max_seq_len, device = args.device)

    @staticmethod
    def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
        batch_size, seq_len, n_kv_heads, head_dim = x.shape
        if n_rep == 1:
            return x
        return (
        x[:, :, :, None, :]
        .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
        .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
        )

    def forward(self, x, start_pos=0):
        #(batch, seq_len, dim)
        batch_size, seq_len, _ = x.shape

        self.rope = RotaryEmbeddings(head_dim = args.dim, seq_len = seq_len, device = args.device)
        # Linear transformation  (batch, seq_len, dim) -> (batch, seq_len, n_heads*head_dim) or (n_kv_heads*head_dim)
        xq, xk, xv = self.W_query(x), self.W_key(x), self.W_value(x)

        # Apply RoPE with position offset
        assert self.rope(xq).shape == xq.shape, "RoPE shape is not correct"
        assert self.rope(xk).shape == xk.shape, "RoPE shape is not correct"

        xq = self.rope(xq)
        xk = self.rope(xk)

        # Change shape to (batch_size, seq_len, n_heads, head_dim)
        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)


        # Update KV cache
        self.cache_k[:batch_size, start_pos:start_pos+seq_len] = xk
        self.cache_v[:batch_size, start_pos:start_pos+seq_len] = xv

        # Retrieve cached keys/values including current sequence
        keys = self.cache_k[:batch_size, :start_pos+seq_len]
        values = self.cache_v[:batch_size, :start_pos+seq_len]

        # Repeat KV heads to match Q heads
        keys = self.repeat_kv(keys, self.n_rep)
        values = self.repeat_kv(values, self.n_rep)

        # Transpose for attention computation
        xq = xq.transpose(1, 2)  # (bs, n_heads_q, seq_len, hd)
        keys = keys.transpose(1, 2)  # (bs, n_heads_kv*rep, cache_len, hd)
        values = values.transpose(1, 2)  # (bs, n_heads_kv*rep, cache_len, hd)

        # Compute attention scores
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        # Weighted sum of values
        output = torch.matmul(scores, values)  # (bs, n_heads_q, seq_len, hd)

        # Combine heads and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.out_proj(output).to(self.device)

class FeedForward(nn.Module):
    def __init__(self, args):
        super().__init__()
        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        # Round the hidden_dim to the nearest multiple of the multiple_of parameter
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False).to(args.device)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False).to(args.device)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False).to(args.device)
    def forward(self, x: torch.Tensor):
        # (B, Seq_Len, Dim) --> (B, Seq_Len, Hidden_Dim)
        swish = F.silu(self.w1(x))
        # (B, Seq_Len, Dim) --> (B, Seq_Len, Hidden_Dim)
        x_V = self.w3(x)
        # (B, Seq_Len, Hidden_Dim) * (B, Seq_Len, Hidden_Dim) --> (B, Seq_Len, Hidden_Dim)
        x = swish * x_V
        # (B, Seq_Len, Hidden_Dim) --> (B, Seq_Len, Dim)
        x = self.w2(x)
        return x

class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.mha = MHA(args)
        self.ffn = FeedForward(args)
        self.rmsnorm = RMSNorm(args.dim)
        self.swish = nn.SiLU()

    def forward(self, x: torch.Tensor):
        x_norm = self.rmsnorm(x)
        x_mha = self.mha(x_norm)
        x = x + x_mha

        x_norm_2 = self.rmsnorm(x)
        x_ffn = self.ffn(x_norm_2)
        x = x + x_ffn
        return x

class Model(nn.Module):
    def __init__(self, args, ):
        super().__init__()

        self.encoder = nn.Sequential(*[Encoder(args) for _ in range(args.n_layers)])

        self.embd = nn.Embedding(args.vocab_size, args.dim).to(args.device)
        self.norm = RMSNorm(args.dim)
        self.linear = nn.Linear(args.dim, args.vocab_size).to(args.device)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor):
        x= x.long()
        x = self.embd(x)
        x = self.encoder(x)
        x = self.norm(x)
        x = self.linear(x)
        return x







