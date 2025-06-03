import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from custom_flashatten import custom_flash_attention_forward, custom_tiled_attention_NO_ONLINE_softmax

# Attempt to import flash_attn
try:
    from flash_attn import flash_attn_func
    # from flash_attn.modules.mha import MHA # Another way to use it for full MHA
    flash_attn_available = True
    print("flash_attn library imported successfully.")
except ImportError:
    flash_attn_available = False
    FlashAttention = None # Placeholder
    print("flash_attn library not found. Install it for full comparison (requires CUDA).")
    print("See: https://github.com/Dao-AILab/flash-attention")


class StandardAttentionModel(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, q, k, v):
        """
        Args:
            q: Query tensor of shape [B, num_heads, N, head_dim]
            k: Key tensor of shape [B, num_heads, N, head_dim]
            v: Value tensor of shape [B, num_heads, N, head_dim]
        Returns:
            out: Output tensor of shape [B, num_heads, N, head_dim]
        """
        B, num_heads, N, head_dim = q.shape

        # Compute attention scores (global attention)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)  # [B, num_heads, N, N]
        attn_weights = F.softmax(scores, dim=-1)  # [B, num_heads, N, N]

        # Apply attention weights to values
        out = torch.matmul(attn_weights, v)  # [B, num_heads, N, head_dim]

        return out, attn_weights

# 2. FlashAttention Library Wrapper (if available)
class FlashAttentionLibModel(nn.Module):
    def __init__(self, num_heads, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'), causal=False):
        super().__init__()
        self.num_heads = num_heads
        self.causal = causal

        if flash_attn_available and device.type == 'cuda':
            self.flash_attn_func = flash_attn_func
        else:
            self.flash_attn_func = None
            print("FlashAttentionLibModel requires flash_attn and CUDA. Will be skipped or use placeholder.")

    def forward(self, q, k, v):
        """
        Args:
            q: Query tensor of shape [B, num_heads, N, head_dim]
            k: Key tensor of shape [B, num_heads, N, head_dim]
            v: Value tensor of shape [B, num_heads, N, head_dim]
        Returns:
            out: Output tensor of shape [B, num_heads, N, head_dim]
        """
        if not self.flash_attn_func or q.device.type != 'cuda':
            print("Skipping FlashAttentionLibModel forward pass (not available/not on CUDA).")
            return q.clone()  # Placeholder output

        B, num_heads, N, head_dim = q.shape
        q = q.permute(0, 2, 1, 3).contiguous().to(torch.float16)
        k = k.permute(0, 2, 1, 3).contiguous().to(torch.float16)
        v = v.permute(0, 2, 1, 3).contiguous().to(torch.float16)

        # Apply flash attention
        out = self.flash_attn_func(q, k, v, causal=self.causal) 

        # Reshape back to original format
        out = out.transpose(1, 2).contiguous().to(torch.float32)  # [B, num_heads, N, head_dim]

        return out, None


# 3. Student's Custom C++ Flash Attention (Forward Only)
class CustomFlashAttentionPyCore(nn.Module):
    """ Custom Python Flash-like Attention - No Projections """
    def __init__(self, num_heads, Q_tile_size, K_tile_size, causal=False): # embed_dim not needed
        super().__init__()
        self.num_heads = num_heads
        self.K_tile_size = K_tile_size
        self.Q_tile_size = Q_tile_size
        self.causal = causal

    def forward(self, q, k, v): # q,k,v: (B, H, N, D_head)
        output, _ = custom_flash_attention_forward(q, k, v, self.Q_tile_size ,self.K_tile_size, causal=self.causal)
        return output, None # Not returning weights for consistency with flash lib

class CustomTiledAttentionNoOnlineSoftmaxPyCore(nn.Module):
    def __init__(self, num_heads, K_tile_size, Q_tile_size=None, causal=False):
        super().__init__()
        self.num_heads = num_heads
        self.K_tile_size = K_tile_size
        self.Q_tile_size = Q_tile_size if Q_tile_size is not None else 1024 # Default Q tile size
        self.causal = causal

    def forward(self, q, k, v): # q,k,v: (B, H, N, D_head)
        output, _ = custom_tiled_attention_NO_ONLINE_softmax(
            q, k, v, self.K_tile_size, self.Q_tile_size
        )
        return output, None

# --- Wrapper for tests to include QKV projection and head splitting ---
class AttentionTestWrapper(nn.Module):
    def __init__(self, C_model, num_heads, attention_core_module):
        super().__init__()
        self.C_model = C_model
        self.num_heads = num_heads
        self.head_dim = C_model // num_heads
        assert C_model % num_heads == 0

        self.qkv_proj = nn.Linear(C_model, C_model * 3) # Or separate Q, K, V projs
        self.out_proj = nn.Linear(C_model, C_model)
        self.attention_core = attention_core_module

    def forward(self, x, x_kv=None, x_v=None): # x, x_kv, x_v all (B, N, C_model)
        if x_kv is None: x_kv = x
        if x_v is None: x_v = x_kv # V usually from same source as K

        B, N_q, _ = x.shape

        qkv = self.qkv_proj(x) # (B, N_q, 3 * C_model)
        qkv = qkv.reshape(B, N_q, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4) # (3, B, H, N_q, D_head)
        q, k, v = qkv[0], qkv[1], qkv[2] # Each (B, H, N_q, D_head)

        attn_output_core, _ = self.attention_core(q, k, v) # (B, H, N_q, D_head)
        
        attn_output_core = attn_output_core.permute(0, 2, 1, 3).reshape(B, N_q, self.C_model) # (B, N_q, C_model)
        final_output = self.out_proj(attn_output_core)
        
        return final_output, None # Wrapper doesn't return weights, core might for debugging