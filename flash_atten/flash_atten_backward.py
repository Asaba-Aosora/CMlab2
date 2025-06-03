import torch
import torch.nn.functional as F
import math

# Forward pass (implemented by you)
from custom_flashatten import custom_tiled_attention_NO_ONLINE_softmax


def flash_attention_backward(grad_output, query, key, value, K_tile_size, Q_tile_size=None, softmax_scale=None):
    # TODO: Need to implement the backward pass for the tiled attention

    B, H, N_q, D_head = query.shape
    _, _, N_kv, _ = key.shape
    
    if Q_tile_size is None:
        Q_tile_size = K_tile_size
    
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(D_head)
    
    # 初始化梯度张量
    grad_query = torch.zeros_like(query, dtype=torch.float32)
    grad_key = torch.zeros_like(key, dtype=torch.float32)
    grad_value = torch.zeros_like(value, dtype=torch.float32)
    
    # 计算分块索引
    q_tile_indices = [(i * Q_tile_size, min((i + 1) * Q_tile_size, N_q)) 
                      for i in range(math.ceil(N_q / Q_tile_size))]
    
    k_tile_indices = [(j * K_tile_size, min((j + 1) * K_tile_size, N_kv)) 
                      for j in range(math.ceil(N_kv / K_tile_size))]
    
    # 前向传播计算注意力权重（用于反向传播）
    _, attn_weights = custom_tiled_attention_NO_ONLINE_softmax(
        query, key, value, K_tile_size, Q_tile_size, softmax_scale
    )
    
    # 计算中间变量：注意力权重与输出梯度的乘积
    temp = torch.matmul(attn_weights, grad_output)  # (B, H, N_kv, D_head)
    
    # 计算值的梯度
    grad_value = torch.matmul(attn_weights.transpose(-2, -1), grad_output)  # (B, H, N_kv, D_head)
    
    # 遍历查询分块计算查询梯度
    for i_q, (q_start, q_end) in enumerate(q_tile_indices):
        Q_i = query[:, :, q_start:q_end, :]
        current_N_q_tile = Q_i.shape[2]
        
        for j_k, (k_start, k_end) in enumerate(k_tile_indices):
            K_j = key[:, :, k_start:k_end, :]
            V_j = value[:, :, k_start:k_end, :]
            current_N_kv_tile = K_j.shape[2]
            
            # 提取当前分块的注意力权重
            attn_block = attn_weights[:, :, q_start:q_end, k_start:k_end]  # (B, H, Q_tile, K_tile)
            
            # 计算中间项用于查询和键的梯度
            part1 = torch.matmul(grad_output[:, :, q_start:q_end, :], V_j.transpose(-2, -1))  # (B, H, Q_tile, K_tile)
            part2 = torch.matmul(temp[:, :, k_start:k_end, :], Q_i.transpose(-2, -1))  # (B, H, K_tile, Q_tile)
            
            # 更新查询梯度
            grad_query_part = softmax_scale * torch.matmul(part1, K_j)  # (B, H, Q_tile, D_head)
            grad_query[:, :, q_start:q_end, :] += grad_query_part
            
            # 更新键梯度
            grad_key_part = softmax_scale * torch.matmul(part2.transpose(-2, -1), Q_i)  # (B, H, K_tile, D_head)
            grad_key[:, :, k_start:k_end, :] += grad_key_part
    
    return grad_query, grad_key, grad_value


# --- Test ---
def run_test():
    B, H, N_q, N_kv, D = 2, 4, 512, 512, 32 # Using non-multiples for tile sizes to test edge cases
    K_tile = 32
    Q_tile = 32 # Can be different from K_tile

    # Forcing float64 for higher precision in gradient checking
    dtype = torch.float64 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu' # For easier debugging initially

    query = torch.randn(B, H, N_q, D, device=device, dtype=dtype, requires_grad=True)
    key = torch.randn(B, H, N_kv, D, device=device, dtype=dtype, requires_grad=True)
    value = torch.randn(B, H, N_kv, D, device=device, dtype=dtype, requires_grad=True)
    
    # Make copies for custom backward
    query_custom = query.detach().clone().requires_grad_(True)
    key_custom = key.detach().clone().requires_grad_(True)
    value_custom = value.detach().clone().requires_grad_(True)

    print(f"Test parameters: B={B}, H={H}, N_q={N_q}, N_kv={N_kv}, D={D}, K_tile={K_tile}, Q_tile={Q_tile}")
    print(f"Using device: {device}, dtype: {dtype}")

    # --- PyTorch Autograd ---
    # For PyTorch's reference, we can use a non-tiled version if inputs are small enough
    # or a PyTorch native implementation like F.scaled_dot_product_attention
    # However, to ensure we are comparing against the *exact* same forward operations
    # as `custom_tiled_attention_NO_ONLINE_softmax`, we should use it.
    
    # It's important that the forward pass used for autograd is numerically
    # identical to the one for which we are writing the custom backward.
    output_pytorch, _ = custom_tiled_attention_NO_ONLINE_softmax(query, key, value, K_tile, Q_tile)
    
    # Use a dummy grad_output (e.g., from a sum loss or random)
    # To make comparison easier, let's use a fixed random grad_output.
    torch.manual_seed(0) # for reproducibility of grad_output
    grad_output_dummy = torch.randn_like(output_pytorch, device=device, dtype=dtype)
    
    output_pytorch.backward(gradient=grad_output_dummy)
    
    dq_pytorch = query.grad.clone()
    dk_pytorch = key.grad.clone()
    dv_pytorch = value.grad.clone()

    # --- Custom Backward ---
    # We need the output of the forward pass for the custom backward if we were to pass P,
    # but our backward recomputes P. So we only need grad_output and inputs.
    # Note: The `custom_tiled_attention_NO_ONLINE_softmax` is called just to ensure the setup is similar,
    # but its output is not directly used by `flash_attention_backward` beyond shape/dtype.
    # However, grad_output_dummy IS derived from an execution of this function.
    
    # Ensure custom inputs don't have grads from pytorch's run
    if query_custom.grad is not None: query_custom.grad.zero_()
    if key_custom.grad is not None: key_custom.grad.zero_()
    if value_custom.grad is not None: value_custom.grad.zero_()

    # We don't actually need to call forward again for custom backward,
    # as it recomputes what it needs from query_custom, key_custom, value_custom.
    # The grad_output_dummy is what we pass.
    softmax_scale_val = 1.0 / math.sqrt(D) # ensure consistency

    dq_custom, dk_custom, dv_custom = flash_attention_backward(
        grad_output_dummy, 
        query_custom, 
        key_custom, 
        value_custom, 
        K_tile, 
        Q_tile,
        softmax_scale=softmax_scale_val # Pass the scale explicitly
    )

    # --- Comparison ---
    # Set a tolerance appropriate for float64. For float32, might need to be higher.
    # For float64, atol=1e-7, rtol=1e-5 is usually good.
    # If using float32, atol=1e-4, rtol=1e-3 might be necessary.
    atol = 1e-7 
    rtol = 1e-5
    if dtype == torch.float32:
        atol = 1e-4
        rtol = 1e-3
        
    print("\nComparing gradients:")
    print(f"dQ close: {torch.allclose(dq_pytorch, dq_custom, atol=atol, rtol=rtol)}")
    print(f"dK close: {torch.allclose(dk_pytorch, dk_custom, atol=atol, rtol=rtol)}")
    print(f"dV close: {torch.allclose(dv_pytorch, dv_custom, atol=atol, rtol=rtol)}")

    if not torch.allclose(dq_pytorch, dq_custom, atol=atol, rtol=rtol):
        print("dQ Pytorch:\n", dq_pytorch[0,0,0,:5])
        print("dQ Custom:\n", dq_custom[0,0,0,:5])
        print("dQ Diff:\n", (dq_pytorch - dq_custom).abs().max())
    if not torch.allclose(dk_pytorch, dk_custom, atol=atol, rtol=rtol):
        print("dK Pytorch:\n", dk_pytorch[0,0,0,:5])
        print("dK Custom:\n", dk_custom[0,0,0,:5])
        print("dK Diff:\n", (dk_pytorch - dk_custom).abs().max())
    if not torch.allclose(dv_pytorch, dv_custom, atol=atol, rtol=rtol):
        print("dV Pytorch:\n", dv_pytorch[0,0,0,:5])
        print("dV Custom:\n", dv_custom[0,0,0,:5])
        print("dV Diff:\n", (dv_pytorch - dv_custom).abs().max())

if __name__ == '__main__':
    run_test()