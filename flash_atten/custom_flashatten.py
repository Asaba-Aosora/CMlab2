# atten_py_implement.py (Optimized Version)
import torch
import torch.nn.functional as F
import math

# If using PyTorch 2.0+, torch.compile can be a game changer for this kind of code.
# Example: @torch.compile (mode="reduce-overhead" or "max-autotune")
# @torch.compile(backend="aot_eager")
def custom_flash_attention_forward(query, key, value, Q_tile_size, K_tile_size, causal=False, softmax_scale=None):
    B, H, N_q, D_head = query.shape
    _, _, N_kv, _ = key.shape

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(D_head)

    # Initialize global output, m_i, and l_i tensors
    output = torch.zeros_like(query, dtype=torch.float32) # Use float32 for accumulator
    # m_i stores the running max for each query element
    m_i = torch.full((B, H, N_q), -float('inf'), device=query.device, dtype=torch.float32)
    # l_i_sum_numerator stores the running sum of exp(scores - new_max) for each query element
    l_i_sum_numerator = torch.zeros((B, H, N_q), device=query.device, dtype=torch.float32)

    # Pre-calculate indices for slicing Q
    q_tile_indices = [(i * Q_tile_size, min((i + 1) * Q_tile_size, N_q))
                      for i in range(math.ceil(N_q / Q_tile_size))]
    
    # Pre-calculate indices for slicing K and V
    k_tile_indices = [(j * K_tile_size, min((j + 1) * K_tile_size, N_kv))
                      for j in range(math.ceil(N_kv / K_tile_size))]

    # Outer loop: Iterate over blocks of Q (Tc in FlashAttention paper)
    for i_q, (q_start, q_end) in enumerate(q_tile_indices):
        Q_i = query[:, :, q_start:q_end, :]  # Current Q block: (B, H, current_Q_tile_size, D_head)
        current_N_q_tile = Q_i.shape[2]

        # Initialize accumulators for THIS Q_i block (conceptually, on-chip SRAM in the paper)
        # These will be updated as we iterate through K_j blocks for the current Q_i
        o_block_accum = torch.zeros((B, H, current_N_q_tile, D_head), device=query.device, dtype=torch.float32)
        m_block_running = torch.full((B, H, current_N_q_tile), -float('inf'), device=query.device, dtype=torch.float32)
        l_block_running_sum_num = torch.zeros((B, H, current_N_q_tile), device=query.device, dtype=torch.float32)

        # Inner loop: Iterate over blocks of K and V (Tr in FlashAttention paper)
        for j_k, (k_start, k_end) in enumerate(k_tile_indices):
            K_j = key[:, :, k_start:k_end, :]    # Current K block: (B, H, current_K_tile_size, D_head)
            V_j = value[:, :, k_start:k_end, :]  # Current V block: (B, H, current_K_tile_size, D_head)
            current_N_kv_tile = K_j.shape[2]

            # Calculate score block S_ij = Q_i @ K_j^T
            # (B, H, current_N_q_tile, D_head) @ (B, H, D_head, current_K_tile_size)
            # -> (B, H, current_N_q_tile, current_K_tile_size)
            S_ij_block = torch.matmul(Q_i, K_j.transpose(-2, -1))

            if causal:
                # Apply causal mask based on global indices within the current (Q_i, K_j) block
                # query_indices_global are [q_start, q_end-1]
                # key_indices_global are [k_start, k_end-1]
                # Mask if key_global_idx > query_global_idx
                
                # Generate indices for the current Q_i block rows
                q_indices_abs = torch.arange(q_start, q_end, device=query.device).view(current_N_q_tile, 1)
                # Generate indices for the current K_j block columns
                k_indices_abs = torch.arange(k_start, k_end, device=query.device).view(1, current_N_kv_tile)
                
                # Create mask: True where k_abs > q_abs (element should be masked)
                # Shape: (current_N_q_tile, current_K_tile_size)
                block_causal_mask = k_indices_abs > q_indices_abs
                
                # Expand mask for Batch and Head dimensions to match S_ij_block
                # S_ij_block shape: (B, H, current_N_q_tile, current_K_tile_size)
                # block_causal_mask needs to be (1, 1, current_N_q_tile, current_K_tile_size) for broadcasting
                S_ij_block.masked_fill_(block_causal_mask.unsqueeze(0).unsqueeze(0), -float('inf'))

            # Online Softmax calculations (for the current Q_i block, using K_j, V_j)
            # m_block_running is m_i^(prev) for Q_i from the previous K_j iteration (or initial -inf)
            # l_block_running_sum_num is l_i^(prev) for Q_i
            # o_block_accum is O_i^(prev) for Q_i
            
            # S_ij_block needs scaling by softmax_scale before max and exp operations
            S_ij_block_scaled = S_ij_block * softmax_scale

            m_ij_block, _ = torch.max(S_ij_block_scaled, dim=-1) # Row-wise max of current scores block (B, H, current_N_q_tile)
            
            # New running max for Q_i: m_i_new = max(m_i_prev, m_ij_curr)
            m_i_new_block = torch.maximum(m_block_running, m_ij_block) # (B, H, current_N_q_tile)

            # Numerator of softmax: P_ij_numerator = exp(S_ij_scaled - m_i_new_block_expanded)
            # (B,H,current_N_q_tile,current_K_tile_size)
            P_ij_numerator_block = torch.exp(S_ij_block_scaled - m_i_new_block.unsqueeze(-1)) 

            # Rescale previous sum of numerators and previous output block
            # exp_diff_m_block = exp(m_i_prev - m_i_new)
            exp_diff_m_block = torch.exp(m_block_running - m_i_new_block) # (B, H, current_N_q_tile)
            
            # l_i_new = l_i_prev * exp(m_i_prev - m_i_new) + sum(P_ij_numerator_curr)
            l_block_running_sum_num_prev_scaled = l_block_running_sum_num * exp_diff_m_block
            l_block_running_sum_num = l_block_running_sum_num_prev_scaled + torch.sum(P_ij_numerator_block, dim=-1) # (B, H, current_N_q_tile)
            
            # O_i_new = O_i_prev * exp(m_i_prev - m_i_new) + P_ij_numerator_curr @ V_j
            # Scale old output contribution
            o_block_accum = o_block_accum * exp_diff_m_block.unsqueeze(-1) # (B,H,current_N_q_tile,D_head)
            
            # Add current tile's contribution
            # P_ij_numerator_block @ V_j
            # (B,H,current_N_q_tile,current_K_tile_size) @ (B, H, current_K_tile_size, D_head)
            # -> (B,H,current_N_q_tile,D_head)
            o_block_accum += torch.matmul(P_ij_numerator_block, V_j)
            
            # Update m_block_running for the next K_j iteration (for this Q_i block)
            m_block_running = m_i_new_block

        # After iterating through all K_j blocks for the current Q_i block:
        # Write the final accumulated local o_block_accum, m_block_running, l_block_running_sum_num
        # to the corresponding slices of the global output, m_i, l_i_sum_numerator tensors.
        output[:, :, q_start:q_end, :] = o_block_accum
        m_i[:, :, q_start:q_end] = m_block_running
        l_i_sum_numerator[:, :, q_start:q_end] = l_block_running_sum_num

    # Final normalization of the global output tensor
    # output_normalized = output / l_i_sum_numerator
    # Add a small epsilon to l_i_sum_numerator to prevent division by zero.
    output_normalized = output / (l_i_sum_numerator.unsqueeze(-1) + 1e-8)
    
    # For autograd, what's returned is what's used.
    # The intermediate m_i and l_i_sum_numerator are part of the computation graph if they are needed for backward.
    return output_normalized, l_i_sum_numerator # Return l_i_sum_numerator (L_i in FlashAttention paper)


def custom_tiled_attention_NO_ONLINE_softmax(query, key, value, K_tile_size, Q_tile_size=None, softmax_scale=None):
    # TODO: Need to implement the tiled attention with no online softmax

    return output_full, None # Not returning detailed weights