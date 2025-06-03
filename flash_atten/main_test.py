import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math
import psutil # For CPU memory usage

# Models are now in atten_model.py
# Assuming atten_model.py and its contents are CPU-compatible or
# relevant parts are conditionally executed (like FlashAttentionLibModel)
# Also assuming models default to non-causal attention if 'causal' param is omitted.
from atten_model import StandardAttentionModel, FlashAttentionLibModel, CustomFlashAttentionPyCore, CustomTiledAttentionNoOnlineSoftmaxPyCore, AttentionTestWrapper

# --- Profiling Utilities ---
_cpu_peak_memory_current_peak_bytes = 0 # Stores absolute peak RSS during a profiled section for CPU

def get_gpu_memory_usage_gb(device): # Reports CURRENT usage
    if device.type == 'cuda':
        return torch.cuda.memory_allocated(device) / (1024**3)
    else:
        # For CPU, returns current process RSS.
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024**3)

def reset_peak_gpu_memory_stats(device): # For CUDA, resets allocator stats. For CPU, resets our RSS peak tracker.
    global _cpu_peak_memory_current_peak_bytes
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    else:
        # Initialize peak with current memory state
        process = psutil.Process(os.getpid())
        _cpu_peak_memory_current_peak_bytes = process.memory_info().rss

def _record_cpu_memory_snapshot_if_needed(device): # Internal helper
    """Updates the CPU peak memory tracker if on CPU."""
    global _cpu_peak_memory_current_peak_bytes
    if device.type == 'cpu':
        process = psutil.Process(os.getpid())
        current_mem_bytes = process.memory_info().rss
        _cpu_peak_memory_current_peak_bytes = max(_cpu_peak_memory_current_peak_bytes, current_mem_bytes)

def get_peak_gpu_memory_usage_gb(device): # For CUDA, actual peak allocator. For CPU, observed peak RSS.
    global _cpu_peak_memory_current_peak_bytes
    if device.type == 'cuda':
        return torch.cuda.max_memory_allocated(device) / (1024**3)
    else:
        # Returns the peak RSS recorded by _record_cpu_memory_snapshot_if_needed()
        # since the last reset_peak_gpu_memory_stats()
        return _cpu_peak_memory_current_peak_bytes / (1024**3)

# --- Verification Function (same as before, already CPU-safe for comparison) ---
def verify_outputs_and_grads(model_A_name, output_A, inputs_A_with_grad,
                             model_B_name, output_B, inputs_B_with_grad,
                             atol=1e-5, rtol=1e-3):
    print(f"\n--- Verifying: {model_A_name} vs {model_B_name} ---")
    
    # Verify Forward Output
    try:
        if output_A is None or output_B is None:
            print("One of the outputs is None, cannot compare.")
            return
        
        out_A_cpu = output_A.detach().cpu()
        out_B_cpu = output_B.detach().cpu()

        forward_match = torch.allclose(out_A_cpu, out_B_cpu, atol=atol, rtol=rtol)
        print(f"Forward Output Match: {forward_match}")

        abs_diff = torch.abs(out_A_cpu - out_B_cpu)
        print(f"  Max absolute difference: {torch.max(abs_diff).item()}")
        print(f"  Mean absolute difference: {torch.mean(abs_diff).item()}")
        if not forward_match:
            print(f"WARNING: Outputs for {model_A_name} and {model_B_name} do not match within tolerance.")
    except Exception as e:
        print(f"Error during forward output verification: {e}")

    # Verify Gradients
    if not inputs_A_with_grad or not inputs_B_with_grad:
        print("No inputs with gradients provided for verification or backward pass not enabled.")
        return

    grad_A_list = [inp.grad for inp in inputs_A_with_grad if inp is not None and inp.grad is not None]
    grad_B_list = [inp.grad for inp in inputs_B_with_grad if inp is not None and inp.grad is not None]

    if not grad_A_list or not grad_B_list:
        if not grad_A_list and any(inp.requires_grad for inp in inputs_A_with_grad if inp is not None):
             print(f"Gradients not found for {model_A_name}, but were expected.")
        if not grad_B_list and any(inp.requires_grad for inp in inputs_B_with_grad if inp is not None):
             print(f"Gradients not found for {model_B_name}, but were expected.")
        if not grad_A_list and not grad_B_list and not any(inp.requires_grad for inp in inputs_A_with_grad if inp is not None):
            pass 
        else:
            print("Gradients not found for one or both models where expected.")
        return
        
    for i in range(min(len(grad_A_list), len(grad_B_list))):
        grad_A_cpu = grad_A_list[i].detach().cpu()
        grad_B_cpu = grad_B_list[i].detach().cpu()
        try:
            grad_match = torch.allclose(grad_A_cpu, grad_B_cpu, atol=atol*10, rtol=rtol*10) 
            print(f"Gradient Match for input {i}: {grad_match}")
            
            abs_diff_grad = torch.abs(grad_A_cpu - grad_B_cpu)
            print(f"  Max absolute grad difference: {torch.max(abs_diff_grad).item()}")
            print(f"  Mean absolute grad difference: {torch.mean(abs_diff_grad).item()}")
            if not grad_match:
                print(f"WARNING: Gradients for input {i} for {model_A_name} and {model_B_name} do not match within tolerance.")
        except Exception as e:
            print(f"Error during gradient verification for input {i}: {e}")


# --- Main Experiment ---
def run_profile(model_name, model_wrapper, x_input_main, criterion, device, enable_backward=True):
    print(f"\n--- Profiling: {model_name} on {device.type} ---")
    model_wrapper.to(device)
    model_wrapper.train() 
    
    x_input = x_input_main.clone().detach().to(device).requires_grad_(enable_backward)

    run_outputs = {}
    run_inputs_with_grad = [x_input] if enable_backward else []

    # Warm-up
    for _ in range(2):
        y_pred, _ = model_wrapper(x_input)
        if enable_backward:
            dummy_target = torch.randn_like(y_pred, device=device)
            loss = criterion(y_pred, dummy_target)
            loss.backward()
            model_wrapper.zero_grad() 
            if x_input.grad is not None:
                x_input.grad.zero_() 
    
    # Actual profiling
    if device.type == 'cuda':
        torch.cuda.synchronize(device) 
    reset_peak_gpu_memory_stats(device) # Resets CUDA peak or initializes CPU peak tracking
    
    # Forward pass
    _record_cpu_memory_snapshot_if_needed(device) # Record memory before fwd for CPU peak tracking
    fwd_start_time = time.time()
    y_pred, _ = model_wrapper(x_input)
    if device.type == 'cuda':
        torch.cuda.synchronize(device) 
    _record_cpu_memory_snapshot_if_needed(device) # Record memory after fwd for CPU peak tracking
    fwd_time = time.time() - fwd_start_time
    run_outputs['output'] = y_pred.detach().clone() 

    bwd_time = float('nan')
    if enable_backward:
        dummy_target = torch.randn_like(y_pred, device=device)
        loss = criterion(y_pred, dummy_target)
        
        _record_cpu_memory_snapshot_if_needed(device) # Record memory before bwd for CPU peak tracking
        bwd_start_time = time.time()
        loss.backward()
        if device.type == 'cuda':
            torch.cuda.synchronize(device) 
        _record_cpu_memory_snapshot_if_needed(device) # Record memory after bwd for CPU peak tracking
        bwd_time = time.time() - bwd_start_time
    
    peak_mem_gb = get_peak_gpu_memory_usage_gb(device) 

    print(f"{model_name} - Forward Time: {fwd_time:.4f} s")
    if enable_backward:
        print(f"{model_name} - Backward Time: {bwd_time:.4f} s")
    
    mem_type_str = "Peak GPU Memory" if device.type == 'cuda' else "Peak Process Memory"
    print(f"{model_name} - {mem_type_str}: {peak_mem_gb:.4f} GB (on {device.type})")

    if enable_backward:
        if x_input.grad is not None:
            x_input_clone_for_grad_storage = x_input.detach().clone()
            x_input_clone_for_grad_storage.grad = x_input.grad.detach().clone()
            run_inputs_with_grad = [x_input_clone_for_grad_storage]
            
            x_input.grad.zero_() 
        else: 
            run_inputs_with_grad = [x_input.detach().clone()] 
            
    model_wrapper.zero_grad() 

    return fwd_time, bwd_time, peak_mem_gb, run_outputs, run_inputs_with_grad


if __name__ == "__main__":
    # Parameters
    B = 2
    N_seq = 256 # 512 1024 2048 4096    
    C_model = 64   
    NUM_HEADS = 8
    K_TILE_SIZE = 128 

    # Determine device
    if torch.cuda.is_available():
        device_str = "cuda"
        print("CUDA is available. Testing on GPU.")
    else:
        device_str = "cpu"
        print("CUDA not available. Testing on CPU. Will report process RSS for memory.")
        # Optionally reduce N_seq or B for CPU if tests are too slow
        # N_seq = 256
        # K_TILE_SIZE = 64
        # B = 1
        # print(f"Adjusted parameters for CPU: N_seq={N_seq}, K_TILE_SIZE={K_TILE_SIZE}, B={B}")

    device = torch.device(device_str)
    print(f"Using device: {device}.")

    x_main_cpu = torch.randn(B, N_seq, C_model, dtype=torch.float32) 
    criterion = nn.MSELoss()
    all_run_data = {} 
    results_summary = {}

    # --- Test Cases ---

    # 1. Standard PyTorch Attention
    print("\n--- Test Case 1: Standard PyTorch Attention ---")
    try:
        std_core = StandardAttentionModel(num_heads=NUM_HEADS)
        model_std_wrapped = AttentionTestWrapper(C_model, NUM_HEADS, std_core)
        f,b,m, out_data, grad_data = run_profile("Standard PyTorch", model_std_wrapped, x_main_cpu, criterion, device)
        all_run_data["Standard PyTorch"] = {"outputs": out_data, "inputs_w_grad": grad_data}
        results_summary["Standard PyTorch"] = (f,b,m)
    except Exception as e:
        print(f"Error during Standard PyTorch Attention profile: {e}")
        results_summary["Standard PyTorch"] = (float('nan'), float('nan'), float('nan'))


    # 2. FlashAttention Library (if available and on CUDA)
    print("\n--- Test Case 2: FlashAttention Library ---")
    if device.type == 'cuda':
        try:
            x_input_for_flash = x_main_cpu 
            
            flash_lib_core = FlashAttentionLibModel(num_heads=NUM_HEADS)
            model_flash_lib_wrapped = AttentionTestWrapper(C_model, NUM_HEADS, flash_lib_core)
            f,b,m, out_data, grad_data = run_profile("FlashAttention Library", model_flash_lib_wrapped, x_input_for_flash, criterion, device)
            all_run_data["FlashAttention Library"] = {"outputs": out_data, "inputs_w_grad": grad_data}
            results_summary["FlashAttention Library"] = (f,b,m)

            if "Standard PyTorch" in all_run_data and all_run_data["Standard PyTorch"]["outputs"]:
                verify_outputs_and_grads(
                    "Standard PyTorch", all_run_data["Standard PyTorch"]["outputs"].get('output'), all_run_data["Standard PyTorch"]["inputs_w_grad"],
                    "FlashAttention Library", out_data.get('output'), grad_data,
                    atol=1e0, rtol=1e-4 
                )
        except ImportError:
             print("FlashAttention library not installed. Skipping FlashAttention Library test.")
             results_summary["FlashAttention Library"] = (float('nan'), float('nan'), float('nan'))
        except Exception as e:
            print(f"Error during FlashAttention Library profile: {e}")
            results_summary["FlashAttention Library"] = (float('nan'), float('nan'), float('nan'))
    else:
        print("Skipping FlashAttention Library profiling (not on CUDA).")
        results_summary["FlashAttention Library"] = (float('nan'), float('nan'), float('nan'))


    # 3. Custom Python Flash-like Attention
    print("\n--- Test Case 3: Custom Python Flash-like Attention ---")
    Q_TILE_SIZE_NO_ONLINE = K_TILE_SIZE 
    try:
        custom_py_core = CustomFlashAttentionPyCore(num_heads=NUM_HEADS, Q_tile_size=Q_TILE_SIZE_NO_ONLINE, K_tile_size=K_TILE_SIZE)
        model_custom_py_wrapped = AttentionTestWrapper(C_model, NUM_HEADS, custom_py_core)
        f,b,m, out_data, grad_data = run_profile("Custom Python Flash-like", model_custom_py_wrapped, x_main_cpu, criterion, device)
        all_run_data["Custom Python Flash-like"] = {"outputs": out_data, "inputs_w_grad": grad_data}
        results_summary["Custom Python Flash-like"] = (f,b,m)
        
        if "Standard PyTorch" in all_run_data and all_run_data["Standard PyTorch"]["outputs"]:
            verify_outputs_and_grads(
                "Standard PyTorch", all_run_data["Standard PyTorch"]["outputs"].get('output'), all_run_data["Standard PyTorch"]["inputs_w_grad"],
                "Custom Python Flash-like", out_data.get('output'), grad_data,
                atol=1e0, rtol=1e-4 
            )
    except Exception as e:
        print(f"Error during Custom Python Flash-like Attention profile: {e}")
        results_summary["Custom Python Flash-like"] = (float('nan'), float('nan'), float('nan'))


    # 4. Custom Python Tiled Attention (No Online Softmax)
    print("\n--- Test Case 4: Custom Python Tiled Attention (No Online Softmax) ---")
    try:
        custom_py_no_online_core = CustomTiledAttentionNoOnlineSoftmaxPyCore(
            num_heads=NUM_HEADS, K_tile_size=K_TILE_SIZE, Q_tile_size=Q_TILE_SIZE_NO_ONLINE
        )
        model_custom_py_no_online_wrapped = AttentionTestWrapper(C_model, NUM_HEADS, custom_py_no_online_core)
        
        f,b,m, out_data, grad_data = run_profile(
            "Custom Py Tiled (No Online Softmax)", 
            model_custom_py_no_online_wrapped, 
            x_main_cpu, criterion, device
        )
        all_run_data["Custom Py Tiled (No Online Softmax)"] = {"outputs": out_data, "inputs_w_grad": grad_data}
        results_summary["Custom Py Tiled (No Online Softmax)"] = (f,b,m)

        if "Standard PyTorch" in all_run_data and all_run_data["Standard PyTorch"]["outputs"]:
            verify_outputs_and_grads(
                "Standard PyTorch", all_run_data["Standard PyTorch"]["outputs"].get('output'), all_run_data["Standard PyTorch"]["inputs_w_grad"],
                "Custom Py Tiled (No Online Softmax)", out_data.get('output'), grad_data,
                atol=1e0, rtol=1e-4 
            )
    except Exception as e:
        print(f"Error during Custom Py Tiled (No Online Softmax) Attention profile: {e}")
        results_summary["Custom Py Tiled (No Online Softmax)"] = (float('nan'), float('nan'), float('nan'))
    
    # --- Final Summary ---
    print("\n\n--- Final Performance Results ---")
    header_mem = "Peak Mem (GB)"
    print(f"{'Method':<35} | {'Fwd Time (s)':<15} | {'Bwd Time (s)':<15} | {header_mem:<20}")
    print("-" * 90)
    for name, (f,b,m) in results_summary.items():
        f_str = f"{f:.4f}" if not math.isnan(f) else "N/A"
        b_str = f"{b:.4f}" if not math.isnan(b) else "N/A"
        m_str = f"{m:.4f}" if not math.isnan(m) else "N/A" 
        print(f"{name:<35} | {f_str:<15} | {b_str:<15} | {m_str:<20}")