import time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve as scipy_spsolve, splu
import torch
import torch.optim as optim


# --- Matrix and Vector Generation ---
def generate_sparse_diagonally_dominant_system(n, density=0.01, seed=None):
    """
    Generates a sparse, strictly diagonally dominant square matrix A
    and a vector b for Ax=b.
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    A_rand = sp.random(
        n, n, density=density, format="csr", random_state=seed, dtype=np.float64
    )
    A_rand.eliminate_zeros()

    A_diag_abs_row_sum = np.array(np.abs(A_rand).sum(axis=1)).flatten()

    new_diag_values = (
        A_diag_abs_row_sum - np.abs(A_rand.diagonal()) + np.random.uniform(0.1, 1.0, n)
    )
    new_diag_values[new_diag_values <= 1e-9] = 1.0

    A = A_rand.copy()
    A.setdiag(0)
    A = A + sp.diags(new_diag_values, format="csr", dtype=np.float64)
    A.eliminate_zeros()

    x_true = np.random.rand(n).astype(np.float64)
    b = A @ x_true

    A = A.astype(np.float64)
    b = b.astype(np.float64)
    x_true = x_true.astype(np.float64)

    print(
        f"Diagonally dominant matrix A ({A.shape}) generated with {A.nnz} non-zero elements."
    )
    # Optional check (can be slow for large N)
    # is_dd = True
    # for i in range(n):
    #     diag_val = np.abs(A[i,i])
    #     row_sum_abs_off_diag = 0
    #     for j_idx in range(A.indptr[i], A.indptr[i+1]):
    #         if A.indices[j_idx] != i:
    #             row_sum_abs_off_diag += np.abs(A.data[j_idx])
    #     if diag_val <= row_sum_abs_off_diag:
    #         is_dd = False
    #         print(f"Row {i} is not strictly diagonally dominant: diag={diag_val}, off_diag_sum={row_sum_abs_off_diag}")
    #         break
    # if is_dd:
    #     print("Matrix A is strictly diagonally dominant.")
    # else:
    #     print("Matrix A is NOT strictly diagonally dominant (check generation). This might affect Gauss-Seidel.")

    return A, b, x_true


def generate_sparse_non_diagonally_dominant_system(n, density=0.01, seed=None):
    """
    Generates a sparse, non-diagonally dominant square matrix A
    and a vector b for Ax=b.
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate a random sparse matrix
    A_rand = sp.random(
        n, n, density=density, format="csr", random_state=seed, dtype=np.float64
    )
    A_rand.eliminate_zeros()

    # Ensure the matrix is not diagonally dominant
    A_diag_abs_row_sum = np.array(np.abs(A_rand).sum(axis=1)).flatten()
    new_diag_values = (
        A_diag_abs_row_sum - np.abs(A_rand.diagonal()) - np.random.uniform(0.1, 1.0, n)
    )
    new_diag_values[new_diag_values <= 0] = (
        -1.0
    )  # Ensure diagonal values are smaller than off-diagonal sums

    A = A_rand.copy()
    A.setdiag(0)  # Clear the diagonal
    A = A + sp.diags(new_diag_values, format="csr", dtype=np.float64)
    A.eliminate_zeros()

    # Generate a random solution vector x_true and compute b
    x_true = np.random.rand(n).astype(np.float64)
    b = A @ x_true

    A = A.astype(np.float64)
    b = b.astype(np.float64)
    x_true = x_true.astype(np.float64)

    print(
        f"Non-diagonally dominant matrix A ({A.shape}) generated with {A.nnz} non-zero elements."
    )
    return A, b, x_true


import numpy as np
import scipy.sparse as sp


def generate_sparse_large_spectral_radius_system(
    n, density=0.01, scale_offdiag=5.0, seed=None, check_spectral_radius=False
):
    """
    生成一个谱半径大于1的稀疏方阵A和向量b (Ax=b) , 返回A, b, x_true。
    scale_offdiag: 放大非对角元素的倍数, 通常>1可使谱半径>1
    check_spectral_radius: 若为True, 返回谱半径 (仅适合小矩阵)
    """
    if seed is not None:
        np.random.seed(seed)

    # 生成稀疏随机矩阵
    A_rand = sp.random(
        n, n, density=density, format="csr", random_state=seed, dtype=np.float64
    )
    A_rand.eliminate_zeros()

    # 放大非对角元素
    A_offdiag = A_rand.copy()
    A_offdiag.setdiag(0)
    A_offdiag = A_offdiag * scale_offdiag

    # 构造较小的对角线, 确保非对角主导
    diag = np.random.uniform(0.1, 0.9, n)
    A = A_offdiag + sp.diags(diag, format="csr", dtype=np.float64)
    A.eliminate_zeros()

    x_true = np.random.rand(n).astype(np.float64)
    b = A @ x_true

    print(
        f"Sparse matrix with potentially large spectral radius generated. Shape: {A.shape}, nnz: {A.nnz}"
    )

    spectral_radius = None
    if check_spectral_radius:
        # 仅适合小矩阵, 否则内存和速度都很慢
        try:
            eigvals = np.linalg.eigvals(A.toarray())
            spectral_radius = np.max(np.abs(eigvals))
            print(f"Spectral radius: {spectral_radius:.3f}")
        except Exception as e:
            print(f"Spectral radius computation failed: {e}")

    if check_spectral_radius:
        return A, b, x_true, spectral_radius
    else:
        return A, b, x_true


# --- Solver Implementations ---


# Method 1: Manual Gaussian Elimination (with Partial Pivoting)
def solve_manual_gaussian_elimination(A_sparse, b_np):
    """
    Solves Ax=b using manual Gaussian elimination with partial pivoting.
    WARNING: This converts the sparse matrix to dense and is very inefficient
             for large N (O(N^3) complexity, O(N^2) memory).
    """
    # TODO: Need to implement this
    start_time = time.time()
    try:
        n = len(b_np)
        # 将稀疏矩阵转为密集矩阵（仅适用于小规模矩阵）
        A_dense = A_sparse.toarray()
        b = b_np.copy()

        # 增广矩阵 [A|b]
        aug_matrix = np.hstack((A_dense, b.reshape(-1, 1)))

        # 正向消元
        for i in range(n):
            # 部分主元选择
            pivot_row = i
            for j in range(i + 1, n):
                if abs(aug_matrix[j, i]) > abs(aug_matrix[pivot_row, i]):
                    pivot_row = j

            # 交换行
            if pivot_row != i:
                aug_matrix[i], aug_matrix[pivot_row] = (
                    aug_matrix[pivot_row].copy(),
                    aug_matrix[i].copy(),
                )

            # 消元
            pivot = aug_matrix[i, i]
            if abs(pivot) < 1e-10:
                raise ValueError("矩阵接近奇异，无法求解")

            for j in range(i + 1, n):
                factor = aug_matrix[j, i] / pivot
                aug_matrix[j] -= factor * aug_matrix[i]

        # 回代求解
        x = np.zeros(n)
        x[n - 1] = aug_matrix[n - 1, n] / aug_matrix[n - 1, n - 1]
        for i in range(n - 2, -1, -1):
            x[i] = (
                aug_matrix[i, n] - np.dot(aug_matrix[i, i + 1 : n], x[i + 1 : n])
            ) / aug_matrix[i, i]

        solve_time = time.time() - start_time
        return x, solve_time, None  # 直接法无迭代次数
    except Exception as e:
        solve_time = time.time() - start_time
        print(f"手动高斯消去法失败: {e}")
        return None, solve_time, None


# Method 2: Gauss-Seidel Iteration
def solve_gauss_seidel(A_sparse, b, max_iters=1000, tol=1e-8, verbose=True):
    # TODO: Need to implement this
    start_time = time.time()
    try:
        A = A_sparse.tocsr()  # 确保使用CSR格式
        n = len(b)
        x = np.zeros(n)  # 初始解
        x_prev = np.zeros(n)  # 前一次迭代的解
        iters = 0
        
        # 预计算对角线元素的倒数（提高效率）
        diag = A.diagonal()
        diag_inv = 1.0 / diag  # 注意：这里假设对角元素非零
        
        for iters in range(max_iters):
            # 复制当前解用于收敛性检查
            x_prev[:] = x
            
            # 高斯-赛德尔迭代更新
            for i in range(n):
                row_start = A.indptr[i]
                row_end = A.indptr[i + 1]
                
                # 计算非对角元素的贡献
                sigma = 0.0
                for j_idx in range(row_start, row_end):
                    j = A.indices[j_idx]
                    if j != i:
                        sigma += A.data[j_idx] * x[j]
                
                # 更新x[i]
                x[i] = (b[i] - sigma) * diag_inv[i]
            
            # 检查收敛性
            rel_error = np.linalg.norm(x - x_prev) / (np.linalg.norm(x) + 1e-10)
            if rel_error < tol:
                break
        
        solve_time = time.time() - start_time
        
        if verbose and iters >= max_iters:
            print(f"高斯-赛德尔迭代达到最大迭代次数 {max_iters} 仍未收敛，相对误差: {rel_error:.6e}")
        
        return x, solve_time, iters + 1  # 返回实际迭代次数
    
    except Exception as e:
        solve_time = time.time() - start_time
        print(f"高斯-赛德尔迭代法失败: {e}")
        return None, solve_time, max_iters


# SciPy LU direct solver (for comparison with manual GE)
def solve_scipy_splu(A_sparse, b):
    start_time = time.time()
    try:
        # splu prefers CSC format for decomposition phase
        lu = splu(A_sparse.tocsc())
        x = lu.solve(b)
        solve_time = time.time() - start_time
        return x, solve_time, None
    except Exception as e:
        solve_time = time.time() - start_time
        print(f"SciPy splu solver failed: {e}")
        return None, solve_time, None


# SciPy general sparse solver (often iterative like GMRES or direct based on matrix)
def solve_scipy_spsolve_baseline(A_sparse, b):
    start_time = time.time()
    try:
        x = scipy_spsolve(A_sparse, b)  # This can choose different methods
        solve_time = time.time() - start_time
        return x, solve_time, None
    except Exception as e:
        solve_time = time.time() - start_time
        print(f"SciPy spsolve (baseline) failed: {e}")
        return None, solve_time, None


# --- Metrics Calculation ---
def calculate_relative_residual_norm(A, x, b):
    if x is None or np.any(np.isnan(x)) or np.any(np.isinf(x)):
        return np.nan
    try:
        residual = b - A @ x
        norm_residual = np.linalg.norm(residual)
        norm_b = np.linalg.norm(b)
        if norm_b < np.finfo(np.float64).eps:
            return np.nan if norm_residual > 1e-9 else 0.0  # If b is zero vector
        return norm_residual / norm_b
    except Exception as e:
        print(
            f"Error calculating residual for x of shape {x.shape if x is not None else 'None'}: {e}"
        )
        return np.nan


def calculate_solution_error(x_computed, x_true):
    if (
        x_computed is None
        or x_true is None
        or np.any(np.isnan(x_computed))
        or np.any(np.isinf(x_computed))
    ):
        return np.nan
    norm_x_true = np.linalg.norm(x_true)
    if norm_x_true < np.finfo(np.float64).eps:
        return np.nan if np.linalg.norm(x_computed - x_true) > 1e-9 else 0.0
    return np.linalg.norm(x_computed - x_true) / norm_x_true

# 针对非严格对角矩阵, 绘制收敛取消对比图
import matplotlib.pyplot as plt
def plot_convergence(residuals_dd, residuals_ndd, n, density):
    """绘制收敛曲线对比图"""
    plt.figure(figsize=(10, 6))
    plt.semilogy(residuals_dd, 'b-', label='Diagonally Dominant')
    plt.semilogy(residuals_ndd, 'r--', label='Non-Diagonally Dominant')
    plt.xlabel('Iterations')
    plt.ylabel('Relative Residual')
    plt.title(f'Convergence Comparison (N={n}, Density={density})')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'convergence_N_{n}_density_{density}.png')
    plt.close()


# --- Main Execution and Comparison ---
if __name__ == "__main__":
    # N = 1000     # For testing manual GE and faster iterations
    # # N = 10000    # Target size (manual GE will be skipped)
    # # N = 10001 # Test with N > 1e4

    # DENSITY = 0.01
    # SEED = 10

    # 以下是严格对角矩阵
    # 定义不同的矩阵大小和稀疏度, 进行测试
    N_values = [100, 500, 2000]
    DESITY_values = [0.01, 0.05, 0.1]
    SEED = 10
    
    all_results={}

    for N in N_values:
        for DENSITY in DESITY_values:
            print(f"--- Generating System (N={N}, Density={DENSITY}) ---")
            # NOTE: Test different matrix types by uncommenting the desired line
            A_sparse, b_np, x_true_np = generate_sparse_diagonally_dominant_system(
                N, density=DENSITY, seed=SEED
            )
            # A_sparse, b_np, x_true_np = generate_sparse_non_diagonally_dominant_system(N, density=DENSITY, seed=SEED)
            # A_sparse, b_np, x_true_np = generate_sparse_large_spectral_radius_system(N, density=DENSITY, scale_offdiag=2.0, seed=SEED)
            print(
                f"Norm of b: {np.linalg.norm(b_np):.2e}, Norm of x_true: {np.linalg.norm(x_true_np):.2e}"
            )
            # Small test for condition number if N is small
            # if N <= 200:
            #     try:
            #         cond_A = np.linalg.cond(A_sparse.toarray())
            #         print(f"Condition number of A (est.): {cond_A:.2e}")
            #     except Exception as e:
            #         print(f"Could not compute condition number: {e}")
            print("-" * 70)

            results = {}

            # SciPy Baseline: spsolve (general, often iterative for large sparse)
            print("\n--- SciPy Baseline: spsolve ---")
            x_spsolve, time_spsolve, _ = solve_scipy_spsolve_baseline(
                A_sparse.copy(), b_np.copy()
            )
            if x_spsolve is not None:
                rel_res_spsolve = calculate_relative_residual_norm(A_sparse, x_spsolve, b_np)
                err_spsolve = calculate_solution_error(x_spsolve, x_true_np)
                print(f"SciPy spsolve Time: {time_spsolve:.4f} s")
                print(f"SciPy spsolve Relative Residual Norm: {rel_res_spsolve:.2e}")
                print(f"SciPy spsolve Solution Error (vs x_true): {err_spsolve:.2e}")
                results["Baseline spsolve"] = {
                    "time": time_spsolve,
                    "rel_res": rel_res_spsolve,
                    "sol_err": err_spsolve,
                    "iters": "N/A",
                }
            else:
                print(f"SciPy spsolve (baseline) failed. Time: {time_spsolve:.4f} s")
                results["Baseline spsolve"] = {
                    "time": time_spsolve,
                    "rel_res": np.nan,
                    "sol_err": np.nan,
                    "iters": "N/A",
                }
            print("-" * 70)

            # SciPy Baseline: splu (direct sparse LU)
            print("\n--- SciPy Baseline: splu (Sparse LU Factorization) ---")
            x_splu_scipy, time_splu_scipy, _ = solve_scipy_splu(A_sparse.copy(), b_np.copy())
            if x_splu_scipy is not None:
                rel_res_splu_scipy = calculate_relative_residual_norm(
                    A_sparse, x_splu_scipy, b_np
                )
                err_splu_scipy = calculate_solution_error(x_splu_scipy, x_true_np)
                print(f"SciPy splu Time: {time_splu_scipy:.4f} s")
                print(f"SciPy splu Relative Residual Norm: {rel_res_splu_scipy:.2e}")
                print(f"SciPy splu Solution Error (vs x_true): {err_splu_scipy:.2e}")
                results["Baseline splu"] = {
                    "time": time_splu_scipy,
                    "rel_res": rel_res_splu_scipy,
                    "sol_err": err_splu_scipy,
                    "iters": "N/A",
                }
            else:
                print(f"SciPy splu (baseline) failed. Time: {time_splu_scipy:.4f} s")
                results["Baseline splu"] = {
                    "time": time_splu_scipy,
                    "rel_res": np.nan,
                    "sol_err": np.nan,
                    "iters": "N/A",
                }
            print("-" * 70)

            # Method 1: Manual Gaussian Elimination
            print("\n--- Method 1: Manual Gaussian Elimination (Dense) ---")
            x_mge, time_mge, _ = solve_manual_gaussian_elimination(A_sparse.copy(), b_np.copy())
            if x_mge is not None:
                rel_res_mge = calculate_relative_residual_norm(A_sparse, x_mge, b_np)
                err_mge = calculate_solution_error(x_mge, x_true_np)
                print(f"Manual GE Time: {time_mge:.4f} s")
                print(f"Manual GE Relative Residual Norm: {rel_res_mge:.2e}")
                print(f"Manual GE Solution Error (vs x_true): {err_mge:.2e}")
                results["Manual GE (Method 1)"] = {
                    "time": time_mge,
                    "rel_res": rel_res_mge,
                    "sol_err": err_mge,
                    "iters": "N/A",
                }
            else:
                print(f"Manual GE (Method 1) failed or was skipped. Time: {time_mge:.4f} s")
                results["Manual GE (Method 1)"] = {
                    "time": time_mge,
                    "rel_res": np.nan,
                    "sol_err": np.nan,
                    "iters": "N/A",
                }
            print("-" * 70)

            # Method 2: Gauss-Seidel
            print("\n--- Method 2: Gauss-Seidel Iteration ---")
            gs_max_iters = (
                5000 if N <= 1000 else (2000 if N <= 10000 else 1000)
            )  # More iters for smaller, fewer for huge
            gs_tol = 1e-8  # Tighter tolerance
            x_gs, time_gs, iters_gs = solve_gauss_seidel(
                A_sparse.copy(), b_np.copy(), max_iters=gs_max_iters, tol=gs_tol
            )
            if x_gs is not None and not (np.any(np.isnan(x_gs)) or np.any(np.isinf(x_gs))):
                rel_res_gs = calculate_relative_residual_norm(A_sparse, x_gs, b_np)
                err_gs = calculate_solution_error(x_gs, x_true_np)
                print(f"Gauss-Seidel Time: {time_gs:.4f} s")
                print(f"Gauss-Seidel Iterations: {iters_gs}")
                print(f"Gauss-Seidel Relative Residual Norm: {rel_res_gs:.2e}")
                print(f"Gauss-Seidel Solution Error (vs x_true): {err_gs:.2e}")
                results["Gauss-Seidel (Method 2)"] = {
                    "time": time_gs,
                    "rel_res": rel_res_gs,
                    "sol_err": err_gs,
                    "iters": iters_gs,
                }
            else:
                print(
                    f"Gauss-Seidel failed or produced invalid solution. Time: {time_gs:.4f} s, Iterations: {iters_gs}"
                )
                results["Gauss-Seidel (Method 2)"] = {
                    "time": time_gs,
                    "rel_res": np.nan,
                    "sol_err": np.nan,
                    "iters": iters_gs,
                }
            print("-" * 70)

            # --- Summary of Results ---
            print("\n\n" + "=" * 30 + " Summary of Results " + "=" * 30)
            print(
                f"{'Method':<30} | {'Time (s)':<10} | {'Rel. Residual':<15} | {'Sol. Error':<15} | {'Iterations':<10}"
            )
            print("-" * 95)
            for method_name, res_vals in results.items():
                time_str = (
                    f"{res_vals['time']:.4f}"
                    if isinstance(res_vals["time"], (int, float))
                    else str(res_vals["time"])
                )
                rel_res_str = (
                    f"{res_vals['rel_res']:.2e}"
                    if isinstance(res_vals["rel_res"], (int, float))
                    and not np.isnan(res_vals["rel_res"])
                    else "N/A"
                )
                sol_err_str = (
                    f"{res_vals['sol_err']:.2e}"
                    if isinstance(res_vals["sol_err"], (int, float))
                    and not np.isnan(res_vals["sol_err"])
                    else "N/A"
                )
                iters_str = str(res_vals["iters"])
                print(
                    f"{method_name:<30} | {time_str:<10} | {rel_res_str:<15} | {sol_err_str:<15} | {iters_str:<10}"
                )
            print("=" * 95)

            all_results[(N,DENSITY)]=results


    # 打印所有结果
    for (N, DENSITY), res in all_results.items():
        print(f"\n\nResults for N={N}, Density={DENSITY}")
        print("=" * 30 + " Summary of Results " + "=" * 30)
        print(f"{'Method':<30} | {'Time (s)':<10} | {'Rel. Residual':<15} | {'Sol. Error':<15} | {'Iterations':<10}")
        print("-" * 95)
        for method_name, res_vals in res.items():
            time_str = f"{res_vals['time']:.4f}" if isinstance(res_vals['time'], (int, float)) else str(res_vals['time'])
            rel_res_str = f"{res_vals['rel_res']:.2e}" if isinstance(res_vals['rel_res'], (int, float)) and not np.isnan(res_vals['rel_res']) else "N/A"
            sol_err_str = f"{res_vals['sol_err']:.2e}" if isinstance(res_vals['sol_err'], (int, float)) and not np.isnan(res_vals['sol_err']) else "N/A"
            iters_str = str(res_vals['iters'])
            print(f"{method_name:<30} | {time_str:<10} | {rel_res_str:<15} | {sol_err_str:<15} | {iters_str:<10}")
        print("=" * 95)


    # 以下是非严格对角矩阵
    N = 500
    DENSITY = 0.05
    SEED = 10
    MAX_ITERS = 500

    print(f"\n=== Testing Matrix Properties (N={N}, Density={DENSITY}) ===")
    
    # 测试对角占优矩阵
    print("\n--- Testing Diagonally Dominant Matrix ---")
    A_dd, b_dd, x_true_dd = generate_sparse_diagonally_dominant_system(N, DENSITY, SEED)
    
    # 高斯-赛德尔迭代法
    print("\nGauss-Seidel Iteration:")
    x_gs_dd, residuals_dd, iter_dd = solve_gauss_seidel(
        A_dd, b_dd, max_iters=MAX_ITERS, tol=1e-10, verbose=True
    )
    
    # 计算指标
    rel_res_gs_dd = calculate_relative_residual_norm(A_dd, x_gs_dd, b_dd)
    err_gs_dd = calculate_solution_error(x_gs_dd, x_true_dd)
    
    print(f"\nGauss-Seidel (Diagonally Dominant):")
    print(f"  Iterations: {iter_dd}")
    print(f"  Relative Residual Norm: {rel_res_gs_dd:.2e}")
    print(f"  Solution Error: {err_gs_dd:.2e}")
    
    # 测试非对角占优矩阵
    print("\n--- Testing Non-Diagonally Dominant Matrix ---")
    A_ndd, b_ndd, x_true_ndd = generate_sparse_non_diagonally_dominant_system(N, DENSITY, SEED)
    
    # 高斯-赛德尔迭代法
    print("\nGauss-Seidel Iteration:")
    x_gs_ndd, iter_ndd, residuals_ndd = solve_gauss_seidel(
        A_ndd, b_ndd, max_iters=MAX_ITERS, tol=1e-10, verbose=True
    )
    
    # 计算指标
    rel_res_gs_ndd = calculate_relative_residual_norm(A_ndd, x_gs_ndd, b_ndd)
    err_gs_ndd = calculate_solution_error(x_gs_ndd, x_true_ndd)
    
    print(f"\nGauss-Seidel (Non-Diagonally Dominant):")
    print(f"  Iterations: {iter_ndd}")
    print(f"  Relative Residual Norm: {rel_res_gs_ndd:.2e}")
    print(f"  Solution Error: {err_gs_ndd:.2e}")
    
    # 绘制收敛曲线对比图
    plot_convergence(residuals_dd, residuals_ndd, N, DENSITY)
    
    # 对比结果
    print("\n=== Comparison Summary ===")
    print(f"Diagonally Dominant: Converged in {iter_dd} iterations")
    print(f"Non-Diagonally Dominant: Converged in {iter_ndd} iterations (or failed to converge)")
    print("\nNote: Diagonally dominant matrices guarantee convergence for Gauss-Seidel method,")
    print("while non-diagonally dominant matrices may not converge or require more iterations.")
