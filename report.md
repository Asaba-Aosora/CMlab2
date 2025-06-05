## task1
### 1. 实验过程
补全了`sovle_le.py`里的`solve_manual_gaussian_elimination`函数和`solve_gauss_seidel`函数，并且在`main.py`里补充了对比不同规模、不同稀疏和非严格对角占优矩阵的逻辑。

- `solve_manual_gaussian_elimination`函数：
  - 先把矩阵转化为稠密矩阵，然后采用高斯主元素消去法，首先选出每一列中绝对值最大的元素，然后通过行交换、把它交换到对角线上，得到 \(a_{ii}\)。然后按照公式 \(a_{jk}^{(new)}=a_{jk}-\frac{a_{ji}}{a_{ii}}a_{ik}\)，消去它下方的元素。经过正向消元后，增广矩阵 \([A|b]\) 变为上三角矩阵 \([U|c]\)，其中 U 是上三角矩阵，c 是新的常数向量。然后进行回代求解。首先，有 \(x_n\)：\(x_n=\frac{c_n}{u_{nn}}\)。然后，对于 \(i = n - 1, n - 2, \cdots, 1\)，求解 \(x_i\)：\(x_i=\frac{c_i-\sum_{j = i + 1}^{n}u_{ij}x_j}{u_{ii}}\)。
  - 此外还添加了一些异常判断
  - 核心代码如下：
  ```python
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
    ```

- `solve_manual_gaussian_elimination`函数：
  - 根据高斯 - 赛德尔迭代公式 \(x_i^{(k+1)} = \frac{b_i - \sum_{j=1}^{i-1} a_{ij}x_j^{(k+1)} - \sum_{j=i+1}^{n} a_{ij}x_j^{(k)}}{a_{ii}}\)，更新第 i 个未知数的值。
  - 然后检查收敛性，计算相邻两次迭代结果的相对误差。当相对误差小于收敛阈值时结束迭代，未收敛时继续迭代。
  - 核心代码如下：
    ```python
    for iters in range(max_iters):
    for i in range(n):
        # 计算非对角元素的贡献
        row = A_csc.getrow(i).toarray()[0]
        non_diag_sum = np.dot(row, x) - row[i] * x[i]
        # 迭代更新
        x[i] = (b[i] - non_diag_sum) / diag[i]

    # 检查收敛性
    if iters > 0:
        rel_error = np.linalg.norm(x - x_prev) / (np.linalg.norm(x) + 1e-10)
        if rel_error < tol:
            break
        x_prev = x.copy()
    ```


### 2. 结果分析
原始实验结果比较多，为了便于阅读，所以放到实验报告的最后，以下只做概括分析
#### 1. 严格对角占优时
参数设置如下
```python
    N_values = [100, 500, 2000]
    DESITY_values = [0.01, 0.05, 0.1]
    SEED = 10
```
##### 速度对比

`SciPy`方法中，N=100时spsolve和splu都很快，约0.0003s。N=2000时两种方法耗时相近，大约1s。稀疏度增加时，两种方法耗时略有增加，但是变化幅度不大。

`Manual GE`方法中，N=100时耗时约0.01s，N=2000时耗时增至9.9s，$O(N^3)$的时间复杂度是导致时间快速增大的原因。稀疏度对该方法的影响不大，因为它将稀疏矩阵转为密集矩阵处理，计算量始终为$O(N^3)$

`Gauss-Seidel`方法中，N=100时耗时约0.05s，N=2000时耗时在6-23s。在高稀疏度下，耗时显著增加。都是对角占优矩阵，所以迭代次数几乎不变，但是每次迭代的计算量会增大

##### 精度对比

`SciPy`方法和`Manual GE`方法中，相对残差范数和解误差均为$10^{-15}$量级，误差非常小，表明直接法的精度很高。

`Gauss-Seidel`方法里，相对残差范数和解误差均为$10^{-10}$量级，精度逊于直接法。

#### 2. 非严格对角占优时
对角占优矩阵和非严格对角占优矩阵都是在第16次迭代后显示收敛。但是，非严格对角占优矩阵的相对残差范数、求解误差都略大。
```text
Gauss-Seidel (Diagonally Dominant):
  Iterations: 16
  Relative Residual Norm: 3.56e-12
  Solution Error: 5.44e-12

Gauss-Seidel (Non-Diagonally Dominant):
  Iterations: 16
  Relative Residual Norm: 1.30e-11
  Solution Error: 2.36e-11
```

## task2


## 完整实验结果
### task1
#### 1. 严格对角占优时
参数如下：
```python
    N_values = [100, 500, 2000]
    DESITY_values = [0.01, 0.05, 0.1]
    SEED = 10
```

结果如下：
```text
Results for N=100, Density=0.01
============================== Summary of Results ==============================
Method                         | Time (s)   | Rel. Residual   | Sol. Error      | Iterations
-----------------------------------------------------------------------------------------------
Baseline spsolve               | 0.0003     | 6.12e-17        | 1.17e-16        | N/A       
Baseline splu                  | 0.0003     | 4.80e-17        | 1.83e-16        | N/A       
Manual GE (Method 1)           | 0.0119     | 8.46e-17        | 1.78e-16        | N/A       
Gauss-Seidel (Method 2)        | 0.0464     | 3.96e-11        | 6.67e-11        | 5         
===============================================================================================


Results for N=100, Density=0.05
============================== Summary of Results ==============================
Method                         | Time (s)   | Rel. Residual   | Sol. Error      | Iterations
-----------------------------------------------------------------------------------------------
Baseline spsolve               | 0.0005     | 2.49e-16        | 3.57e-16        | N/A       
Baseline splu                  | 0.0006     | 2.38e-16        | 3.69e-16        | N/A       
Manual GE (Method 1)           | 0.0117     | 2.52e-16        | 4.07e-16        | N/A       
Gauss-Seidel (Method 2)        | 0.1178     | 9.05e-10        | 2.30e-09        | 14        
===============================================================================================


Results for N=100, Density=0.1
============================== Summary of Results ==============================
Method                         | Time (s)   | Rel. Residual   | Sol. Error      | Iterations
-----------------------------------------------------------------------------------------------
Baseline spsolve               | 0.0006     | 2.78e-16        | 4.94e-16        | N/A       
Baseline splu                  | 0.0007     | 3.18e-16        | 5.70e-16        | N/A       
Manual GE (Method 1)           | 0.0121     | 3.07e-16        | 5.64e-16        | N/A       
Gauss-Seidel (Method 2)        | 0.1098     | 4.99e-10        | 9.35e-10        | 12        
===============================================================================================


Results for N=500, Density=0.01
============================== Summary of Results ==============================
Method                         | Time (s)   | Rel. Residual   | Sol. Error      | Iterations
-----------------------------------------------------------------------------------------------
Baseline spsolve               | 0.0092     | 4.67e-16        | 8.15e-16        | N/A       
Baseline splu                  | 0.0097     | 3.94e-16        | 9.50e-16        | N/A       
Manual GE (Method 1)           | 0.3814     | 4.56e-16        | 9.66e-16        | N/A       
Gauss-Seidel (Method 2)        | 0.7291     | 5.59e-10        | 9.49e-10        | 14        
===============================================================================================


Results for N=500, Density=0.05
============================== Summary of Results ==============================
Method                         | Time (s)   | Rel. Residual   | Sol. Error      | Iterations
-----------------------------------------------------------------------------------------------
Baseline spsolve               | 0.0205     | 6.61e-16        | 1.13e-15        | N/A       
Baseline splu                  | 0.0192     | 5.72e-16        | 1.02e-15        | N/A       
Manual GE (Method 1)           | 0.5289     | 6.97e-16        | 1.24e-15        | N/A       
Gauss-Seidel (Method 2)        | 1.3098     | 3.40e-10        | 4.85e-10        | 13        
===============================================================================================


Results for N=500, Density=0.1
============================== Summary of Results ==============================
Method                         | Time (s)   | Rel. Residual   | Sol. Error      | Iterations
-----------------------------------------------------------------------------------------------
Baseline spsolve               | 0.0404     | 6.79e-16        | 1.25e-15        | N/A       
Baseline splu                  | 0.0530     | 5.68e-16        | 1.03e-15        | N/A       
Manual GE (Method 1)           | 0.6623     | 7.33e-16        | 1.32e-15        | N/A       
Gauss-Seidel (Method 2)        | 0.9441     | 3.97e-10        | 5.37e-10        | 13        
===============================================================================================


Results for N=2000, Density=0.01
============================== Summary of Results ==============================
Method                         | Time (s)   | Rel. Residual   | Sol. Error      | Iterations
-----------------------------------------------------------------------------------------------
Baseline spsolve               | 0.8624     | 1.41e-15        | 2.48e-15        | N/A       
Baseline splu                  | 0.8533     | 1.10e-15        | 1.91e-15        | N/A       
Manual GE (Method 1)           | 9.9766     | 1.22e-15        | 2.28e-15        | N/A       
Gauss-Seidel (Method 2)        | 6.3787     | 2.58e-10        | 3.78e-10        | 13        
===============================================================================================


Results for N=2000, Density=0.05
============================== Summary of Results ==============================
Method                         | Time (s)   | Rel. Residual   | Sol. Error      | Iterations
-----------------------------------------------------------------------------------------------
Baseline spsolve               | 1.2251     | 1.57e-15        | 2.82e-15        | N/A       
Baseline splu                  | 0.8577     | 1.05e-15        | 1.86e-15        | N/A       
Manual GE (Method 1)           | 9.9572     | 1.41e-15        | 2.56e-15        | N/A       
Gauss-Seidel (Method 2)        | 14.4722    | 4.83e-10        | 6.37e-10        | 13        
===============================================================================================


Results for N=2000, Density=0.1
============================== Summary of Results ==============================
Method                         | Time (s)   | Rel. Residual   | Sol. Error      | Iterations
-----------------------------------------------------------------------------------------------
Baseline spsolve               | 0.8059     | 1.62e-15        | 2.86e-15        | N/A       
Baseline splu                  | 1.0760     | 1.16e-15        | 2.06e-15        | N/A       
Manual GE (Method 1)           | 9.6568     | 1.44e-15        | 2.61e-15        | N/A       
Gauss-Seidel (Method 2)        | 23.7396    | 5.02e-10        | 6.63e-10        | 13        
===============================================================================================
```
#### 2. 非严格对角占优时
参数如下：
```python
    N = 500
    DENSITY = 0.05
    SEED = 10
    MAX_ITERS = 500
```
结果如下：
```text
=== Testing Matrix Properties (N=500, Density=0.05) ===

--- Testing Diagonally Dominant Matrix ---
Diagonally dominant matrix A ((500, 500)) generated with 12972 non-zero elements.

Gauss-Seidel Iteration:

Gauss-Seidel (Diagonally Dominant):
  Iterations: 16
  Relative Residual Norm: 3.56e-12
  Solution Error: 5.44e-12

--- Testing Non-Diagonally Dominant Matrix ---
Non-diagonally dominant matrix A ((500, 500)) generated with 12972 non-zero elements.

Gauss-Seidel Iteration:

Gauss-Seidel (Non-Diagonally Dominant):
  Iterations: 16
  Relative Residual Norm: 1.30e-11
  Solution Error: 2.36e-11

=== Comparison Summary ===
Diagonally Dominant: Converged in 16 iterations
Non-Diagonally Dominant: Converged in 16 iterations (or failed to converge)

Note: Diagonally dominant matrices guarantee convergence for Gauss-Seidel method,
while non-diagonally dominant matrices may not converge or require more iterations.

```