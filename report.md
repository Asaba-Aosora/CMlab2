## task1
### 1. 实验过程


### 2. 结果分析
原始实验结果比较多，为了便于阅读，所以放到本部分的最后，以下只做概括分析
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

### 3. 完整实验结果
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