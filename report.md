## task1
```text
--- Generating System (N=1000, Density=0.01) ---
Diagonally dominant matrix A ((1000, 1000)) generated with 10992 non-zero elements.
Norm of b: 1.85e+02, Norm of x_true: 1.85e+01
----------------------------------------------------------------------

--- SciPy Baseline: spsolve ---
SciPy spsolve Time: 0.0996 s
SciPy spsolve Relative Residual Norm: 8.38e-16
SciPy spsolve Solution Error (vs x_true): 1.47e-15
----------------------------------------------------------------------

--- SciPy Baseline: splu (Sparse LU Factorization) ---
SciPy splu Time: 0.0972 s
SciPy splu Relative Residual Norm: 7.54e-16
SciPy splu Solution Error (vs x_true): 1.34e-15
----------------------------------------------------------------------

--- Method 1: Manual Gaussian Elimination (Dense) ---
Manual GE Time: 1.7632 s
Manual GE Relative Residual Norm: 7.64e-16
Manual GE Solution Error (vs x_true): 1.38e-15
----------------------------------------------------------------------

--- Method 2: Gauss-Seidel Iteration ---
Gauss-Seidel Time: 1.8164 s
Gauss-Seidel Iterations: 13
Gauss-Seidel Relative Residual Norm: 3.04e-10
Gauss-Seidel Solution Error (vs x_true): 5.54e-10
----------------------------------------------------------------------


============================== Summary of Results ==============================
Method                         | Time (s)   | Rel. Residual   | Sol. Error      | Iterations
-----------------------------------------------------------------------------------------------
Baseline spsolve               | 0.0996     | 8.38e-16        | 1.47e-15        | N/A       
Baseline splu                  | 0.0972     | 7.54e-16        | 1.34e-15        | N/A       
Manual GE (Method 1)           | 1.7632     | 7.64e-16        | 1.38e-15        | N/A       
Gauss-Seidel (Method 2)        | 1.8164     | 3.04e-10        | 5.54e-10        | 13        
===============================================================================================
```