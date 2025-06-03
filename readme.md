# 数值积分课程编程实验

欢迎来到本次实验的代码仓库！本次实验涵盖课程第6-8章的内容，对应的主程序文件为 `sovle_le.py`和 `flash_atten`文件夹。希望大家在探索数值分析的过程中，既能收获知识，也能享受编程的乐趣 😊。

---

## 实验环境

1. **Python 3.9+**  
   推荐使用 Python 3.9 或更高版本。

2. **PyTorch 2.0+**  
   建议安装 PyTorch 2.0 及以上版本。

3. **CUDA Toolkit**（可选）

   CUDA 版本需与 PyTorch 和 FlashAttention 库兼容，推荐使用 CUDA 11.x 或 12.x。

4. **FlashAttention 官方库**  
   推荐通过 PyPI 安装 `flash-attn` 包。需注意安装该库可能需要特定的编译环境及 GPU 架构（如 NVIDIA Ampere 系列）支持。若安装遇到困难，可将相关对比作为选做项或进行理论分析。

5.  **NumPy**
6.  **SciPy**

---

## 环境搭建
1. 如果你使用学院实验室计算机或者没有配置conda和cuda，可以按照以下步骤安装conda和cuda,否则跳过这一步。

首先进入Ubantu1系统
打开终端，依次输入以下命令
```bash
cd /home/nju/
# 学院服务器本地有miniconda.sh，如果没有请执行wget -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh首先下载
bash Miniconda3-latest-Linux-x86_64.sh  # 然后不断enter，输入yes，安装完成后重开一个终端
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
chmod 755 cuda_11.8.0_520.61.05_linux.run
mkdir cuda
sh cuda_11.8.0_520.61.05_linux.run
```
这时进入cuda安装页面
- 选择continue
- 输入accept
- 去除所有的x，只保留CUDA Toolkit
- 进入Options，然后选择Toolkit Options ，去掉所有的X
- 再选择Change Toolkit Install Path，输入/home/nju/cuda
- 回到Options，选择Library install path，修改为/home/nju/cuda
- 回到主界面，选择Install

安装完成后输入以下命令
```bash
echo 'export PATH=/home/nju/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/home/nju/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc         #安装完成
nvcc -V                  #输入此命令验证安装成功

```

2.配置虚拟环境
```bash
conda create -n Numerical_Analysis_02 python==3.10.14  # 创建虚拟环境
conda activate Numerical_Analysis_02                  # 激活虚拟环境
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia   # 安装pytorch
git clone https://git.nju.edu.cn/mq-yuan/numerical_analysis_2025.git --branch exp2
cd numerical_analysis_2025
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple                # 安装依赖
```
从[nju云盘](https://box.nju.edu.cn/d/283b5e195ea349cea037/)下载对应whl文件，或者其他版本在[flash-attn-linux](https://github.com/Dao-AILab/flash-attention/releases), [flash-attn-window](https://github.com/kingbri1/flash-attention/releases)下载，然后在终端执行
```bash
pip install flash_attn-2.7.3+cu11torch2.1cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
```

---
## 实验框架
本实验的代码框架已经搭建完成，主要包括以下几个部分：
- `solve_le.py`：实现高斯消去法和高斯-赛德尔迭代法求解线性方程组的代码。
- `flash_atten/`：实现 FlashAttention 的核心代码。
  - `main_test.py`：测试代码，包含对比实验的实现。
  - `atten_model.py`：实现对比实验中所需的模型。
  - `custom_flashaten.py`：需要实现的 FlashAttention 代码。
  - `flash_atten_backward.py`：需要实现 FlashAttention 的反向传播。
- `README.md`：实验说明文档。
- `requirements.txt`：依赖包列表。

## 伪代码
下面给出了 FlashAttention 的伪代码供参考。

   ***Forwoard***
   
   **Inputs:** $Q, K, V$, Tile sizes $T_q, T_k$. (Shapes: $Q \in \mathbb{R}^{B \times H \times N_q \times D_h}$, etc.). Let $s = 1/\sqrt{D_h}$ be the softmax scale.

   **Initialize:**
   $O_{full} = \mathbf{0}$ (tensor of zeros, shape $B \times H \times N_q \times D_h$)

   For each query tile $Q_i$ (from $Q$, with length up to $T_q$):
   1.  $S_i^{full} = \mathbf{0}$ (tensor of zeros, shape $B \times H \times \text{length}(Q_i) \times N_{kv}$)

   2.  For each key tile $K_j$ (from $K$, with length up to $T_k$):

         a.  **Calculate Partial Scores**: $S_{ij} = (Q_i \cdot K_j^T) \times s$
         
         b.  **Accumulate Scores**: Store $S_{ij}$ into the corresponding columns of $S_i^{full}$.
         (e.g., if $K_j$ covers columns $k_{start}$ to $k_{end}$ of $K$, then $S_i^{full}[:,:,:, k_{start}:k_{end}] = S_{ij}$)

   3.  $P_i = \text{softmax}(S_i^{full}, \text{dim}=-1)$
   4.  $O_i = P_i \cdot V$ (Matrix multiply $P_i$ with the full $V$ tensor)
   5.  Place $O_i$ into the corresponding rows of $O_{full}$.

   **Return:** $O_{full}$
   
   ---
   ***Backward***
   
   **Inputs:** $dO, Q, K, V, T_q, T_k$. (Let $s = 1/\sqrt{D_h}$ be softmax scale. Shapes: $dO, Q \in \mathbb{R}^{B \times H \times N_q \times D_h}$; $K, V \in \mathbb{R}^{B \times H \times N_{kv} \times D_h}$)

   **Initialize:** $dQ = \mathbf{0}, dK = \mathbf{0}, dV = \mathbf{0}$ (zero tensors with shapes of $Q, K, V$ respectively).

   For each query tile $Q_i$ (from $Q$, with length $N_{q\_curr}$ up to $T_q$) and corresponding $dO_i$ (from $dO$):

   6.  **Recompute attention scores $S_i^{full}$ and probabilities $P_i$ for $Q_i$ against all $K$**:
      
         a.  $S_i^{full} = \mathbf{0}$ (tensor of zeros, shape $B \times H \times N_{q\_curr} \times N_{kv}$)
         
         b.  For each key tile $K_j$ (from $K$, with length up to $T_k$):

         i.  $S_{ij} = (Q_i \cdot K_j^T) \cdot s$
      
         ii. Store $S_{ij}$ into the corresponding columns of $S_i^{full}$.
         (e.g., if $K_j$ covers columns $k_{start}$ to $k_{end}$ of $K$, then $S_i^{full}[:,:,:, k_{start}:k_{end}] = S_{ij}$)
         
         c.  $P_i = \text{softmax}(S_i^{full}, \text{dim}=-1)$

   7.  **Compute and Accumulate Gradients**:
      
         a.  $dV = dV + (P_i^T \cdot dO_i)$
         
         b.  $dP_i = dO_i \cdot V^T$
         
         c.  $dS_i = P_i \odot (dP_i - \sum_{\text{last dim}}(dP_i \odot P_i, \text{keepdim=True}))$
            ($\odot$ denotes element-wise product)
         
         d.  $dQ_i^{\text{block}} = (dS_i \cdot K) \cdot s$. Place $dQ_i^{\text{block}}$ into the rows of $dQ$ corresponding to $Q_i$.
         
         e.  $dK = dK + (dS_i^T \cdot Q_i) \cdot s$

   **Return:** $dQ, dK, dV$

---

最终，你需要提交本仓库代码的打包文件以及实验报告，提交链接见课程主页。加油！🚀
