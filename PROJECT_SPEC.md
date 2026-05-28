# MGCP_Pro 项目环境规范

## 1. Conda 环境

**环境名称**: `MGC`

**路径**: `/home/ubuntu/miniconda3/envs/MGC/`

**Python 版本**: 3.10.20

### 激活方式

```bash
conda activate MGC
```

### 验证环境

```bash
conda run -n MGC python -c "import mgcp; print(mgcp.__version__)"
```

### 核心依赖

| 包 | 版本 | 用途 |
|---|---|---|
| numpy | 2.2.6 | 数值计算 |
| scipy | 1.15.3 | 科学计算 |
| galois | 0.4.11 | 有限域 / 纠错码 |
| reedsolo | 1.7.0 | Reed-Solomon 外码 |
| matplotlib | 3.10.9 | 绘图 |
| kalign-python | 3.5.1 | 多序列比对 (MSA) |
| py-cdhit | 1.1.5 | CD-HIT 聚类封装 |
| cd-hit | 4.8.1 | DNA 序列聚类 (bioconda) |
| pandas | 2.3.3 | 数据处理 |
| psutil | 7.2.2 | 系统资源监控 |
| tqdm | 4.67.3 | 进度条 |
| typer | 0.25.1 | CLI 框架 |

## 2. 项目目录

**根目录**: `/home/ubuntu/gongrui/MGCP_Pro/`

**MGCP 源码**: 通过 `pip install -e /home/ubuntu/gongrui/MGCP_Pro` 以 editable 模式安装，
CLI 入口点为 `mgcp`。

## 3. 权限与认证

| 项目 | 值 |
|---|---|
| sudo 密码 | `kk12345678` |
| sudo 路径 | `/home/ubuntu/miniconda3/bin/conda`（全局 conda 不在 sudo PATH 中）|

> 注意：需要 sudo 权限的操作（如通过 conda/bioconda 安装包）时，使用完整路径调用：
> ```bash
> echo "kk12345678" | sudo -S /home/ubuntu/miniconda3/bin/conda install -n MGC -c bioconda <package> -y
> ```

## 4. MGCP 项目说明

本项目基于 [ramy-khabbaz/MGCP](https://github.com/ramy-khabbaz/MGCP)，实现了 **MGC+ (Marker Guess & Check Plus)** 纠错码，
用于处理 DNA 存储和二进制通道中的插入、删除、替换 (IDS) 错误。

### CLI 子命令

```
mgcp --help
mgcp binary   # 二进制级编码/解码/绘图
mgcp dna      # DNA 级编码/解码/绘图
mgcp codec    # 文件级编码/解码
```

### 快速示例

```bash
# 二进制编码
mgcp binary encode "0101010011110110" 4 4 2

# DNA 编码
mgcp dna encode "0101010011110110" 4 4 0

# 文件编码 (max_length=120, inner_redundancy=4, outer_redundancy=200)
mgcp codec encode "data.bin" 120 4 200 --no-marker

# 文件解码 (4 进程并行)
mgcp codec decode "reads.txt" --processes 4

# FER vs 码率绘图
mgcp dna plot fer-vs-coderate 256 8 2 6,8,10,12,14 --pe 0.01 --num-iterations 1000
```

### 完整 Pipeline 演示

```bash
cd /home/ubuntu/gongrui/MGCP_Pro
conda run -n MGC python demo/demo_dna_pipeline.py
```

## 5. 规范

1. **所有新增项目或实验一律在 `MGC` 环境下运行**，禁止使用 base 或其他环境，避免依赖冲突。
2. 如需在 `MGC` 中添加新依赖，优先通过 `pip install`；如需 bioconda 包则通过 `sudo` + conda 安装。
3. sudo 命令中 conda 路径必须使用完整路径 `/home/ubuntu/miniconda3/bin/conda`。
4. 修改 `setup.cfg` 的 `install_requires` 后，需重新执行 `pip install -e .`。
