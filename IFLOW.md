# LETTER 项目指南

## 项目概述

LETTER (LEarnable Tokenizer for generaTivE Recommendation) 是一个用于生成式推荐的可学习项目标记器。该项目基于论文 [Learnable Item Tokenization for Generative Recommendation](https://arxiv.org/abs/2405.07314) 实现，集成了层次语义、协作信号和代码分配多样性，满足标识符的基本需求。

LETTER 包含三个主要组件：
- **RQ-VAE**: 残差量化变分自编码器，用于项目标记化
- **LETTER-TIGER**: 基于 T5 模型的生成式推荐系统实例化
- **LETTER-LC-Rec**: 基于 LLaMA 模型的大规模序列推荐系统实例化

## 项目结构

```
LETTER/
├── README.md                  # 项目说明文档
├── requirements.txt           # Python 依赖包列表
├── data/                      # 数据集目录
│   ├── Beauty/               # Beauty 数据集
│   ├── Instruments/          # Instruments 数据集
│   └── Yelp/                 # Yelp 数据集
├── RQ-VAE/                   # 残差量化变分自编码器实现
│   ├── main.py               # 训练主程序
│   ├── models/               # 模型定义
│   ├── trainer.py            # 训练器
│   ├── train_tokenizer.sh    # 训练标记器脚本
│   └── tokenize.sh           # 标记化脚本
├── LETTER-TIGER/             # T5 模型实例化
│   ├── finetune.py           # 微调脚本
│   ├── modeling_letter.py    # LETTER 模型实现
│   ├── data.py               # 数据处理
│   ├── collator.py           # 数据整理器
│   ├── evaluate.py           # 评估脚本
│   └── run_train.sh          # 训练脚本
└── LETTER-LC-Rec/            # LLaMA 模型实例化
    ├── lora_finetune.py      # LoRA 微调脚本
    ├── modeling_letter.py    # LETTER 模型实现
    ├── data.py               # 数据处理
    ├── collator.py           # 数据整理器
    ├── evaluate.py           # 评估脚本
    └── run_train.sh          # 训练脚本
```

## 环境配置

安装项目依赖：
```bash
pip install -r requirements.txt
```

主要依赖包括：
- torch==1.13.1+cu117
- transformers
- accelerate
- bitsandbytes
- deepspeed
- peft
- sentencepiece
- datasets
- evaluate
- tqdm

## 使用流程

### 1. 训练 RQ-VAE 标记器

首先训练项目标记器：

```bash
cd RQ-VAE
bash train_tokenizer.sh
```

训练脚本参数说明：
- `--device`: 指定 GPU 设备 (如 cuda:0)
- `--data_path`: 数据路径 (如 ../data/Instruments/Instruments.emb-llama-td.npy)
- `--alpha`: 对比损失权重 (默认 0.01)
- `--beta`: 多样性损失权重 (默认 0.0001)
- `--cf_emb`: 协作嵌入路径
- `--ckpt_dir`: 模型检查点保存目录

### 2. 数据标记化

使用训练好的标记器对数据集进行标记化：

```bash
cd RQ-VAE
bash tokenize.sh
```

标记化脚本参数：
- `--dataset`: 数据集名称 (如 Beauty)
- `--alpha`: 对比损失权重
- `--beta`: 多样性损失权重
- `--epoch`: 训练轮数
- `--checkpoint`: 模型检查点路径

### 3. 训练 LETTER-TIGER

基于 T5 模型训练生成式推荐系统：

```bash
cd LETTER-TIGER
bash run_train.sh
```

训练脚本参数：
- `DATASET`: 数据集名称 (如 Instruments)
- `OUTPUT_DIR`: 模型输出目录
- `--per_device_batch_size`: 每设备批次大小
- `--learning_rate`: 学习率
- `--epochs`: 训练轮数
- `--index_file`: 索引文件
- `--temperature`: 生成温度参数

### 4. 训练 LETTER-LC-Rec

基于 LLaMA 模型训练大规模序列推荐系统：

```bash
cd LETTER-LC-Rec
bash run_train.sh
```

训练脚本参数：
- `DATASET`: 数据集名称
- `BASE_MODEL`: 基础模型路径 (LLaMA)
- `DATA_PATH`: 数据路径
- `OUTPUT_DIR`: 模型输出目录
- `--per_device_batch_size`: 每设备批次大小
- `--learning_rate`: 学习率
- `--epochs`: 训练轮数
- `--tasks`: 任务类型 (seqrec)
- `--index_file`: 索引文件
- `--temperature`: 生成温度参数

## 数据集格式

项目支持三种数据集格式，每个数据集包含三个文件：
- `{Dataset}.index.json`: 项目索引信息
- `{Dataset}.inter.json`: 用户-项目交互数据
- `{Dataset}.item.json`: 项目特征数据

## 模型评估

使用相应的评估脚本评估模型性能：

```bash
# LETTER-TIGER 评估
cd LETTER-TIGER
python evaluate.py --args...

# LETTER-LC-Rec 评估
cd LETTER-LC-Rec
python evaluate.py --args...
```

## 分布式训练

项目支持分布式训练，使用 torchrun 进行多 GPU 训练：

```bash
# LETTER-TIGER 分布式训练 (2 GPU)
torchrun --nproc_per_node=2 --master_port=2314 ./finetune.py [参数...]

# LETTER-LC-Rec 分布式训练 (4 GPU)
torchrun --nproc_per_node=4 --master_port=3325 ./lora_finetune.py [参数...]
```

## 注意事项

1. 训练前确保数据集已正确放置在 `data/` 目录下
2. LETTER-LC-Rec 需要设置 `BASE_MODEL` 变量指向 LLaMA 模型路径
3. 训练过程中会自动创建检查点目录
4. 可通过设置 `WANDB_MODE=disabled` 禁用 wandb 日志记录
5. 训练脚本支持通过环境变量控制 GPU 使用 (`CUDA_VISIBLE_DEVICES`)

## 引用

如果本项目对您的研究有帮助，请引用：

```
@inproceedings{wang2024learnableitemtokenizationgenerative,
  title = {Learnable Item Tokenization for Generative Recommendation},
  author = {Wang, Wenjie and Bao, Honghui and Lin, Xinyu and Zhang, Jizhi and Li, Yongqi and Feng, Fuli and Ng, See-Kiong and Chua, Tat-Seng},
  booktitle = {International Conference on Information and Knowledge Management},
  year = {2024}
}
```