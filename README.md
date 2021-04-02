# Implmenetation of Deep Self-Evolution Clustering in PyTorch（IEEE T-PAMI）
A general clustering method for speech, image and text clustering.

# Setup
1. Create virtual environment: python -m venv dsec-env
2. Activate virtual environment: source dsec-env/Scripts/activate or source dsec-env/bin/activate for Linux
3. Install dependencies using pip install -r requirements.txt

If there are issues installing pytorch versions, use the following command
`pip install torch===1.7.0+cu110 torchvision==0.8.1+cu110 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html`

# Compute Canada Commands

In order to see current jobs in the queue: `sq`

In order to add a job to the queue: `sbatch <bash script> <python script to pass as argument to the batch script>`

In order to cancel a job in the queue: `scancel <job_id>`

# Logs
Our current best model utilizes DA with 15 epochs, with normalization followed by DSEC without normalization.

It would be a good idea to test the model with normal weight initialization for conv2d layers and xavier_uniform for linear layers.

Definitely also a good idea to test more epochs for dsec


## Bibtex
Credits to original author: Jianlong Chang (jianlong.chang@nlpr.ia.ac.cn)
```
@InProceedings{Chang_2018_PAMI,
author = {Chang, Jianlong and Meng, Gaofeng and Wang, Lingfeng and Xiang, Shiming and Pan, Chunhong},
title = {Local-Aggregation Graph Networks},
booktitle = {IEEE Transactions on Pattern Analysis and Machine Intelligence}
}
```

