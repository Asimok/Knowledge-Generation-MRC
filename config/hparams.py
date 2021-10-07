from collections import defaultdict

PARAMS = defaultdict(
    # Environment
    device='cuda',
    dataset_dir='../dataset/',
    bert_dir="/data0/wangbingchao/pretrained/torch/chinese-roberta-wwm-ext",
    workers=24,
    gpu_ids=[0],
    save_dirpath='../checkpoints/',
    # Training Hyperparemter
    batch_size=1,
    num_epochs=100,
    buffer_size=5000,
    bucket_width=5,
    max_length=500,
    max_episode_length=5,
    max_knowledge=32,
    knowledge_truncate=34,
    pad_to_max=True,

)
