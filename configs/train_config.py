from dataclasses import dataclass

@dataclass
class TrainConfig:
    train_data_path: str = "dataset/MBTI 500_train.csv"
    eval_data_path: str = "dataset/MBTI 500_eval.csv"
    optimizer: str = "AdamW"
    model_name: str = "bert-base-uncased"
    # model_name: str = "meta-llama/Llama-2-7b-hf"
    batch_size_training: int = 32
    num_epochs: int = 10
    num_workers_dataloader: int = 1
    gamma: float = 0.85
    seed: int = 2
    val_batch_size: int = 1
    micro_batch_size: int = 32
    save_model: bool = True
    checkpoint_root_folder: str = "model_checkpoints"
    checkpoint_folder: str = "MBTI500"
    lr: float = 2e-5
    max_length: int = 512
    class_num = 16