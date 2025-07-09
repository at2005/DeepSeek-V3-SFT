class LoraConfig:
    def __init__(
        self,
        lora_rank: int,
        lora_alpha: int,
        lora_dropout: float,
        lora_trainable_modules: set[str],
        max_train_seq_len: int,
    ):
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_trainable_modules = lora_trainable_modules
        self.max_train_seq_len = max_train_seq_len

    def get_lora_key(self, module_name):
        return True if module_name in self.lora_trainable_modules else False
