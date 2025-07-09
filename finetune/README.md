### DeepSeek-V3 671B LoRA finetuning code

This is a LoRA SFT framework for DeepSeek-V3 671B. Yes, the large one.

The LoRA configuration code can be found in `finetune/finetune_model.py`. To run, use the `finetune/run.sh` script which then calls torchrun.

You need a minimum of 16 H100s to load the model.

The dataset must contain three fields:

1. The tokens to train on 
2. 1D attention masks
3. A mask for the user chat tokens