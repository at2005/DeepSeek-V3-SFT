# DeepSeek-V3 671B LoRA SFT Framework 

This is a LoRA SFT framework for DeepSeek-V3 671B. Yes, the large one.

The LoRA configuration code can be found in `finetune_model.py`. To start training, adapt this configuration and run the `run.sh` script on each node.

### Resource Constraints
You need a minimum of 16 H100s to load the model.

### Dataset
The dataset must contain three fields:

1. The tokens to train on 
2. 1D attention masks
3. A mask for the user chat tokens