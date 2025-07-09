#!/bin/bash

NCCL_ALGO=Ring torchrun   --nnodes 2   --nproc_per_node 8  --node_rank $JOB_COMPLETION_INDEX  --master_addr=${MASTER_ADDR}  --master_port=${MASTER_PORT} finetune_model.py