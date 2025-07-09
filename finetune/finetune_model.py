import datetime
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.distributed.nn as distnn

from tqdm import tqdm
import sys

# import wandb
import os
from safetensors.torch import load_model, save_file
from model_utils import *
from torch.utils.data import DataLoader
import json
from torch.utils.checkpoint import checkpoint
from lora_config import LoraConfig

# DEFINE THIS AS EOS OR WHATEVER YOU WANT
pad_id = 0  # tokenizer.eos_token_id


class Transformer(nn.Module):
    """
    Transformer model with positional embeddings, multiple layers, and output projection.

    Attributes:
        max_seq_len (int): Maximum sequence length for the transformer.
        embed (nn.Module): Embedding layer for input tokens.
        layers (torch.nn.ModuleList): List of transformer blocks.
        norm (nn.Module): Layer normalization applied after all blocks.
        head (nn.Module): Output projection layer mapping to vocabulary size.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
    """

    def __init__(self, args: ModelArgs, lora_config: LoraConfig = None):
        """
        Initializes the Transformer model.

        Args:
            args (ModelArgs): Model arguments containing transformer parameters.
        """
        Linear.dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
        super().__init__()
        self.lora_config = lora_config
        self.max_seq_len = args.max_seq_len
        self.embed = ParallelEmbedding(args.vocab_size, args.dim)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(Block(layer_id, args, lora_config=lora_config))
        self.norm = RMSNorm(args.dim)
        self.head = ColumnParallelLinear(
            args.dim,
            args.vocab_size,
            dtype=torch.get_default_dtype(),
        )
        self.register_buffer("freqs_cis", precompute_freqs_cis(args), persistent=False)

    def forward(
        self,
        tokens: torch.Tensor,
        dummy: torch.Tensor,
        attn_mask=None,
        start_pos=0,
        use_lora=True,
    ):
        """
        Forward pass for the Transformer model.

        Args:
            tokens (torch.Tensor): Input tensor of token IDs with shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, vocab_size).
        """
        assert dummy is not None
        h = self.embed(tokens)

        # -1 for 0-based
        pos = (tokens != pad_id).cumsum(dim=1) - 1
        freqs_cis = self.freqs_cis[start_pos : start_pos + pos.max() + 1][
            pos
        ].masked_fill((tokens == pad_id).unsqueeze(-1), 0)

        for layer in self.layers:
            h = checkpoint(
                layer,
                h,
                freqs_cis,
                dummy,
                attn_mask,
                start_pos,
                use_lora,
                use_reentrant=True,
            )

        logits = self.head(self.norm(h))
        logits = distnn.all_gather(logits)
        logits = torch.cat(logits, dim=-1)

        return logits


def get_model(device="cuda", lora_config: LoraConfig = None):
    with open("./configs/config_671B.json") as f:
        default_args = ModelArgs(**json.load(f))
    with torch.device(device):
        model = Transformer(default_args, lora_config=lora_config)
        return model


def save_lora_only(safetensor_path, model: Transformer, rank, world_size):
    sd = model.state_dict()
    lora_sd = {k: v for k, v in sd.items() if "lora" in k}
    save_file(
        lora_sd, os.path.join(safetensor_path, f"lora-{rank}-{world_size}.safetensors")
    )


def train(ckpt_sharded_path, output_path, dataset_path, lora_config: LoraConfig):
    # hyperparameters
    num_epochs = 4
    lr = 3e-4
    batch_size = 16
    mb_sz = batch_size
    mbs = batch_size // mb_sz
    num_logging_steps = 20

    train_with_kl = False
    kl_weight = 0.04

    world_size = int(os.getenv("WORLD_SIZE"))

    dist.init_process_group(
        "nccl",
        timeout=datetime.timedelta(minutes=30),
        world_size=world_size,
        init_method="env://",
    )
    torch.set_num_threads(8)
    torch.set_default_dtype(torch.bfloat16)
    local_rank = int(os.getenv("LOCAL_RANK"))

    rank = int(os.getenv("RANK"))

    torch.cuda.set_device(local_rank)
    shard_path = os.path.join(
        f"{ckpt_sharded_path}/model{rank}-mp{world_size}.safetensors"
    )
    model = get_model("cpu", lora_config)
    model.train()
    load_model(model, shard_path, strict=False, device="cpu")

    for name, module in model.named_children():
        module.to(torch.cuda.current_device())
    model = model.to(torch.cuda.current_device())
    print(f"Rank {rank} loaded model")

    dataset_tensor = torch.load(dataset_path, weights_only=False)
    dataloader = DataLoader(dataset_tensor, batch_size=batch_size)

    model.train()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=lr,
        fused=True,
        weight_decay=0.0,
    )

    # this is required for the checkpoint() function to work
    dummy = torch.tensor(2.0).to(torch.cuda.current_device())
    dummy.requires_grad = True
    global_step = 0

    for _ in tqdm(range(num_epochs), disable=(rank != 0), file=sys.stdout):
        for (
            all_tokens_global,
            attention_masks_global,
            assistant_masks_global,
        ) in dataloader:
            optimizer.zero_grad(set_to_none=True)

            prompt_tokens_global = all_tokens_global[:, :-1]
            positions = 0

            assistant_masks_global = assistant_masks_global[:, :-1].to(
                torch.cuda.current_device()
            )
            attention_masks_global = attention_masks_global[:, :-1].to(
                torch.cuda.current_device()
            )
            prompt_tokens_global = prompt_tokens_global.to(
                torch.cuda.current_device()
            ).contiguous()
            targets_global = (
                all_tokens_global[:, 1:].to(torch.cuda.current_device()).contiguous()
            )

            for i in range(mbs):
                mb_start = i * mb_sz
                mb_end = mb_start + mb_sz

                if mb_end > prompt_tokens_global.shape[0]:
                    break

                assistant_masks = assistant_masks_global[mb_start:mb_end]
                attention_masks = attention_masks_global[mb_start:mb_end]
                prompt_tokens = prompt_tokens_global[mb_start:mb_end]
                targets = targets_global[mb_start:mb_end]

                logits = model(
                    prompt_tokens,
                    dummy,
                    attn_mask=attention_masks,
                    start_pos=positions,
                    use_lora=True,
                )

                loss = 0
                nll = None
                kl = None

                element_loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    reduction="none",
                )

                element_loss = element_loss.view(mb_sz, -1)

                loss_mask = assistant_masks.view(mb_sz, -1)
                sample_loss = (element_loss * loss_mask).sum(1) / (
                    loss_mask.sum(1) + 1e-8
                )

                nll = sample_loss.mean()
                loss = nll

                if train_with_kl:
                    with torch.no_grad():
                        base_logits = model(
                            prompt_tokens,
                            dummy,
                            attn_mask=attention_masks,
                            start_pos=0,
                            use_lora=False,
                        )

                        base_logp = F.log_softmax(base_logits, dim=-1)

                        kl_mask: torch.Tensor = assistant_masks.unsqueeze(-1)

                    log_p = F.log_softmax(logits, dim=-1)

                    kl = (log_p.exp() * (log_p - base_logp) * kl_mask).sum() / (
                        kl_mask.sum() + 1e-8
                    )

                    loss = loss + kl_weight * kl

                if rank == 0 and global_step % num_logging_steps == 0:
                    # add your own wandb logging here
                    """
                    wandb.log(
                        {
                            "nll": nll.item() if nll is not None else 0,
                            "kl": kl.item() if kl is not None else 0,
                            "loss": loss.item(),
                        }
                    )
                    """
                    print(f"Loss : {loss.item()}")

                loss.backward()

            nn.utils.clip_grad_norm_(trainable_params, 0.2)
            optimizer.step()
            global_step += 1

    torch.cuda.empty_cache()
    save_lora_only(output_path, model, rank, world_size)
    if rank == 0:
        print("Saved LoRA weights")

    torch.save(optimizer.state_dict(), f"{output_path}/checkpoint-{rank}.pt")


if __name__ == "__main__":
    lora_config = LoraConfig(
        lora_rank=16,
        lora_alpha=32,
        lora_trainable_modules=[
            "q_proj",
            "kv_proj",
            "o_proj",
            "gate_proj",
            "down_proj",
            "up_proj",
        ],
        lora_dropout=0.05,
        max_train_seq_len=512,
    )

    train("weights_sharded", "weights_finetune", "test.pt", lora_config)
    dist.barrier()
    dist.destroy_process_group()
