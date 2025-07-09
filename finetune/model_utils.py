import math
from dataclasses import dataclass
from typing import Optional, Literal, Tuple

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.distributed.nn as distnn
from lora_config import LoraConfig

from kernel import (
    CustomActQuant,
    CustomFP8GEMM,
    CustomWeightDequant,
    CustomWeightQuant,
)

custom_gemm = CustomFP8GEMM.apply
custom_act_quant = CustomActQuant.apply
custom_weight_dequant = CustomWeightDequant.apply
custom_weight_quant = CustomWeightQuant.apply

block_size = 128
gemm_impl: Literal["bf16", "fp8"] = "bf16"  # "fp8"
attn_impl: Literal["naive", "absorb"] = "absorb"


@dataclass
class ModelArgs:
    """
    Data class for defining model arguments and hyperparameters.

    Attributes:
        max_batch_size (int): Maximum batch size.
        max_seq_len (int): Maximum sequence length.
        dtype (Literal["bf16", "fp8"]): Data type for computations.
        vocab_size (int): Vocabulary size.
        dim (int): Model dimension.
        inter_dim (int): Intermediate dimension for MLP layers.
        moe_inter_dim (int): Intermediate dimension for MoE layers.
        n_layers (int): Number of transformer layers.
        n_dense_layers (int): Number of dense layers in the model.
        n_heads (int): Number of attention heads.
        n_routed_experts (int): Number of routed experts for MoE layers.
        n_shared_experts (int): Number of shared experts for MoE layers.
        n_activated_experts (int): Number of activated experts in MoE layers.
        n_expert_groups (int): Number of expert groups.
        n_limited_groups (int): Number of limited groups for MoE routing.
        score_func (Literal["softmax", "sigmoid"]): Scoring function for MoE routing.
        route_scale (float): Scaling factor for routing scores.
        q_lora_rank (int): LoRA rank for query projections.
        kv_lora_rank (int): LoRA rank for key-value projections.
        qk_nope_head_dim (int): Dimension for query-key projections without positional embeddings.
        qk_rope_head_dim (int): Dimension for query-key projections with rotary embeddings.
        v_head_dim (int): Dimension for value projections.
        original_seq_len (int): Original sequence length.
        rope_theta (float): Base for rotary positional encoding.
        rope_factor (float): Scaling factor for extended sequence lengths.
        beta_fast (int): Fast beta correction factor.
        beta_slow (int): Slow beta correction factor.
        mscale (float): Scaling factor for extended attention.
    """

    max_batch_size: int = 768
    max_seq_len: int = 4096 * 4

    dtype: Literal["bf16", "fp8"] = "fp8"
    vocab_size: int = 102400
    dim: int = 2048
    inter_dim: int = 10944
    moe_inter_dim: int = 1408
    n_layers: int = 27
    n_dense_layers: int = 1
    n_heads: int = 16
    # moe
    n_routed_experts: int = 64
    n_shared_experts: int = 2
    n_activated_experts: int = 6
    n_expert_groups: int = 1
    n_limited_groups: int = 1
    score_func: Literal["softmax", "sigmoid"] = "softmax"
    route_scale: float = 1.0
    # mla
    q_lora_rank: int = 0
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    # yarn
    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.0


class ParallelEmbedding(nn.Module):
    """
    Embedding layer with parallelism support across distributed processes.

    Args:
        vocab_size (int): Vocabulary size.
        dim (int): Embedding dimension.
    """

    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        world_size = torch.distributed.get_world_size()
        assert (
            vocab_size % world_size == 0
        ), f"Vocabulary size must be divisible by world size (world_size={world_size})"
        self.part_vocab_size = vocab_size // world_size
        self.vocab_start_idx = torch.distributed.get_rank() * self.part_vocab_size
        self.vocab_end_idx = self.vocab_start_idx + self.part_vocab_size
        self.weight = nn.Parameter(
            torch.empty(self.part_vocab_size, self.dim), requires_grad=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for parallel embedding layer.

        Args:
            x (torch.Tensor): Input tensor containing token indices.

        Returns:
            torch.Tensor: Embedded representations.

        Raises:
            ValueError: If `world_size` is not defined.
        """
        mask = (x < self.vocab_start_idx) | (x >= self.vocab_end_idx)

        x = x - self.vocab_start_idx

        x[mask] = 0
        y = F.embedding(x, self.weight)
        y[mask] = 0

        y = distnn.all_reduce(y)
        return y


def linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    column_parallel=False,
) -> torch.Tensor:
    """
    Applies a linear transformation to the incoming data: y = xA^T + b.
    This function supports specialized implementations based on quantization
    and tensor formats.

    Args:
        x (torch.Tensor): The input tensor.
        weight (torch.Tensor): The weight tensor. It may be quantized and
            requires dequantization for certain cases.
        bias (Optional[torch.Tensor]): The bias tensor to be added. Default is None.

    Returns:
        torch.Tensor: The result of the linear transformation, which may involve
        quantization-aware computations depending on the input parameters.

    Notes:
        - If `weight` is quantized (e.g., `element_size() == 1`), a dequantized version
          is used for computation.
        - If `gemm_impl == "bf16"`, dequantization and a `bf16` GEMM operation are applied.
        - For other cases, the function applies quantization to `x` and uses `fp8_gemm` for computation.
    """
    if weight.element_size() > 1:
        return F.linear(x, weight, bias)
    elif gemm_impl == "bf16":
        weight = custom_weight_dequant(weight, weight.scale)
        return F.linear(x, weight, bias)
    else:
        x, scale = custom_act_quant(x, block_size)
        y = custom_gemm(x, scale, weight, weight.scale, column_parallel)
        if bias is not None:
            y = y + bias
        return y


class Linear(nn.Module):
    """
    Custom linear layer with support for quantized weights and optional bias.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
    """

    dtype = torch.float8_e4m3fn

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        dtype=None,
        lora=False,
        lora_config: LoraConfig = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=dtype or Linear.dtype),
            requires_grad=False,
        )

        self.lora_enabled = lora
        if self.lora_enabled:
            self.rank = lora_config.lora_rank
            self.alpha = lora_config.lora_alpha
            self.lora_finetune_scale = lora_config.lora_alpha / lora_config.lora_rank
            self.dropout = nn.Dropout(p=lora_config.lora_dropout)
            self.lora_down = nn.Parameter(
                torch.empty(lora_config.lora_rank, in_features, dtype=torch.bfloat16),
                requires_grad=self.training,
            )
            self.lora_up = nn.Parameter(
                torch.empty(out_features, lora_config.lora_rank, dtype=torch.bfloat16),
                requires_grad=self.training,
            )

            nn.init.kaiming_uniform_(self.lora_down, a=math.sqrt(5))
            nn.init.zeros_(self.lora_up)

        if self.weight.element_size() == 1:
            scale_out_features = (out_features + block_size - 1) // block_size
            scale_in_features = (in_features + block_size - 1) // block_size
            self.weight.scale = self.scale = nn.Parameter(
                torch.empty(scale_out_features, scale_in_features, dtype=torch.float32),
                requires_grad=False,
            )
        else:
            self.register_parameter("scale", None)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features), requires_grad=False)
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor, use_lora=True) -> torch.Tensor:
        """
        Forward pass for the custom linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor after linear computation.
        """

        original_path = linear(x, self.weight)
        if not self.lora_enabled or not use_lora:
            if self.bias is not None:
                original_path = original_path + self.bias
            return original_path

        lora_path = self.dropout(F.linear(F.linear(x, self.lora_down), self.lora_up))
        total_lora = original_path + self.lora_finetune_scale * lora_path
        if self.bias is not None:
            total_lora = total_lora + self.bias

        return total_lora


class ColumnParallelLinear(Linear):
    """
    Linear layer with column parallelism, splitting output features across distributed processes.

    Args:
        in_features (int): Number of input features.
        out_features (int): Total number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        dtype=None,
        lora=False,
        lora_config: LoraConfig = None,
    ):
        world_size = torch.distributed.get_world_size()
        assert (
            out_features % world_size == 0
        ), f"Output features must be divisible by world size (world_size={world_size})"
        self.part_out_features = out_features // world_size

        super().__init__(
            in_features,
            self.part_out_features,
            bias,
            dtype,
            lora=lora,
            lora_config=lora_config,
        )

    def forward(self, x: torch.Tensor, use_lora=True) -> torch.Tensor:
        """
        Forward pass for column parallel linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor with column-parallel computation.
        """
        if x.requires_grad:
            x.register_hook(self._all_reduce_grad)

        y = linear(x, self.weight)
        if not self.lora_enabled or not use_lora:
            if self.bias is not None:
                y = y + self.bias
            return y
        lora_path = self.dropout(F.linear(F.linear(x, self.lora_down), self.lora_up))
        total_lora = y + self.lora_finetune_scale * lora_path
        if self.bias is not None:
            total_lora = total_lora + self.bias

        return total_lora

    @staticmethod
    def _all_reduce_grad(grad):
        dist.all_reduce(grad, op=dist.ReduceOp.AVG)
        return grad


class RowParallelLinear(Linear):
    """
    Linear layer with row parallelism, splitting input features across distributed processes.

    Args:
        in_features (int): Total number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        dtype=None,
        lora=False,
        lora_config: LoraConfig = None,
    ):
        world_size = torch.distributed.get_world_size()
        assert (
            in_features % world_size == 0
        ), f"Input features must be divisible by world size (world_size={world_size})"
        self.part_in_features = in_features // world_size
        super().__init__(
            self.part_in_features,
            out_features,
            bias,
            dtype,
            lora=lora,
            lora_config=lora_config,
        )

    def forward(self, x: torch.Tensor, use_lora=True) -> torch.Tensor:
        """
        Forward pass for row parallel linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor with row-parallel computation.
        """

        y = linear(x, self.weight)
        if not self.lora_enabled or not use_lora:
            y = distnn.all_reduce(y)
            if self.bias is not None:
                y = y + self.bias
            return y

        lora_path = self.dropout(F.linear(F.linear(x, self.lora_down), self.lora_up))
        total_lora = y + self.lora_finetune_scale * lora_path

        total_lora = distnn.all_reduce(total_lora)
        if self.bias is not None:
            total_lora = total_lora + self.bias

        return total_lora


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    Args:
        dim (int): Dimension of the input tensor.

        eps (float): Epsilon value for numerical stability. Defaults to 1e-6.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim), requires_grad=False)

    def forward(self, x: torch.Tensor):
        """
        Forward pass for RMSNorm.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor with the same shape as input.
        """
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)


def precompute_freqs_cis(args: ModelArgs) -> torch.Tensor:
    """
    Precomputes frequency-based complex exponential values for rotary positional embeddings.

    Args:
        args (ModelArgs): Model arguments containing positional embedding parameters.

    Returns:
        torch.Tensor: Precomputed complex exponential values for positional embeddings.
    """
    dim = args.qk_rope_head_dim
    seqlen = args.max_seq_len
    beta_fast = args.beta_fast
    beta_slow = args.beta_slow
    base = args.rope_theta
    factor = args.rope_factor

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        """
        Computes the correction dimension for a given number of rotations in the rotary positional embedding.

        Args:
            num_rotations (float): Number of rotations to compute the correction for.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            float: The correction dimension based on the input parameters.
        """
        return (
            dim
            * math.log(max_seq_len / (num_rotations * 2 * math.pi))
            / (2 * math.log(base))
        )

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        """
        Computes the range of correction dimensions for rotary positional embeddings.

        Args:
            low_rot (float): Lower bound for the number of rotations.
            high_rot (float): Upper bound for the number of rotations.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            Tuple[int, int]: The range of correction dimensions (low, high), clamped to valid indices.
        """
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min, max, dim):
        """
        Computes a linear ramp function used to smooth values between a minimum and maximum range.

        Args:
            min (float): Minimum value for the ramp function.
            max (float): Maximum value for the ramp function.
            dim (int): Dimensionality of the ramp tensor.

        Returns:
            torch.Tensor: A tensor of shape (dim,) with values linearly interpolated between 0 and 1,
                clamped to the range [0, 1].
        """
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if seqlen > args.original_seq_len:
        low, high = find_correction_range(
            beta_fast, beta_slow, dim, base, args.original_seq_len
        )
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Applies rotary positional embeddings to the input tensor.

    Args:
        x (torch.Tensor): Input tensor with positional embeddings to be applied.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for positional embeddings.

    Returns:
        torch.Tensor: Tensor with rotary embeddings applied.
    """
    dtype = x.dtype
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(-1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    return y.to(dtype)


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) used as a feed-forward layer.

    Attributes:
        w1 (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Linear layer for hidden-to-output transformation.
        w3 (nn.Module): Additional linear layer for feature transformation.
    """

    def __init__(self, dim: int, inter_dim: int, lora_config: LoraConfig):
        """
        Initializes the MLP layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        self.w1 = ColumnParallelLinear(
            dim,
            inter_dim,
            lora=lora_config.get_lora_key("up_proj"),
            lora_config=lora_config,
        )
        self.w2 = RowParallelLinear(
            inter_dim,
            dim,
            lora=lora_config.get_lora_key("down_proj"),
            lora_config=lora_config,
        )
        self.w3 = ColumnParallelLinear(
            dim,
            inter_dim,
            lora=lora_config.get_lora_key("gate_proj"),
            lora_config=lora_config,
        )

    def forward(
        self,
        x: torch.Tensor,
        use_lora=True,
        attn_mask=None,
    ) -> torch.Tensor:
        """
        Forward pass for the MLP layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after MLP computation.
        """
        return self.w2(
            F.silu(self.w1(x, use_lora=use_lora)) * self.w3(x, use_lora=use_lora),
            use_lora=use_lora,
        )


class Expert(nn.Module):
    """
    Expert layer for Mixture-of-Experts (MoE) models.

    Attributes:
        w1 (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Linear layer for hidden-to-output transformation.
        w3 (nn.Module): Additional linear layer for feature transformation.
    """

    def __init__(self, dim: int, inter_dim: int, lora_config: LoraConfig):
        """
        Initializes the Expert layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        self.w1 = Linear(
            dim,
            inter_dim,
            lora=lora_config.get_lora_key("up_proj"),
            lora_config=lora_config,
        )
        self.w2 = Linear(
            inter_dim,
            dim,
            lora=lora_config.get_lora_key("down_proj"),
            lora_config=lora_config,
        )
        self.w3 = Linear(
            dim,
            inter_dim,
            lora=lora_config.get_lora_key("gate_proj"),
            lora_config=lora_config,
        )

    def forward(
        self,
        x: torch.Tensor,
        use_lora=True,
    ) -> torch.Tensor:
        """
        Forward pass for the Expert layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert computation.
        """
        return self.w2(
            F.silu(self.w1(x, use_lora=use_lora)) * self.w3(x, use_lora=use_lora),
            use_lora=use_lora,
        )


class Gate(nn.Module):
    """
    Gating mechanism for routing inputs in a mixture-of-experts (MoE) model.

    Attributes:
        dim (int): Dimensionality of input features.
        topk (int): Number of top experts activated for each input.
        n_groups (int): Number of groups for routing.
        topk_groups (int): Number of groups to route inputs to.
        score_func (str): Scoring function ('softmax' or 'sigmoid').
        route_scale (float): Scaling factor for routing weights.
        weight (torch.nn.Parameter): Learnable weights for the gate.
        bias (Optional[torch.nn.Parameter]): Optional bias term for the gate.
    """

    def __init__(self, args: ModelArgs, lora=False, lora_config: LoraConfig = None):
        """
        Initializes the Gate module.

        Args:
            args (ModelArgs): Model arguments containing gating parameters.
        """
        super().__init__()
        self.dim = args.dim
        self.topk = args.n_activated_experts
        self.n_groups = args.n_expert_groups
        self.topk_groups = args.n_limited_groups
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.weight = nn.Parameter(
            torch.empty(args.n_routed_experts, args.dim), requires_grad=False
        )

        self.lora_enabled = lora

        if lora:
            self.lora_down = nn.Parameter(
                torch.empty(lora_config.lora_rank, args.dim), requires_grad=True
            )
            nn.init.kaiming_uniform_(self.lora_down, a=math.sqrt(5))
            self.lora_up = nn.Parameter(
                torch.zeros(args.n_routed_experts, lora_config.lora_rank),
                requires_grad=True,
            )

            self.lora_finetune_scale = lora_config.lora_alpha / lora_config.lora_rank

        self.bias = (
            nn.Parameter(torch.empty(args.n_routed_experts), requires_grad=False)
            if self.dim == 7168
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        use_lora=True,
        attn_mask=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the gating mechanism.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Routing weights and selected expert indices.
        """
        scores = linear(x, self.weight)
        if self.lora_enabled and use_lora:
            lora_path = F.linear(x, self.lora_down)
            lora_path = F.linear(lora_path, self.lora_up)
            scores = scores + self.lora_finetune_scale * lora_path
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        else:
            scores = scores.sigmoid()
        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias
        if self.n_groups > 1:
            scores = scores.view(x.size(0), self.n_groups, -1)
            if self.bias is None:
                group_scores = scores.amax(dim=-1)
            else:
                group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            mask = scores.new_ones(x.size(0), self.n_groups, dtype=bool).scatter_(
                1, indices, False
            )
            scores = scores.masked_fill(mask.unsqueeze(-1), float("-inf")).flatten(1)
        indices = torch.topk(scores, self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func == "sigmoid":
            weights = weights / (weights.sum(dim=-1, keepdim=True))
        weights *= self.route_scale
        return weights.type_as(x), indices


class MoE(nn.Module):
    """
    Mixture-of-Experts (MoE) module.

    Attributes:
        dim (int): Dimensionality of input features.
        n_routed_experts (int): Total number of experts in the model.
        n_local_experts (int): Number of experts handled locally in distributed systems.
        n_activated_experts (int): Number of experts activated for each input.
        gate (nn.Module): Gating mechanism to route inputs to experts.
        experts (nn.ModuleList): List of expert modules.
        shared_experts (nn.Module): Shared experts applied to all inputs.
    """

    def __init__(
        self,
        args: ModelArgs,
        lora_config: LoraConfig = None,
    ):
        """
        Initializes the MoE module.

        Args:
            args (ModelArgs): Model arguments containing MoE parameters.
        """
        super().__init__()
        self.dim = args.dim
        world_size = torch.distributed.get_world_size()
        assert (
            args.n_routed_experts % world_size == 0
        ), f"Number of experts must be divisible by world size (world_size={world_size})"
        self.n_routed_experts = args.n_routed_experts
        self.n_local_experts = args.n_routed_experts // world_size
        self.n_activated_experts = args.n_activated_experts
        self.experts_start_idx = torch.distributed.get_rank() * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        self.gate = Gate(
            args, lora=lora_config.get_lora_key("moe_gate"), lora_config=lora_config
        )
        self.experts = nn.ModuleList(
            [
                (
                    Expert(args.dim, args.moe_inter_dim, lora_config=lora_config)
                    if self.experts_start_idx <= i < self.experts_end_idx
                    else None
                )
                for i in range(self.n_routed_experts)
            ]
        )
        self.shared_experts = MLP(
            args.dim,
            args.n_shared_experts * args.moe_inter_dim,
            lora_config=lora_config,
        )

    def forward(
        self,
        x: torch.Tensor,
        use_lora=True,
        attn_mask=None,
    ) -> torch.Tensor:
        """
        Forward pass for the MoE module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert routing and computation.
        """
        shape = x.size()
        S = shape[1]
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x, use_lora=use_lora, attn_mask=attn_mask)
        y = torch.zeros_like(x)

        for i in range(self.experts_start_idx, self.experts_end_idx):
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            expert_output = torch.zeros_like(x)
            expert_output[idx] = (
                expert(
                    x[idx],
                    use_lora=use_lora,
                )
                * weights[idx, top, None]
            )
            y = y + expert_output

        z = self.shared_experts(
            x,
            use_lora=use_lora,
        )

        y = distnn.all_reduce(y)
        return (y + z).view(shape)


class MLA(nn.Module):
    """
    Multi-Headed Attention Layer (MLA).

    Attributes:
        dim (int): Dimensionality of the input features.
        n_heads (int): Number of attention heads.
        n_local_heads (int): Number of local attention heads for distributed systems.
        q_lora_rank (int): Rank for low-rank query projection.
        kv_lora_rank (int): Rank for low-rank key/value projection.
        qk_nope_head_dim (int): Dimensionality of non-positional query/key projections.
        qk_rope_head_dim (int): Dimensionality of rotary-positional query/key projections.
        qk_head_dim (int): Total dimensionality of query/key projections.
        v_head_dim (int): Dimensionality of value projections.
        softmax_scale (float): Scaling factor for softmax in attention computation.
    """

    def __init__(self, args: ModelArgs, lora_config: LoraConfig):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads // torch.distributed.get_world_size()
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim

        self.wq_a = Linear(
            self.dim,
            self.q_lora_rank,
            lora=lora_config.get_lora_key("q_proj"),
            lora_config=lora_config,
        )
        self.q_norm = RMSNorm(self.q_lora_rank)
        self.wq_b = ColumnParallelLinear(
            self.q_lora_rank,
            self.n_heads * self.qk_head_dim,
            lora=lora_config.get_lora_key("q_proj"),
            lora_config=lora_config,
        )
        self.wkv_a = Linear(
            self.dim,
            self.kv_lora_rank + self.qk_rope_head_dim,
            lora=lora_config.get_lora_key("kv_proj"),
            lora_config=lora_config,
        )
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        self.wkv_b = ColumnParallelLinear(
            self.kv_lora_rank,
            self.n_heads * (self.qk_nope_head_dim + self.v_head_dim),
            lora=lora_config.get_lora_key("kv_proj"),
            lora_config=lora_config,
        )

        self.wo = RowParallelLinear(
            self.n_heads * self.v_head_dim,
            self.dim,
            lora=lora_config.get_lora_key("o_proj"),
            lora_config=lora_config,
        )
        self.softmax_scale = self.qk_head_dim**-0.5

        causal_mask = torch.triu(
            torch.ones(
                lora_config.max_train_seq_len,
                lora_config.max_train_seq_len,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        self.register_buffer("causal_mask", causal_mask)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        use_lora=True,
        start_pos=0,
    ):
        """
        Forward pass for the Multi-Headed Attention Layer (MLA).

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
            mask (Optional[torch.Tensor]): Mask tensor to exclude certain positions from attention.

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        # low-rank projection of the query
        q = self.wq_b(self.q_norm(self.wq_a(x, use_lora=use_lora)), use_lora=use_lora)

        q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
        # this splits the query into a part where the PE gets applied and a part where it doesn't
        q_nope_original, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        # apply the rotary embedding to the PE part of the query
        q_pe = apply_rotary_emb(q_pe, freqs_cis)

        # apply the low-rank projection to the key and value
        kv = self.wkv_a(x, use_lora=use_lora)

        # split the key and value into a part where the PE gets applied and a part where it doesn't
        # only the key gets split
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        # apply the rotary embedding to the PE part of the key
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)

        wkv_b = (
            self.wkv_b.weight
            if self.wkv_b.scale is None
            else custom_weight_dequant(self.wkv_b.weight, self.wkv_b.scale, block_size)
        )

        if self.wkv_b.lora_enabled and use_lora:
            wkv_b = (
                wkv_b
                + self.wkv_b.lora_finetune_scale
                * self.wkv_b.lora_up
                @ self.wkv_b.lora_down
            )

        wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)

        wkv_b_key = wkv_b[:, : self.qk_nope_head_dim]  # heads, k_head_dim, kv_rank
        wkv_b_value = wkv_b[:, -self.v_head_dim :]  # heads, val_head_dim, kv_rank

        q_nope = torch.einsum(
            "bshd,hdc->bshc", q_nope_original, wkv_b_key
        )  # low-rank query

        if self.wkv_b.lora_enabled and use_lora:
            wkv_b_lora_up = self.wkv_b.lora_up.view(
                self.n_local_heads, -1, self.wkv_b.rank
            )  # heads, kv_head_dim, rank
            wkv_b_lora_down = self.wkv_b.lora_down
            wkv_b_lora_up_key = wkv_b_lora_up[:, : self.qk_nope_head_dim]
            wkv_b_lora_up_value = wkv_b_lora_up[:, -self.v_head_dim :]
            q_nope_lora = torch.einsum(
                "bshd,hdR->bshR", q_nope_original, wkv_b_lora_up_key
            )
            q_nope_lora_lowrank = torch.einsum(
                "bshR,Rc->bshc", q_nope_lora, wkv_b_lora_down
            )
            q_nope = q_nope + self.wkv_b.lora_finetune_scale * q_nope_lora_lowrank

        all_kv = self.kv_norm(kv)
        all_pe = k_pe.squeeze(2)

        scores: torch.Tensor = (
            torch.einsum("bshc,btc->bsht", q_nope, all_kv)
            + torch.einsum("bshr,btr->bsht", q_pe, all_pe)
        ) * self.softmax_scale

        kv_len = all_kv.shape[1]

        q_start, q_end = start_pos, start_pos + seqlen
        mask = self.causal_mask[q_start:q_end, :kv_len]
        scores = scores.masked_fill(mask[None, :, None, :], -float("inf"))

        if attn_mask is not None:
            mask_len = attn_mask.size(1)
            mask = (attn_mask == 0).unsqueeze(1).unsqueeze(2)
            scores[..., :mask_len] = scores[..., :mask_len].masked_fill(
                mask, float("-inf")
            )

        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
        x = torch.einsum("bsht,btc->bshc", scores, all_kv)
        x = torch.einsum("bshc,hdc->bshd", x, wkv_b_value)

        if self.wkv_b.lora_enabled and use_lora:
            x_lora_val = torch.einsum("bshc,hcd->bshc", x, wkv_b_lora_up_value)
            x = x + self.wkv_b.lora_finetune_scale * x_lora_val

        x = self.wo(x.flatten(2), use_lora=use_lora)
        return x


class Block(nn.Module):
    """
    Transformer block combining attention and feed-forward layers.

    Attributes:
        attn (nn.Module): Attention layer (MLA).
        ffn (nn.Module): Feed-forward network (MLP or MoE).
        attn_norm (nn.Module): Layer normalization for attention.
        ffn_norm (nn.Module): Layer normalization for feed-forward network.
    """

    def __init__(self, layer_id: int, args: ModelArgs, lora_config: LoraConfig):
        """
        Initializes the Transformer block.

        Args:
            layer_id (int): Layer index in the transformer.
            args (ModelArgs): Model arguments containing block parameters.
        """
        super().__init__()

        self.attn = MLA(
            args,
            lora_config=lora_config,
        )

        self.ffn = (
            MLP(args.dim, args.inter_dim, lora_config=lora_config)
            if layer_id < args.n_dense_layers
            else MoE(args, lora_config=lora_config)
        )

        self.attn_norm = RMSNorm(args.dim)
        self.ffn_norm = RMSNorm(args.dim)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        dummy: torch.Tensor,
        attn_mask=None,
        start_pos=0,
        use_lora=True,
    ) -> torch.Tensor:
        """
        Forward pass for the Transformer block.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
            mask (Optional[torch.Tensor]): Mask tensor to exclude certain positions from attention.

        Returns:
            torch.Tensor: Output tensor after block computation.
        """

        x = x + self.attn(
            self.attn_norm(x),
            freqs_cis,
            attn_mask=attn_mask,
            use_lora=use_lora,
            start_pos=start_pos,
        )
        x = x + self.ffn(self.ffn_norm(x), use_lora=use_lora, attn_mask=attn_mask)
        return x
