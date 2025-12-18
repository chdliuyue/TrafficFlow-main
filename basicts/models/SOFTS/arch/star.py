import torch
import torch.nn.functional as F
from basicts.modules.activations import ACT2FN
from basicts.modules.mlps import MLPLayer
from torch import nn


class STAR(nn.Module):
    """
    STar Aggregate-Redistribute Module.
    """
    def __init__(self, hidden_size: int, core_size: int, hidden_act: str = "gelu"):
        super().__init__()
        self.ffn1 = MLPLayer(hidden_size, hidden_size, core_size, hidden_act)
        self.ffn2 = MLPLayer(hidden_size + core_size, hidden_size, hidden_size, hidden_act)
        self.act = ACT2FN[hidden_act]

    def forward(
        self,
        hidden_states: torch.Tensor = None,
        attention_mask=None,
        key_padding_mask=None,
        output_attentions: bool = False,
        **kwargs,
    ):
        """
        Compatible with BasicTS EncoderLayer.self_attn_forward, which calls:
            self.self_attn(hidden_states=..., attention_mask=..., key_padding_mask=..., output_attentions=...)
        """
        # 兼容：也允许外部用 STAR(x) 或 STAR(inputs=...) 或 STAR(x=...)
        inputs = hidden_states
        if inputs is None:
            inputs = kwargs.get("inputs", None)
        if inputs is None:
            inputs = kwargs.get("x", None)
        if inputs is None:
            raise TypeError("STAR.forward expected `hidden_states` (or `inputs`/`x`)")

        batch_size, num_features, _ = inputs.shape

        # FFN
        combined_mean = self.ffn1(inputs)  # [B, num_features, core_size]

        # stochastic pooling
        if self.training:
            ratio = F.softmax(combined_mean, dim=1)
            ratio = ratio.transpose(1, 2).reshape(-1, num_features)  # [B*core_size, num_features]
            indices = torch.multinomial(ratio, 1)
            indices = indices.view(batch_size, -1, 1).transpose(1, 2)
            combined_mean = torch.gather(combined_mean, 1, indices)
            combined_mean = combined_mean.repeat(1, num_features, 1)
        else:
            weight = F.softmax(combined_mean, dim=1)
            combined_mean = torch.sum(combined_mean * weight, dim=1, keepdim=True).repeat(1, num_features, 1)

        # FFN fusion
        combined_mean_cat = torch.cat([inputs, combined_mean], -1)  # [B, num_features, hidden+core]
        output = self.ffn2(combined_mean_cat)

        # ✅返回 3 个值，满足 EncoderLayer 的解包
        attn_weights = None
        return output, attn_weights, None

