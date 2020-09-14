import torch
import torchvision

from models import utils


class PointerTransformer(torch.nn.Module):
    def __init__(self):
        super(PointerTransformer, self).__init__()

    def forward(self, inputs, lengths=None):
        batch_size, max_len, _ = inputs.size()
        padding_mask = None if lengths is None else utils.batch_sequence_mask(lengths, max_len)


class ScaledDotProductAttention(torch.nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = torch.nn.Dropout(attention_dropout)
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        """前向传播.

        Args:
        	q: Queries张量，形状为[B, L_q, D_q]
        	k: Keys张量，形状为[B, L_k, D_k]
        	v: Values张量，形状为[B, L_v, D_v]，一般来说就是k
        	scale: 缩放因子，一个浮点标量
        	attn_mask: Masking张量，形状为[B, L_q, L_k]

        Returns:
        	上下文张量和attetention张量
        """
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
        	attention = attention * scale
        if attn_mask:
        	# 给需要mask的地方设置一个负无穷
        	attention = attention.masked_fill_(attn_mask, -np.inf)
		# 计算softmax
        attention = self.softmax(attention)
		# 添加dropout
        attention = self.dropout(attention)
		# 和V做点积
        context = torch.bmm(attention, v)
        return context, attention

