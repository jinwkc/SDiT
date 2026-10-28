"""Self-distilling Vision Transformer with a configurable encoder depth.

The default configuration matches the encoder count and base dimensions of
the ViT-Base model: 12 encoder blocks, 768 dimensions, and
12 attention heads. Auxiliary classifiers are attached after every encoder
except the final one.

``SDiT_Small`` (384 dimensions, 6 heads) and ``SDiT_Tiny`` (192 dimensions,
3 heads) keep the same 12-block depth and accept any ``SDiT`` keyword argument
as an override, e.g. ``SDiT_Tiny(img_size=32, patch_size=4, n_classes=10)``.
"""

import torch
from torch import Tensor, nn


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: int = 384,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError("img_size must be divisible by patch_size")
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int = 12,
        qkv_bias: bool = True,
        attn_p: float = 0.0,
        proj_p: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError("dim must be divisible by n_heads")
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, n_tokens, dim = x.shape
        qkv = self.qkv(x).reshape(
            batch_size, n_tokens, 3, self.n_heads, self.head_dim
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attention = (q @ k.transpose(-2, -1)) * self.scale
        attention = self.attn_drop(attention.softmax(dim=-1))
        x = (attention @ v).transpose(1, 2).reshape(batch_size, n_tokens, dim)
        return self.proj_drop(self.proj(x))


class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        p: float = 0.0,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)

    def forward(self, x: Tensor) -> Tensor:
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


class Encoder(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        p: float = 0.0,
        attn_p: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim,
            n_heads=n_heads,
            qkv_bias=qkv_bias,
            attn_p=attn_p,
            proj_p=p,
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dim, p=p)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class AuxMLP(nn.Module):
    """Auxiliary classifier based on the original torchSDiT.py head."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        n_classes: int,
        p: float = 0.0,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)
        self.norm = nn.LayerNorm(out_features, eps=1e-6)
        self.head = nn.Linear(out_features, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        x = self.norm(x)
        return self.head(x[:, 0])


class SDiT(nn.Module):
    """
    Vision Transformer with one shallow auxiliary head per early block.
    ``depth`` defaults to 12, matching ViT-Base. The returned tuple preserves
    ``torchSDiT.py``'s convention: ``(final_logits, aux_1, ..., aux_n)``.
    Set ``return_features=True`` to additionally receive encoder features in
    the same final-first ordering.
    """

    def __init__(self,
                img_size: int = 384,
                patch_size: int = 16,
                in_chans: int = 3,
                n_classes: int = 1000,
                embed_dim: int = 768,
                depth: int = 12,
                n_heads: int = 12,
                mlp_ratio: float = 4.0,
                qkv_bias: bool = True,
                p: float = 0.0,
                attn_p: float = 0.0,
                aux_hidden_features: int = 64,
                aux_out_features: int = 32,
                ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1")

        self.depth = depth
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim)
        )
        self.pos_drop = nn.Dropout(p=p)

        self.blocks = nn.ModuleList(
            [
                Encoder(
                    dim=embed_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    p=p,
                    attn_p=attn_p,
                )
                for _ in range(depth)
            ]
        )
        self.aux_heads = nn.ModuleList(
            [
                AuxMLP(
                    in_features=embed_dim,
                    hidden_features=aux_hidden_features,
                    out_features=aux_out_features,
                    n_classes=n_classes,
                    p=p,
                )
                for _ in range(depth - 1)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x: Tensor, return_features: bool = False):
        batch_size = x.shape[0]
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        if x.shape[1:] != self.pos_embed.shape[1:]:
            raise ValueError(
                "input image size does not match the configured img_size/patch_size"
            )
        x = self.pos_drop(x + self.pos_embed)

        auxiliary_logits = []
        encoder_features = []
        for block, aux_head in zip(self.blocks[:-1], self.aux_heads):
            x = block(x)
            encoder_features.append(x)
            auxiliary_logits.append(aux_head(x))

        x = self.blocks[-1](x)
        final_feature = self.norm(x)
        final_logits = self.head(final_feature[:, 0])

        outputs = (final_logits, *auxiliary_logits)
        if return_features:
            features = (final_feature, *encoder_features)
            return outputs, features
        return outputs


SDiT_Base = SDiT


def SDiT_Small(**kwargs) -> SDiT:
    """ViT-Small sized SDiT: 12 blocks, 384 dimensions, 6 heads."""
    config = {"embed_dim": 384, "depth": 12, "n_heads": 6}
    config.update(kwargs)
    return SDiT(**config)


def SDiT_Tiny(**kwargs) -> SDiT:
    """ViT-Tiny sized SDiT: 12 blocks, 192 dimensions, 3 heads."""
    config = {"embed_dim": 192, "depth": 12, "n_heads": 3}
    config.update(kwargs)
    return SDiT(**config)


__all__ = [
    "Attention",
    "AuxMLP",
    "Encoder",
    "MLP",
    "PatchEmbed",
    "SDiT",
    "SDiT_Base",
    "SDiT_Small",
    "SDiT_Tiny",
]

# ┌──────────────────┬───────────┬───────┬─────────┬──────────────────────────┐
# │       모델       │ embed_dim │ depth │ n_heads │ 파라미터 (CIFAR-10 설정) │
# ├──────────────────┼───────────┼───────┼─────────┼──────────────────────────┤
# │ SDiT_Base (기존) │ 768       │ 12    │ 12      │ 85.72M                   │
# ├──────────────────┼───────────┼───────┼─────────┼──────────────────────────┤
# │ SDiT_Small       │ 384       │ 12    │ 6       │ 21.64M                   │
# ├──────────────────┼───────────┼───────┼─────────┼──────────────────────────┤
# │ SDiT_Tiny        │ 192       │ 12    │ 3       │ 5.53M                    │
# └──────────────────┴───────────┴───────┴─────────┴──────────────────────────┘
