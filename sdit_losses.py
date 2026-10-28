from collections.abc import Sequence

import torch
from torch import Tensor
from torch.nn import functional as F


def _validate_inputs(
    logits: Sequence[Tensor], targets: Tensor, alpha: float, temperature: float
) -> Tensor:
    if len(logits) < 2:
        raise ValueError("logits must contain at least one auxiliary head and the final head")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")

    final_logits = logits[-1]
    if final_logits.ndim != 2:
        raise ValueError("each classifier output must have shape (batch_size, num_classes)")
    if targets.ndim != 1 or targets.shape[0] != final_logits.shape[0]:
        raise ValueError("targets must have shape (batch_size,)")
    for head_logits in logits:
        if head_logits.shape != final_logits.shape:
            raise ValueError("all classifier outputs must have the same shape")
    return final_logits


def lossWithoutFeatureExtraction(
    logits: Sequence[Tensor],
    targets: Tensor,
    alpha: float,
    temperature: float = 1.0,
) -> Tensor:
    """CE와 최종 분류기를 기준으로 지식 증류 손실을 합산하는 함수.

    `logits`에는 보조 분류기의 로짓을 먼저, 최종 분류기의 로짓을 마지막에 담는 순서.
    모든 분류기에 교차 엔트로피를 계산하고, 보조 분류기에는 최종 분류기와의 KL 손실을 계산.
    최종 분류기 자체의 KL 손실은 0으로 설정.

    PyTorch `kl_div` 규약에 따른 `KL(최종 분류기 || 보조 분류기)` 계산.
    입력된 temperature를 KL 분포에 적용하고, 표준 지식 증류 방식에 따라 KL 항에 temperature의 제곱을 곱하는 처리.
    각 분류기의 CE 및 KL 항에 배치 평균을 적용한 뒤 합산."""
    final_logits = _validate_inputs(logits, targets, alpha, temperature)
    final_probs = F.softmax(final_logits.detach() / temperature, dim=-1)

    total = final_logits.new_zeros(())
    for head_logits in logits:
        ce = F.cross_entropy(head_logits, targets)
        if head_logits is final_logits:
            kl = ce.new_zeros(())
        else:
            kl = F.kl_div(
                F.log_softmax(head_logits / temperature, dim=-1),
                final_probs,
                reduction="batchmean",
            ) * (temperature**2)
        total = total + (1.0 - alpha) * ce + alpha * kl
    return total


def _feature_matching_loss(feature: Tensor, teacher_feature: Tensor) -> Tensor:
    """얕은 분류기의 특징을 최종 분류기 특징에 맞추기 위한 제곱 L2 손실."""
    return torch.sum((feature - teacher_feature.detach()) ** 2)


def lossWithFeatureExtraction(
    logits: Sequence[Tensor],
    targets: Tensor,
    encoder_features: Sequence[Tensor],
    alpha: float,
    feature_weight: float,
    temperature: float = 1.0,
) -> Tensor:
    """식 (13)과 인코더 특징 간 제곱 L2 거리를 계산하는 함수.

    `encoder_features`는 `logits`와 일대일 대응 및 동일한 순서로 전달.
    마지막 특징 텐서는 최종 분류기에 해당.

    논문에 기술된 특징 간 차이의 제곱 합산.
    최종 분류기 특징과 자기 자신 사이의 거리는 0이므로 손실에 영향 없음."""
    final_logits = _validate_inputs(logits, targets, alpha, temperature)
    if feature_weight < 0.0:
        raise ValueError("feature_weight must be non-negative")
    if len(encoder_features) != len(logits):
        raise ValueError("encoder_features must have one tensor per classifier head")

    final_feature = encoder_features[-1]
    if any(feature.shape != final_feature.shape for feature in encoder_features):
        raise ValueError("all encoder feature tensors must have the same shape")

    final_probs = F.softmax(final_logits.detach() / temperature, dim=-1)
    total = final_logits.new_zeros(())
    for head_logits, feature in zip(logits, encoder_features):
        ce = F.cross_entropy(head_logits, targets)
        if head_logits is final_logits:
            kl = ce.new_zeros(())
            feature_loss = ce.new_zeros(())
        else:
            kl = F.kl_div(
                F.log_softmax(head_logits / temperature, dim=-1),
                final_probs,
                reduction="batchmean",
            ) * (temperature**2)
            feature_loss = _feature_matching_loss(feature, final_feature)
        total = total + (1.0 - alpha) * ce + alpha * kl + feature_weight * feature_loss
    return total
