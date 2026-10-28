from __future__ import annotations

import platform
import sys


def check_pytorch_mps() -> bool:
    try:
        import torch
    except Exception as exc:
        print(f"[PyTorch] import 실패: {type(exc).__name__}: {exc}")
        return False

    print(f"[PyTorch] 버전: {torch.__version__}")
    print(f"[PyTorch] MPS 빌드 포함: {torch.backends.mps.is_built()}")
    if not torch.backends.mps.is_available():
        print("[PyTorch] MPS를 사용할 수 없습니다.")
        return False

    try:
        values = torch.tensor([1.0, 2.0, 3.0], device="mps")
        result = (values * 2).sum()
        torch.mps.synchronize()
        print(f"[PyTorch] MPS 연산 성공: {result.item()}")
        return True
    except Exception as exc:
        print(f"[PyTorch] MPS 연산 실패: {type(exc).__name__}: {exc}")
        return False


def check_tensorflow_metal() -> bool:
    try:
        import tensorflow as tf
    except Exception as exc:
        print(f"[TensorFlow] import 실패: {type(exc).__name__}: {exc}")
        return False

    print(f"[TensorFlow] 버전: {tf.__version__}")
    try:
        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            print("[TensorFlow] GPU 장치가 없습니다 (Metal 미사용).")
            return False

        # CPU 대체 실행을 막아, GPU 장치에서 실제 연산되는지 확인합니다.
        tf.config.set_soft_device_placement(False)
        with tf.device("/GPU:0"):
            values = tf.constant([1.0, 2.0, 3.0])
            result = tf.reduce_sum(values * 2)
        print(f"[TensorFlow] GPU 장치: {gpus[0].name}")
        print(f"[TensorFlow] GPU 연산 성공: {result.numpy()}")
        return True
    except Exception as exc:
        print(f"[TensorFlow] Metal/GPU 연산 실패: {type(exc).__name__}: {exc}")
        return False


def main() -> int:
    print(f"Python: {sys.version.split()[0]} ({platform.machine()})")
    torch_ok = check_pytorch_mps()
    tensorflow_ok = check_tensorflow_metal()
    print("\n결과:")
    print(f"  PyTorch MPS: {'사용 가능' if torch_ok else '사용 불가'}")
    print(f"  TensorFlow Metal: {'사용 가능' if tensorflow_ok else '사용 불가'}")
    return 0 if torch_ok and tensorflow_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
