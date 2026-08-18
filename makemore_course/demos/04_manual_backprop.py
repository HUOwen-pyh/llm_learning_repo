"""第四阶段最终组合任务：完全不调用 autograd 训练 MLP。"""

from __future__ import annotations

from _common import data_path, parse_args, prepare_artifacts


def run(full: bool, seed: int) -> None:
    output = prepare_artifacts()
    path = data_path(full)

    # TODO FINAL-BACKPROP：
    # 1. 原子 CE 与 BN backward 审计；
    # 2. fused CE/BN 对照；
    # 3. 完整 manual_backward 与 autograd oracle 的一次 parity；
    # 4. 正式训练路径不得调用 loss.backward()/autograd.grad；
    # 5. calibrate_manual_batchnorm 后，用 manual_inference_logits 做 batch=1 采样；
    # 6. 生成 manual_dlogits/manual_grad_diff/manual_training/manual_samples。
    raise NotImplementedError("组合任务：完成无 autograd 训练与梯度审计")


if __name__ == "__main__":
    args = parse_args("手写张量反向传播最终组合任务")
    run(args.full, args.seed)
