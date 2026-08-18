"""任务 18：在课程中的四条数据上训练完整 MLP。"""

from __future__ import annotations

import random
import sys

from micrograd import MLP, fit, forward_loss


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


XS = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
YS = [1.0, -1.0, -1.0, 1.0]


def main() -> None:
    random.seed(1337)
    model = MLP(3, [4, 4, 1])
    losses = fit(model, XS, YS, steps=120, learning_rate=0.05)
    predictions, final_loss = forward_loss(model, XS, YS)

    print(f"参数数量: {len(model.parameters())}")
    print(f"初始 loss: {losses[0]:.6f}")
    print(f"最终 loss: {final_loss.data:.6f}")
    print("预测值:", [round(value.data, 4) for value in predictions])
    print("目标值:", YS)


if __name__ == "__main__":
    main()
