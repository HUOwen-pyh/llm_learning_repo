"""第五阶段最终组合任务：训练并诊断 WaveNet-inspired 层次模型。"""

from __future__ import annotations

from _common import data_path, parse_args, prepare_artifacts


def run(full: bool, seed: int) -> None:
    output = prepare_artifacts()
    path = data_path(full)

    # TODO FINAL-WAVENET：
    # 1. 构建 block_size=8 数据和 flat/hierarchical 模型；
    # 2. 输出 shape CSV 与感受野图；
    # 3. 展示并修复 3D BatchNorm axis bug；
    # 4. 用 batchnorm_factory 重跑 axis0 bug/fixed，并完成
    #    WAVENET_REQUIRED_EXPERIMENTS 中固定的五组实验；
    # 5. 保存 WaveTrainingHistory（含 schedule 与逐参数 update:data）、评估、采样；
    # 6. 在 2D/3D 中间张量上重跑全部内部诊断；
    # 7. validate_wave_artifacts；全部阶段结束后再运行 python check.py --artifacts。
    raise NotImplementedError("组合任务：生成 WaveNet 全部结构、实验与诊断工件")


if __name__ == "__main__":
    args = parse_args("WaveNet-inspired 最终组合任务")
    run(args.full, args.seed)
