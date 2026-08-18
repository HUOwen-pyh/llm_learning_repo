"""52 个递进关卡的轻量检查入口。--list 不需要先安装 PyTorch。"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


TITLES = [
    "读取语料与基础统计",
    "词表、编码与解码",
    "生成 bigram 训练对",
    "Bigram 计数矩阵",
    "概率、广播与 smoothing",
    "NLL 与 perplexity",
    "Bigram 自回归采样",
    "Bigram 三张必做图",
    "神经 Bigram",
    "Bigram 训练、trigram 与比较",
    "按单词划分 train/dev/test",
    "上下文窗口数据集",
    "Embedding lookup 与形状",
    "初始化显式 MLP 参数",
    "MLP 前向传播与缓存",
    "稳定交叉熵",
    "Minibatch 与一步训练",
    "Learning-rate range test",
    "训练历史、评估与容量实验",
    "Embedding 与输出分析",
    "均匀预测基线",
    "修复输出层初始化",
    "tanh 饱和率与热图",
    "Kaiming 与方差传播",
    "手写二维 BatchNorm",
    "BatchNorm train/eval 与 running stats",
    "Batch coupling 与 bias 冗余",
    "BatchNorm folding",
    "自定义层、深层 MLP 与 BN 校准",
    "前向 activation statistics",
    "反向 activation-gradient statistics",
    "参数梯度与 update:data",
    "统一 dashboard 与全零初始化",
    "原子化 forward cache 与比较器",
    "原子 CrossEntropy backward",
    "化简 CE backward 与热图",
    "Linear2 与 tanh backward",
    "原子 BatchNorm backward",
    "化简 BatchNorm backward",
    "Linear1 与 embedding scatter-add",
    "完整手写 backward 与梯度审计",
    "无 autograd 训练闭环",
    "8 字符上下文",
    "任意前导维度 Linear 与 Embedding",
    "FlattenConsecutive",
    "Sequential 模式与 shape trace",
    "8 字符扁平 MLP 基线",
    "树状层次模型",
    "感受野结构图",
    "修复 3D BatchNorm 轴问题",
    "WaveNet 训练、评估与采样",
    "最终实验与全套内部诊断",
]


STAGE_ENDS = {
    "bigram": 10,
    "mlp": 20,
    "diagnostics": 33,
    "backprop": 42,
    "wavenet": 52,
    "all": 52,
}


def module_for(index: int) -> str:
    if index <= 10:
        return "checks.stage1_bigram"
    if index <= 20:
        return "checks.stage2_mlp"
    if index <= 33:
        return "checks.stage3_diagnostics"
    if index <= 42:
        return "checks.stage4_backprop"
    return "checks.stage5_wavenet"


def parse_target(raw: str) -> int:
    lowered = raw.lower()
    if lowered in STAGE_ENDS:
        return STAGE_ENDS[lowered]
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("目标必须是 1-52 或阶段名") from exc
    if not 1 <= value <= len(TITLES):
        raise argparse.ArgumentTypeError("任务编号必须是 1-52")
    return value


def ensure_dependencies() -> None:
    try:
        import torch  # noqa: F401
        import matplotlib  # noqa: F401
        import numpy  # noqa: F401
    except ModuleNotFoundError as exc:
        print(f"[缺少依赖] {exc.name}")
        print("请先运行：python -m pip install -r requirements.txt")
        raise SystemExit(2) from exc


def run_one(index: int) -> None:
    module = importlib.import_module(module_for(index))
    check = getattr(module, f"check_{index:02d}")
    title = TITLES[index - 1]
    try:
        check()
    except NotImplementedError as exc:
        print(f"[待完成] {index:02d}. {title}: {exc}")
        raise SystemExit(1) from exc
    except Exception as exc:
        print(f"[未通过] {index:02d}. {title}: {exc}")
        raise SystemExit(1) from exc
    else:
        print(f"[通过]   {index:02d}. {title}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Makemore 逐关检查")
    parser.add_argument("target", nargs="?", default="all", help="1-52、阶段名或 all")
    parser.add_argument("--only", type=int, help="只运行指定任务，不检查前置任务")
    parser.add_argument("--list", action="store_true", help="列出全部任务")
    parser.add_argument(
        "--artifacts",
        action="store_true",
        help="最终组合验收：检查 artifacts/ 中全部必做 PNG/JSON/CSV/TXT",
    )
    args = parser.parse_args()

    if args.list:
        for index, title in enumerate(TITLES, start=1):
            print(f"{index:02d}. {title}")
        return

    if args.artifacts:
        from makemore.reporting import REQUIRED_ARTIFACTS, validate_artifacts

        artifact_dir = Path(__file__).resolve().parent / "artifacts"
        try:
            invalid = validate_artifacts(artifact_dir, REQUIRED_ARTIFACTS)
        except NotImplementedError as exc:
            print(f"[待完成] 最终工件验收: {exc}")
            raise SystemExit(1) from exc
        if invalid:
            print("[未通过] 以下工件缺失、为空或格式无效：")
            for name in invalid:
                print(f"  - {name}")
            raise SystemExit(1)
        print(f"[通过]   全部 {len(REQUIRED_ARTIFACTS)} 个最终工件")
        return

    ensure_dependencies()
    if args.only is not None:
        if not 1 <= args.only <= len(TITLES):
            parser.error("--only 必须是 1-52")
        run_one(args.only)
        return

    try:
        target = parse_target(args.target)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    for index in range(1, target + 1):
        run_one(index)
    print(f"\n已通过前 {target} 个任务。")


if __name__ == "__main__":
    main()
