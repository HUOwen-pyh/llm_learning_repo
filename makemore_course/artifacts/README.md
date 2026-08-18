# 必做工件

完整清单见根目录 `TASKS.md` 的“最终工件清单”。

约定：

- 图像使用 PNG；
- 结构表使用 UTF-8 CSV；
- 原始统计和实验配置使用 UTF-8 JSON；
- 采样结果使用 UTF-8 TXT；
- 绘图函数统一返回 `(fig, axes)`，demo 负责保存；
- 不把经过手工修改的“漂亮结果”写进报告，JSON 必须来自实际张量统计。

五个 demo 完成后，在项目根目录运行 `python check.py --artifacts` 做最终组合验收。
