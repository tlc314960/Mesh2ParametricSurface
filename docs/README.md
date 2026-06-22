# 技术参考文档

本目录是 Mesh2ParametricSurface 的技术参考，记录设计决策、算法细节、参数语义、
踩过的坑与验证结果。面向需要修改 / 调参 / 复现的开发者。概览见根目录 [README](../README.md)。

| 文档 | 内容 |
| --- | --- |
| [pipeline.md](pipeline.md) | 整体管线、模块职责、I/O 约定、运行方式、环境 |
| [swept_tube.md](swept_tube.md) | 扫掠管（generalized cylinder）算法与自适应半径门控 |
| [revolve_surface.md](revolve_surface.md) | 旋转面（surface of revolution）轴估计与母线拟合 |
| [params_and_results.md](params_and_results.md) | 参数总表、`params.json` 字段、评估指标、当前结果 |
| [original_parts_preprocess.md](original_parts_preprocess.md) | 原始 AI FBX 分件、PLY 导出与 alpha-wrap 对齐诊断 |

## 核心结论速查

- 输入是 **watertight（封口）** 的 alpha-wrap 逐零件网格 → 不能用边界环 / 开口拓扑。
- 两类代理，按零件几何自动选型：
  - **扫掠管**（弯曲中心线 + 截面）：`spout`、`handle`。
  - **旋转面**（直轴 + 母线 `r(h)`）：`teapot_body`、`lid`、`knob`。
- 环境用 **uv**（不是 conda），`.venv` + python 3.11。
- 两个最关键的工程修复：
  1. handle 的**自适应半径门控**（避免切片同时切到 C 形两臂）。
  2. lid 的**三主轴圆度选轴 + 回退保护**（避免扁平圆盘把轴选成水平 / 漂移）。
