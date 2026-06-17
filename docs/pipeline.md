# 管线与模块（pipeline）

## 目标与前提

把分割后、经 CGAL alpha-wrapping 的逐零件网格拟合为低维参数化代理曲面。

输入：`input/alpha_wrapping_per_part/part_*_wrapped.obj`，每个约 500 顶点，
**封闭水密（ends sealed）**。零件元数据在 `input/teapot.json`、`input/pot.json`。

由前提推导出的硬约束（务必遵守）：

- 不使用 Point2CAD。
- 不依赖网格边界环 / 开口拓扑（包裹后没有开口可用）。
- 不用纯质心作为截面中心。
- 不把表面图最短路径直接当作最终中心线（仅作初值）。

## 两类代理与自动选型

| 代理类型 (`proxy_type`) | 几何不变量 | 适用零件 | 主流程函数 |
| --- | --- | --- | --- |
| `generalized_cylinder` | 弯曲中心线 + 截面 | `spout`、`handle` | `fit_swept_tube` |
| `surface_of_revolution` | 直轴 + 母线 `r(h)` | `teapot_body`、`lid`、`knob` | `fit_revolve_surface` |

选型逻辑在 `scripts/fit_spout_proxy.py: auto_proxy_type()`：按零件名映射
`body/lid/knob → revolve`，`spout/handle → generalized_cylinder`。
可用 `--proxy-type {auto,generalized_cylinder,revolve}` 覆盖。

> 几何直觉：spout/handle 是沿**弯曲中心线**扫掠的管，不变量是那条曲线；
> body/lid/knob 是绕**直轴**旋转的体，不变量是「轴 + 母线」。
> 用扫掠管去拟合圆形体会很别扭（body 是没有细长方向的圆容器，lid 是扁盘）。

## 共同前处理

1. `io_utils.load_mesh`：读网格，Scene 自动 concat。
2. `mesh_preprocess.clean_mesh`：merge_vertices、去退化面、去重复面、丢弃极小连通块
   （`min_component_face_ratio=0.05`）。
3. `mesh_preprocess.mesh_scale`：bbox 对角线，作为各处阈值的尺度基准。
4. `sampling.sample_surface`：表面采样 `n_points=20000`，返回点 + 法向。

之后按 `proxy_type` 进入扫掠管或旋转面分支（详见
[swept_tube.md](swept_tube.md)、[revolve_surface.md](revolve_surface.md)）。

## 模块职责

```
src/
  io_utils.py            网格/点云/折线/环读写 (Path3D 折线, 闭合环)
  mesh_preprocess.py     clean_mesh, mesh_scale
  sampling.py            sample_surface -> (points, normals)
  endpoint_estimation.py PCA 轴 + kNN 图 + 测地最远端点 (扫掠管用)
  centerline.py          中心线初始化 / 加权 B 样条 / 平行移动坐标系
  cross_section.py       切片选点 + 圆(Kasa/RANSAC)/椭圆(Fitzgibbon) 拟合 + 可靠度
  swept_tube_fitter.py   扫掠管主流程 + 自适应半径门控
  revolve_surface.py     旋转面主流程 (轴估计 + 母线 + 成面)
  proxy_export.py        export_proxy / export_revolve, 写 params.json
  evaluation.py          evaluate / evaluate_revolve (chamfer, coverage, 统计)
  visualization.py       save_debug_figure / save_revolve_debug_figure (matplotlib)
scripts/
  fit_spout_proxy.py     单零件 CLI, run() 返回 proxy_type
  run_all_parts.py       全零件批处理 + summary.json
notebooks/
  visualize_spout_proxy.ipynb  plotly 交互可视化 (按 proxy_type 切换图层)
```

## 输出布局

每个零件一个目录，文件名按代理类型不同（不带前缀，目录名即零件标识）：

```
output/
  spout/        proxy.ply  centerline.ply  cross_sections.ply  sampled_points.ply  params.json  debug.png
  handle/       (同上)
  teapot_body/  proxy.ply  axis.ply  profile.ply  sampled_points.ply  params.json  debug.png
  lid/          (同上)
  knob/         (同上)
  summary.json  # run_all_parts 写: 每零件 status + proxy_type + coverage
```

## 运行

环境（uv，**不要 conda**）：

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python \
    numpy scipy trimesh matplotlib scikit-learn plotly ipywidgets ipykernel nbconvert pandas
```

单零件：

```bash
.venv/bin/python scripts/fit_spout_proxy.py \
    --mesh input/alpha_wrapping_per_part/part_1_Teapot_Body_wrapped.obj \
    --output output
```

全部零件：

```bash
.venv/bin/python scripts/run_all_parts.py --input input --output output
```

可视化 notebook 内核为 `m2ps-venv`（`.venv`）。headless 校验：

```bash
.venv/bin/python -m nbconvert --to notebook --execute \
    --ExecutePreprocessor.kernel_name=m2ps-venv \
    --output /tmp/_nbtest.ipynb notebooks/visualize_spout_proxy.ipynb
```
