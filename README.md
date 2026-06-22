# Mesh2ParametricSurface

把分割后、经 alpha-wrapping 处理的零件网格（watertight、封口）拟合为**低维参数化代理曲面**。
针对茶壶（teapot）的五个零件，按几何形态自动选择两种代理类型：

| 代理类型 | 几何不变量 | 适用零件 | 输出曲线 |
| --- | --- | --- | --- |
| `generalized_cylinder`（扫掠管） | 弯曲中心线 + 截面 | `spout`、`handle` | 中心线 + 横截面环 |
| `surface_of_revolution`（旋转面） | 直轴 + 母线 `r(h)` | `teapot_body`、`lid`、`knob` | 旋转轴 + 母线 + 切片环 |

> 详细技术参考见 [`docs/`](docs/README.md)：
> [管线与模块](docs/pipeline.md) · [扫掠管](docs/swept_tube.md) ·
> [旋转面](docs/revolve_surface.md) · [参数与结果](docs/params_and_results.md)。

## 设计前提

输入是 CGAL alpha-wrap 的逐零件网格（`input/alpha_wrapping_per_part/part_*_wrapped.obj`，约 500 顶点），
它们是**封闭水密**的，端面被封住。因此：

- 不使用 Point2CAD。
- 不依赖网格边界环 / 开口拓扑（包裹后没有开口）。
- 不用纯质心作为截面中心。
- 不把表面图最短路径直接当作最终中心线。

## 整体路线

```
                 alpha-wrapped part mesh (watertight)
                              │
                  清洗 + 表面采样点 (法向)
                              │
          ┌───────────────────┴────────────────────┐
   tube-like (spout/handle)             round (body/lid/knob)
   generalized_cylinder                 surface_of_revolution
          │                                     │
   PCA+kNN 图测地端点                      三主轴圆度打分选旋转轴
   图最短路初始化中心线(拉向内部)            交替细化轴(穿过切片圆心,带回退保护)
   交替细化:                              切片求稳健中位半径 + 可靠度
     · 垂直切片 (自适应半径门控)            母线样条 r(h)
     · 法向剔除端帽                          │
     · RANSAC 圆/椭圆拟合                   旋转生成曲面 S(h,θ)
     · 可靠度加权 B 样条中心线
          │
   平行移动坐标系扫掠生成曲面
                              │
              评估 (chamfer / coverage / 可靠度) + 调试图
                              │
            output/<part>/{proxy.ply, ..., params.json, debug.png}
```

### 扫掠管（generalized cylinder）

`src/swept_tube_fitter.py`。流程：PCA 轴 + kNN 图测地最远端点 → 图最短路中心线初值（拉向内部）→
交替细化（垂直切片、法向剔除端帽、RANSAC 圆/椭圆拟合、可靠度加权 B 样条中心线重拟合）→ 平行移动坐标系扫掠成面。

**关键：自适应半径门控**。handle 是一个扁平且紧凑的 C 形，垂直切片会同时切到 C 的两条臂，
使圆/椭圆拟合得到一个落在两臂之间的巨大半径，导致中心线打圈。修复方式：

- `estimate_tube_radius()` 用「点到中心线中位距离」(scipy cKDTree) 自举管半径；
- 切片只保留 `radius_gate_factor * running_radius` 以内的点；
- 截面拟合带 `max_radius`，对 `oversized` 截面降权/打标记，使其不参与中心线重拟合与成面；
- 每轮更新 `running_radius`。默认 `radius_gate_factor=2.0`、`max_radius_factor=1.7`。

### 旋转面（surface of revolution）

`src/revolve_surface.py`。流程：

1. **旋转轴**：`best_principal_axis()` 尝试三个 PCA 主轴，选「垂直切片最圆」的那个
   （`_axis_circularity_score` = 角覆盖 × 半径紧致度的均值）。这比「特征值间隙」启发式稳健——
   后者会把扁平的 lid 选成水平轴。
2. **轴细化**：`refine_axis()` 用穿过各切片圆心的最佳拟合直线 (SVD) 迭代；带回退保护——
   若细化降低了圆度则放弃（扁平圆盘的圆心散布极小，易漂移）。
3. **母线**：逐切片取稳健中位半径 + 可靠度，对 `(h, r)` 拟合加权平滑样条。
4. **成面**：母线绕直轴旋转 `S(h, θ) = O + h·A + r(h)·(cosθ·U + sinθ·V)`。

## 输出布局

每个零件一个目录，文件名按代理类型不同：

```
output/
  spout/        proxy.ply  centerline.ply  cross_sections.ply  sampled_points.ply  params.json  debug.png
  handle/       (同上)
  teapot_body/  proxy.ply  axis.ply  profile.ply  sampled_points.ply  params.json  debug.png
  lid/          (同上)
  knob/         (同上)
  summary.json  # 每个零件的状态 + proxy_type + coverage
```

`params.json` 带 `proxy_type` 字段：扫掠管含端点、中心线 B 样条控制点、各横截面参数；
旋转面含轴 `{origin, direction, u, v, h_range}`、母线 `{h, r}`、逐切片诊断。

## 安装

需要 Python 3.10+（开发用 3.11）。环境用 **uv**（不是 conda），在项目目录下创建 `.venv`：

```bash
# 1. 克隆
git clone https://github.com/tlc314960/Mesh2ParametricSurface.git
cd Mesh2ParametricSurface

# 2. 建虚拟环境并装依赖（依赖清单见 requirements.txt）
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt

# 3. 自检：拟合全部零件
.venv/bin/python scripts/run_all_parts.py --input input --output output
```

> 没有 uv 时也可用标准 venv：`python -m venv .venv && .venv/bin/pip install -r requirements.txt`。

## 使用

拟合单个零件（`--proxy-type auto` 按零件名自动选型）：

```bash
.venv/bin/python scripts/fit_spout_proxy.py \
    --mesh input/alpha_wrapping_per_part/part_1_Teapot_Body_wrapped.obj \
    --output output
```

一次拟合全部零件并写 `summary.json`：

```bash
.venv/bin/python scripts/run_all_parts.py --input input --output output
```

`--proxy-type` 可选 `auto` / `generalized_cylinder` / `revolve`；`auto` 将
`body/lid/knob → revolve`，`spout/handle → generalized_cylinder`。

## 原始 AI 网格分件预处理

详细外表面使用原始 AI 生成网格，而不是 alpha-wrap 网格。两类数据职责不同：

- `input/alpha_wrapping_per_part/`：仅用于拟合平滑参数化代理。
- `input/TeportParts.fbx`：保留视觉细节，预处理后供后续外表面采样。

使用 Blender 正确解析 FBX 层级和物体变换，并导出逐零件 PLY：

```bash
.venv/bin/python scripts/preprocess_original_fbx_parts.py \
    --fbx input/TeportParts.fbx \
    --alpha-dir input/alpha_wrapping_per_part \
    --output input/original_parts_ply
```

输出包括 `part_*_original.ply`、`part_mapping.json`、
`preprocess_summary.json` 和 `debug_alignment.png`。该步骤只做保守清理、
三角化、法向检查和坐标对齐诊断，不做平滑、简化、alpha-wrap 或外层提取。
技术细节见 [`docs/original_parts_preprocess.md`](docs/original_parts_preprocess.md)。

## 交互式可视化

`notebooks/visualize_spout_proxy.ipynb`（plotly）。设置 `PART` 切换零件查看单零件细节，
底部「All parts」对 `output/` 下每个零件渲染组合视图 + 可靠度视图 + 指标总表。
notebook 会按 `proxy_type` 自动切换图层（扫掠管显示中心线/横截面/端点；旋转面显示轴/母线/切片环）。

先把 `.venv` 注册为 Jupyter 内核 `m2ps-venv`，再在该内核下运行 notebook：

```bash
.venv/bin/python -m ipykernel install --user --name m2ps-venv --display-name "Python 3 (.venv)"
```

## 当前结果

`run_all_parts.py` 自动选型（coverage = 代理覆盖输入点的比例）：

| 零件 | 代理类型 | coverage |
| --- | --- | --- |
| handle | generalized_cylinder | 0.830 |
| spout | generalized_cylinder | 0.805 |
| knob | surface_of_revolution | 0.759 |
| teapot_body | surface_of_revolution | 0.685 |
| lid | surface_of_revolution | 0.568 |

旋转面在圆形零件上明显优于扫掠管（lid 0.45→0.57，body 0.61→0.69，knob 0.65→0.76）。
已知局限：lid 是薄圆盘，顶/底平面在同一高度对应多个半径，单值母线 `r(h)` 会出现 X 形交叉，属固有限制。

## 代码结构

```
src/
  io_utils.py            网格/点云/折线/环的读写
  mesh_preprocess.py     清洗网格、计算尺度
  sampling.py            表面采样 (点 + 法向)
  endpoint_estimation.py PCA 轴 + kNN 图测地端点
  centerline.py          中心线初始化 / B 样条 / 坐标系
  cross_section.py       切片选点 + 圆/椭圆拟合 + 可靠度
  swept_tube_fitter.py   扫掠管主流程 (自适应半径门控)
  revolve_surface.py     旋转面主流程 (轴估计 + 母线)
  proxy_export.py        导出 proxy/params (两种代理类型)
  evaluation.py          chamfer / coverage / 截面 / 母线统计
  visualization.py       matplotlib 调试图 (两种代理类型)
scripts/
  fit_spout_proxy.py     单零件拟合 CLI (--proxy-type)
  run_all_parts.py       全零件批处理 + summary.json
  preprocess_original_fbx_parts.py  原始 FBX 分件与对齐诊断
notebooks/
  visualize_spout_proxy.ipynb  plotly 交互可视化
```
