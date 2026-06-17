# 扫掠管 / 广义圆柱（generalized cylinder）

实现：`src/swept_tube_fitter.py`（主流程）+ `endpoint_estimation.py`、`centerline.py`、
`cross_section.py`。适用 `spout`、`handle`。代理 = 沿一条弯曲中心线扫掠的截面。

## 流程

1. **端点估计**（`endpoint_estimation.estimate_endpoints`）
   PCA 主轴给出初始方向 → `build_knn_graph` 建对称 kNN 图（连通性不足时自动增大 k）
   → Dijkstra 测地，取最远点对作为两端点。
2. **中心线初值**（`centerline.initial_centerline_from_graph`）
   两端点间图最短路 → 把路径点用 kNN 邻域均值**拉向内部**（避免贴着表面）→ 拉普拉斯平滑。
   > 最短路只作初值，不是最终中心线。
3. **交替细化**（`fit_swept_tube`，`n_iters=4`）
   每轮沿中心线取 `n_sections` 个 `u`：
   - 在垂直于切向的**薄板（slab）**里选点（`cross_section.select_slab_points`），
     带半径门控 + 基于法向的端帽剔除（`cap_normal_thresh`）。
   - 拟合截面（圆 Kasa/RANSAC+IRLS，或椭圆 Fitzgibbon），算可靠度。
   - 用可靠度加权重拟合中心线 B 样条（`fit_spline_centerline`，`splprep`）。
4. **成面**（`generate_swept_surface`）
   沿中心线建**平行移动坐标系**（parallel transport frames，避免扭转），每个截面生成一圈
   `angular_resolution` 个点，连成扫掠面。`ring_polylines` 导出横截面环。

## 截面可靠度

`cross_section._reliability` 综合多项打分（乘积）：

```
reliability = s_count · s_cov · s_res · s_aspect · s_size · (1 − end_penalty)
```

- `s_count`：内点数；`s_cov`：角覆盖（`_angular_coverage`）；`s_res`：拟合 RMS；
- `s_aspect`：椭圆长短轴比；`s_size`：相对 `max_radius` 的尺寸惩罚（超限→ `oversized` 标记）；
- `end_penalty`：两端 `end_skip_frac` 内线性降权（`_end_penalty`）。

`_interp_section_params` 按 `min_reliability` 过滤——低可靠度 / `oversized` 截面**不参与**
中心线重拟合与成面。

## 关键修复：自适应半径门控

**问题（handle）**：handle 是扁平且紧凑的 C 形（bbox ≈ `[0.36, 0.09, 0.60]`，真实管半径 ≈ 0.045）。
垂直薄板会同时切到 C 的**两条臂**，圆/椭圆拟合得到一个落在两臂之间的巨大半径（≈ 0.75，
大于零件尺度），中心线随之打圈（`curvature_max` 飙到 284）。

**修复**：在 `swept_tube_fitter` 引入随迭代更新的 `running_radius` 半径门控：

- `estimate_tube_radius(points, centerline)`：用「点到中心线中位距离」（scipy `cKDTree`）
  自举一个管半径作为初值。
- 薄板选点只保留 `radius_gate_factor · running_radius` 以内的点 → 切到对侧臂的远点被排除。
- 截面拟合带 `max_radius = max_radius_factor · running_radius`；超限截面被惩罚 / 标 `oversized`，
  排除在中心线重拟合与成面之外。
- 每轮用「几何估计 + 已拟合半径中位数」混合更新 `running_radius`。

默认 `radius_gate_factor=2.0`、`max_radius_factor=1.7`（从初版 3.0 / 2.5 收紧）。

**结果**：handle coverage 0.51 → 0.83，`curvature_max` 284 → 9.6；spout 不受影响（0.805）。
附带提升 body 0.18 → 0.61、lid 0.00 → 0.45（后来这两个改用旋转面更佳）。

## 关键参数（`FitConfig`）

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `n_sections` | 40 | 沿中心线的截面数 |
| `n_iters` | 4 | 交替细化轮数 |
| `cross_section_kind` | `"circle"`（CLI 默认 `ellipse`） | 截面类型 |
| `slab_width_frac` | 0.04 | 薄板宽度（占中心线长度比例） |
| `end_skip_frac` | 0.08 | 两端降权比例 |
| `cap_normal_thresh` | 0.7 | 端帽法向剔除阈值 |
| `min_reliability` | 0.15 | 进入成面的最低可靠度 |
| `angular_resolution` | 48 | 每环采样点数 |
| `radius_gate_factor` | 2.0 | 薄板半径门控倍数 |
| `max_radius_factor` | 1.7 | 截面最大半径倍数（超限即 oversized） |
| `spline_smooth` | None | 中心线样条平滑量（None=自适应） |

## 已知局限

- 仅适合**单一管状**零件；分叉 / 多分支不在范围内。
- 极扁的 C 形依赖半径门控；门控倍数过大会重新引入「切到对臂」的问题，过小会丢点。
