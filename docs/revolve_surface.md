# 旋转面（surface of revolution）

实现：`src/revolve_surface.py`。适用 `teapot_body`、`lid`、`knob`。
代理 = 一条 2D 母线 `r(h)` 绕一条**直轴**旋转：

```
S(h, θ) = O + h·A + r(h)·(cosθ·U + sinθ·V)
```

其中 `O` 轴上一点，`A` 单位轴向，`U、V` 与 `A` 正交的基。轴是直的 → 没有中心线曲率，
不变量是「轴 + 母线」。

## 流程（`fit_revolve_surface`）

1. **轴种子**：`best_principal_axis(points)`
2. **轴细化**：`refine_axis(...)`（带回退保护）
3. **母线**：`build_profile(...)`
4. **成面**：`generate_revolved_surface(...)`；`profile_polyline_3d` 导出母线 3D 折线。

## 关键决策一：三主轴圆度选轴

朴素做法是用惯量张量特征值的「间隙」判断哪个主轴是对称轴（`inertia_axis`，仍保留作参考）。
**但它对扁平 lid 失效**：薄圆盘的特征值排序会让启发式把轴选成**水平**方向，整个拟合崩掉
（曾出现 lid 轴水平、母线 X 形、chamfer 巨大）。

**正确做法**：`best_principal_axis` 直接尝试三个 PCA 主轴，对每个用
`_axis_circularity_score` 打分，选「垂直切片最圆」的那个：

```
circularity_score = mean over slices of ( angular_coverage · radius_tightness )
```

旋转面的真轴：沿轴方向切片应是接近完整的圆环（角覆盖高）且半径带很窄（紧致度高）。
这对**细长体**（高瓶）和**扁平体**（圆盘 / lid）都成立，与特征值排序无关。

## 关键决策二：轴细化 + 回退保护

`refine_axis`：交替执行「垂直当前轴切片 → 每片拟合圆心 → 对所有圆心用 SVD 拟合最佳直线，
重新定位 + 定向轴」，迭代 `n_iters` 次。旋转面所有圆形截面的圆心都共线于轴。

**保护**：扁平圆盘的圆心散布极小，SVD 定向容易**漂移**。因此在 `fit_revolve_surface` 里，
若细化后的圆度低于种子轴的圆度，则**回退**到种子轴：

```python
if circularity(refined) < circularity(seed) - 1e-6:
    origin, axis = origin0, axis0
```

## 母线 r(h)（`build_profile`）

沿轴把点投到 `(h, radius, angle)`，分 `n_slices` 个薄板：

- 每片取**稳健中位半径** `r_med`（旋转面半径带很窄，中位数抗离群）。
- 计算可靠度 `s_cov · s_res · s_count`（角覆盖 × 半径紧致 × 点数），并记录 `residual`、
  `angular_coverage`、`flags`（`low_coverage` / `high_scatter` / `collapsed`）。
- 用可靠度加权的 `UnivariateSpline` 对 `(h, r_med)` 拟合平滑母线，输出 `profile_h / profile_r`。

## 数据结构

- `RevolveConfig`：`n_profile=60, n_slices=60, n_iters=4, angular_resolution=64,
  end_skip_frac=0.04, min_reliability=0.2, profile_smooth=None, seed=0`。
- `ProfileSample`：`h, r, reliability, residual, angular_coverage, n_points, flags`。
- `RevolveResult`：`axis_origin, axis_dir, axis_u, axis_v, h_range, profile_h, profile_r,
  samples, surface, config, axis_segment, profile_polyline3d`。

## 导出 / 评估 / 可视化

- 导出 `proxy_export.export_revolve` → `proxy.ply`、`axis.ply`、`profile.ply`、`params.json`
  （`proxy_type="surface_of_revolution"`，含轴 `{origin, direction, u, v, h_range}`、
  母线 `{h, r}`、逐切片 `profile_slices`）。
- 评估 `evaluation.evaluate_revolve` = chamfer + coverage + `revolve_stats`
  （半径范围、`mean_radial_deviation`、`n_profile_slices`、`axis_length`、平均可靠度）。
- 调试图 `visualization.save_revolve_debug_figure`：点+轴 / 成面 vs 点 / 母线 `r(h)` 散点 /
  逐切片可靠度。

## 已知局限

- 母线是**单值** `r(h)`。薄圆盘（lid）的顶 / 底平面在同一高度对应多个半径，母线散点出现
  X 形交叉——这是单值母线的固有限制，可接受。轴方向已正确、chamfer 大幅下降。
