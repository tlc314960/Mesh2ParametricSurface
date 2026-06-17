# 参数、输出字段与结果

## `params.json` 字段

两种代理共有：`part_name`、`source_mesh_path`、`proxy_type`、`fitting_metrics`、`config`。

### `proxy_type = "generalized_cylinder"`

```jsonc
{
  "proxy_type": "generalized_cylinder",
  "cross_section_type": "ellipse",
  "endpoints": { "a": [x,y,z], "b": [x,y,z] },
  "centerline": {
    "sampled_points": [[x,y,z], ...],
    "bspline": { "control_points": [...], "knots": [...], "degree": 3 }
  },
  "cross_sections": [
    { "u":..., "center3d":[...], "tangent":[...], "axis_u":[...], "axis_v":[...],
      "cross_section_type":"ellipse", "a":..., "b":..., "theta":...,
      "radius":..., "reliability":..., "residual":..., "angular_coverage":...,
      "n_points":..., "flags":{...} }
  ]
}
```

### `proxy_type = "surface_of_revolution"`

```jsonc
{
  "proxy_type": "surface_of_revolution",
  "axis": { "origin":[...], "direction":[...], "u":[...], "v":[...], "h_range":[lo,hi] },
  "profile": { "h":[...], "r":[...] },
  "profile_slices": [
    { "h":..., "r":..., "reliability":..., "residual":...,
      "angular_coverage":..., "n_points":..., "flags":{...} }
  ]
}
```

## 评估指标（`fitting_metrics`）

共有：

| 指标 | 含义 |
| --- | --- |
| `chamfer` | 输入点 ↔ 代理面 双向倒角距离 |
| `coverage_ratio` | 输入点被代理面覆盖的比例（阈值 `0.03·scale`） |
| `mean_input_to_proxy` | 输入点到代理面平均距离 |
| `mean_proxy_to_input` | 代理面到输入点平均距离 |

扫掠管专有：`n_sections`、`pct_unreliable`、`mean_reliability`、`radius_mean`、
`centerline_length`、`curvature_max` 等（`section_stats` + `centerline_stats`）。

旋转面专有（`revolve_stats`）：`n_profile_slices`、`pct_unreliable`、`mean_reliability`、
`mean_slice_residual`、`mean_angular_coverage`、`radius_min/max/mean`、`axis_length`、
`mean_radial_deviation`。

## 配置默认值

### `FitConfig`（扫掠管）

| 参数 | 默认 |
| --- | --- |
| `n_sections` | 40 |
| `n_iters` | 4 |
| `cross_section_kind` | `circle`（CLI 默认 `ellipse`） |
| `slab_width_frac` | 0.04 |
| `end_skip_frac` | 0.08 |
| `cap_normal_thresh` | 0.7 |
| `min_reliability` | 0.15 |
| `angular_resolution` | 48 |
| `radius_gate_factor` | 2.0 |
| `max_radius_factor` | 1.7 |
| `spline_smooth` | None |
| `seed` | 0 |

### `RevolveConfig`（旋转面）

| 参数 | 默认 |
| --- | --- |
| `n_profile` | 60 |
| `n_slices` | 60 |
| `n_iters` | 4 |
| `angular_resolution` | 64 |
| `end_skip_frac` | 0.04 |
| `min_reliability` | 0.2 |
| `profile_smooth` | None |
| `seed` | 0 |

## 当前结果（`run_all_parts.py` 自动选型）

| 零件 | 代理类型 | coverage | 备注 |
| --- | --- | --- | --- |
| handle | generalized_cylinder | 0.830 | 自适应半径门控后 `curvature_max` 284→9.6 |
| spout | generalized_cylinder | 0.805 | 基线，弯曲 S 形中心线 |
| knob | surface_of_revolution | 0.759 | 旋转面 0.65→0.76 |
| teapot_body | surface_of_revolution | 0.685 | 旋转面 0.61→0.69 |
| lid | surface_of_revolution | 0.568 | 旋转面 0.45→0.57；薄盘 r(h) 有 X 形交叉（固有限制） |

旋转面在圆形零件上全面优于扫掠管。spout 在引入半径门控前后 coverage 基本不变（~0.805），
说明门控不会损害本就良好的管状拟合。

## 变更脉络（按时间）

1. 搭建扫掠管全流程（spout 验证 coverage 0.82）。
2. 输出重组为逐零件目录 + `summary.json`，文件名去前缀。
3. handle 自适应半径门控修复（C 形两臂问题）。
4. 新增旋转面代理（`revolve_surface.py`）+ 导出 / 评估 / 可视化 / CLI 选型。
5. 三主轴圆度选轴 + 回退保护，修复 lid 轴水平 / 漂移。
6. notebook 按 `proxy_type` 分支渲染；headless 校验 spout 与 teapot_body 两条路径。
