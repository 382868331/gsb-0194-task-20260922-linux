# 透镜畸变映射的可逆区域检查

离线机器视觉工程用可复用模块：归一化平面上的透镜畸变正/逆映射，
并对逆映射做**可逆区域检查**（多解枚举 + 径向唯一性证书）。

- **Python 版本**：3.14.7（仅用标准库，无第三方依赖）
- **平台**：Linux 原生，安装 Python 即可运行，无需网络/账号/硬件/Docker
- **安装**：无需安装任何依赖，直接 `import lensdistort`

## 模型

归一化平面坐标，r² = x² + y²：

```
s  = 1 + k1·r² + k2·r⁴
x' = x·s + 2·p1·x·y + p2·(r² + 2·x²)
y' = y·s + p1·(r² + 2·y²) + 2·p2·x·y
```

`p1 = p2 = 0` 时为纯径向模型，可走径向证书路径（可证明完备/证无解）。

## 接口

```python
from lensdistort import (
    DistortionModel,          # 畸变系数（构造即校验）
    distort_point, distort_points,      # 正向映射
    jacobian,                            # 解析 Jacobian
    undistort_point, undistort_points,  # 逆映射
    radial_uniqueness_certificate,      # 径向唯一性证书
    solve_radial, critical_radii, monotonic_segments,
    Candidate, UndistortResult, RadialCertificate,
    STATUS_CONVERGED, STATUS_EXHAUSTED, STATUS_PROVEN_NO_SOLUTION,
    InvalidInputError, LensDistortError, MAX_POINTS,
)
```

### 正向

```python
m = DistortionModel(k1=-2.0, k2=1.0)
xd, yd = distort_point(m, 0.3, 0.0)
pts = distort_points(m, [(0.1, 0.2), (0.3, -0.1)])
```

### 逆向（多起点阻尼 Newton + 残差检验 + 去重）

```python
res = undistort_point(m, xd, yd, box_radius=1.5)
# res.status: "converged" | "exhausted" | "proven_no_solution"
# res.candidates: tuple[Candidate(x, y, residual)]
# res.completeness: 完备性说明（Newton 不证明完备；径向证书完备）
# res.notes: 附加说明（如原点存在整圆原像）
```

`undistort_point` 关键字参数：

| 参数 | 默认 | 含义 |
|---|---|---|
| `method` | `"auto"` | `"auto"`（径向→证书，否则 Newton）/ `"newton"` / `"radial"` |
| `max_starts` | 64 | Newton 起点数上限（确定性起点集） |
| `max_iterations` | 60 | 每起点 Newton 迭代上限 |
| `tol` | 1e-12 | Newton 收敛残差阈值 |
| `accept_tol` | 1e-9 | 候选残差接受阈值 |
| `dedup_tol` | 1e-7 | 候选去重距离 |

### 径向唯一性证书（p1=p2=0 特例）

```python
cert = radial_uniqueness_certificate(m, box_radius=1.0)
# cert.is_globally_injective: 盒内是否至多一个原像
# cert.segments: g 的单调段（方向 + 端点值）
# cert.max_candidate_radii: 候选半径数上界
# cert.statement: 人类可读证书陈述
```

g(r) = r(1+k1·r²+k2·r⁴)，g'(r)=1+3k1·r²+5k2·r⁴。令 u=r²，
g'(r)=0 化为二次方程，最多 2 个正根 ⇒ g 在 [0, r_max] 最多 3 个单调段。
每段是双射，故逐段枚举完备（**不取首根**）。

## 输入上限与错误语义

- **点数上限**：`distort_points` / `undistort_points` 最多 **100** 点（`MAX_POINTS`）。
- **搜索盒**：`[-R, R]²`，`R > 0` 有限。
- **数值输入**：拒绝 `NaN` / `±Inf`；拒绝 `bool` 冒充数值（`bool` 是 `int` 子类，显式排除）。
- **整数参数**（`max_starts` 等）：拒绝 `bool`，必须 ≥ 1。
- **非法输入**：抛 `InvalidInputError`（继承 `ValueError`），消息含出错参数名，**不静默纠正**。
- **批次原子性**：任一点非法 → 整体拒绝，不返回部分结果。
- **状态区分**：`exhausted`（Newton 预算耗尽，**不证明无解**）≠ `proven_no_solution`（径向证书**证明**无解）。

## 误差口径（浮点结果）

- 正向 `distort_point`：float64 精确求值，误差 ~1 ulp。
- 逆映射候选 `residual = max(|F(z)-target|∞)`，接受阈值 `accept_tol=1e-9`。
- Newton 收敛 `tol=1e-12`；径向二分 200 次（区间缩 2⁻²⁰⁰，远超 float64）。
- **切触/重根**（g'(r)=0 且 g(r)=ρ）：浮点下 g 在重根附近下溢，
  二分只能到 √eps≈1e-8 精度；库改用**临界点精确判定**（g'(p)=0 的 p 精确已知）。
- 往返一致性（distort∘undistort）仅作**补充**验证，不作为正确性依据。

## 设计取舍

- **多起点阻尼 Newton**：起点 = 裁剪目标点 + 原点 + 5×5 网格（确定性）；
  奇异 Jacobian 用 Levenberg 式正则化，不崩溃；步长减半回溯（阻尼）。
- **径向证书优先**：`method="auto"` 时径向模型走证书路径（可证完备/证无解），
  一般模型走 Newton（耗尽≠证无解）。`method="newton"/"radial"` 可强制。
- **原点整圆原像**：g(r)=0 的正根对应整圆映射到原点（连续统），
  在 `notes` 中显式说明，不当作普通多解。

## 已知限制

- 仅实现题面给定的 5 参数模型（k1,k2,p1,p2），**不宣称**兼容完整行业标准
  （如 OpenCV 的 k3-k6、薄棱镜、鱼眼等）。
- Newton 路径**不证明完备性**（可能漏根）；完备性仅径向模型由证书保证。
- 切触重根处的半径精度受 float64 限制（√eps 量级），已用临界点精确判定缓解。
- 未做大规模性能优化；单点逆解为毫秒级。

## 运行

```bash
# 测试（37 个用例）
python -m unittest discover -s tests -v

# 演示（<8s，真实计算：多解+残差、盒外证无解、输入拒绝）
python demo.py
```

## 文件

```
lensdistort/
  __init__.py      # 公开接口导出
  errors.py        # LensDistortError / InvalidInputError
  _validation.py   # 输入校验（NaN/Inf/bool/上限）
  model.py         # DistortionModel + distort + jacobian
  radial.py        # 径向 g/g'、临界点、单调段、证书、solve_radial
  inverse.py       # 多起点阻尼 Newton + 去重 + 状态机
tests/
  test_model.py    # 正向解析值 + 有限差分 Jacobian 参考
  test_radial.py   # 临界点解析值 + 证书 + solve_radial（独立二分参考）
  test_inverse.py  # 无畸变/折叠多根/盒外/中心/奇异点/Newton/批次/校验
demo.py            # 固定输入演示（多解+残差、盒外证无解、NaN 拒绝）
```
