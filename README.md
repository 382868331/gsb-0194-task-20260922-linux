# distortion — 透镜畸变映射的可逆区域检查

在归一化平面上对四系数多项式透镜畸变模型做**前向映射**与**有界盒逆解**，
并在纯径向下给出**分段单调性唯一性/无解证书**。仅使用 Python 3.14
标准库，离线可导入，无第三方依赖，不调用任何同类第三方实现替代核心算法。

本版新增**连续分支跟踪入口**：给定一串畸变后坐标和首个未畸变锚点，逐帧
仍由原求解器给出全部残差合格候选，再选距上一结果最近且不超过显式步长
上限的根；等距根在容差内判为 ambiguous，绝不按列表顺序偷选；无候选或
越步即断开轨迹，必须用新锚点才能继续。跟踪只是分支连续性启发式，**不
改变逆函数的多根性质**，原完备性限制与径向证书逐帧保留。

## 环境与运行

- Python：**3.14.7**（仅用标准库 `math`；3.10+ 亦可运行，但以 3.14.7 为准）
- 依赖：无。安装 Python 后无需 `pip install` 任何包。
- 网络 / 账号 / 硬件 / 数据库 / 中间件 / Docker：均不需要。

准确命令（在仓库根目录执行）：

```bash
./.venv/bin/python -m unittest discover -s tests -v   # 全部测试（含跟踪）
./.venv/bin/python demo.py                            # 固定输入演示（实测 < 1 秒，预算 90 秒）
```

## 数学模型（本题契约的精确子集）

```
x' = x·(1 + k1·r² + k2·r⁴) + 2·p1·x·y + p2·(r² + 2·x²)
y' = y·(1 + k1·r² + k2·r⁴) + p1·(r² + 2·y²) + 2·p2·x·y
r² = x² + y²
```

这是本题指定的精确 4 系数子集；**不宣称**与 OpenCV/KB 等行业标准
兼容（系数符号约定、切向项约定均只按本题公式）。

## 公开接口

```python
from distortion import (DistortionModel, Candidate, InverseResult,
                        TrackStep, TrackResult, track_undistorted,
                        OK, AMBIGUOUS, NO_CANDIDATE, OVER_STEP,
                        INVALID_ANCHOR, NEEDS_ANCHOR, finite_real)

m = DistortionModel(k1=0.0, k2=0.0, p1=0.0, p2=0.0)

# 前向
xp, yp = m.distort(x, y)                 # 单点，输入必须为有限实数
xys      = m.distort_all([(x, y), ...])  # 批量，最多 100 点

# 逆解：在闭盒 [-R,R]² 内寻找全部经残差检验、去重后的候选
res: InverseResult = m.undistort(xp, yp, R,
                                 residual_tol=1e-10,   # >0 有限实数
                                 starts_per_axis=5,    # 1..12 的真整数
                                 max_iterations=80)    # 1..10000 的真整数
results = m.undistort_all([(xp, yp), ...], R, **同上)

# 连续分支跟踪：一串畸变坐标 + 首个未畸变锚点
track = m.track(
    [(xp0, yp0), (xp1, yp1), ...],  # 畸变后坐标，1..100 个
    anchor=(x0, y0),                 # targets[0] 的未畸变锚点（盒内）
    R=3.0,
    max_step=0.05,                   # 必填：未畸变坐标上的逐帧步长硬上限
    tie_tol=1e-9,                    # 距离差 <= tie_tol 视为同距 -> ambiguous
    anchor_tol=1e-8,                 # 锚点正向校验残差上限
    resume_anchors={3: (x3, y3)},    # 可选：断点后某帧的新锚点
    residual_tol=1e-10,              # 以下参数原样透传给原求解器
    starts_per_axis=5, max_iterations=80)
# 或函数式：track_undistorted(m, targets, anchor, R, max_step=..., ...)
for step in track:                   # step: TrackStep
    step.status                      # OK / AMBIGUOUS / NO_CANDIDATE /
                                     # OVER_STEP / INVALID_ANCHOR / NEEDS_ANCHOR
    step.x, step.y, step.distance    # 选中根与到上一根的距离
    step.candidates                  # 该帧原求解器的全部合格候选（始终保留）
    step.proven_no_solution          # 逐帧保留的径向无解证书
    step.budget_exhausted            # 逐帧保留的数值完备性限制标志
track.active        # 末帧是否仍在轨迹上
track.broken        # 是否曾断开
track.break_index   # 首个断点帧号（无则 None）
track.last          # 最近一个被选根 (x, y) 或 None

**跟踪语义（每帧）**

- 第 0 帧必须给出锚点，且通过**正向校验**：
  残差 ‖distort(anchor) − target‖ ≤ anchor_tol，否则该帧为
  INVALID_ANCHOR，轨迹立即断开。求解器仍照常运行并保留候选。
- 之后每帧先调用原 undistort_points 得到全部残差合格候选，再取距上一
  选中根欧氏距离最小者：
  - 最近距离 > max_step → OVER_STEP，宁断不跳；
  - 有两个及以上候选的距离与最小距离之差 ≤ tie_tol → AMBIGUOUS，
    **不选根、不按候选列表顺序决断**，候选全部保留；
  - 无候选 → NO_CANDIDATE（径向路径可同时带 proven_no_solution）。
- 上述任一情况都**断开轨迹**：其后帧在没有新锚点前为 NEEDS_ANCHOR，
  且**不再运行求解器**；新锚点通过 resume_anchors={帧号: (x,y)} 给出，
  与首锚点同样做盒内检查和正向校验。
- 步长、容差都是**未畸变平面坐标**上的绝对距离；调用方必须显式给出
  max_step（没有默认步长，避免静默跳跃）。
- 跟踪不把逆函数变成唯一解：每帧 candidates、proven_no_solution、
  budget_exhausted 与证书路径标志原样保留；切向（通用）路径的多根
  集合仍是数值搜索结果，不证明完备；径向特例仍走分段证书。
```

`InverseResult` 字段：

| 字段 | 含义 |
|---|---|
| `candidates` | 全部通过残差检验并去重的 `Candidate`，按残差升序 |
| `unique` | 恰有一个候选（径向情形为分段证书；通用情形仅描述本次搜索结果） |
| `proven_no_solution` | **证书性**无解（只有纯径向路径可能置真） |
| `budget_exhausted` | 有数值起点在预算内未收敛到已接受候选；**不证明无解**，只表示完备性未确立 |
| `radial` | 是否走了纯径向上的证书路径（`p1 == p2 == 0`） |
| `critical_radii` | 径向路径上 `g'(r)=0` 且位于盒内的临界点 |
| `certificate` | 人类可读的证书/搜索说明（明确写出"数值搜索不证明完备"） |

`Candidate` 字段：`x, y, residual, method`；`method` 为
`"radial_certified"` 或 `"damped_newton"`；`residual =
‖distort(x,y) − (xp,yp)‖₂`。候选可解包为 `(x, y)`。

### 输入上限与错误语义

- 批量接口（`distort_all` / `undistort_all`）**最多 100 点**，超限抛
  `ValueError`。
- 所有数值输入拒绝 `NaN` 与 `±Infinity`（`ValueError`，消息带字段名，
  如 `"xp: must be finite ..."`）。
- 整数参数（`starts_per_axis`、`max_iterations`）拒绝 `bool`
  （`TypeError`）；类型错误抛 `TypeError` 并带字段名。
- `R > 0`、`residual_tol > 0`、`starts_per_axis ∈ [1,12]`、
  `max_iterations ∈ [1,10000]`，违反抛 `ValueError`。
- 批量操作**先整体校验后计算**：任何非法输入都不会产生部分结果列表；
  模型本身无状态，不存在未说明的部分变更。
- **搜索预算耗尽与证明无解严格分开**：通用 Newton 路径永不置
  `proven_no_solution`；证书无解只可能来自径向分段路径。

## 算法与设计取舍

### 前向与解析 Jacobian

前向公式直接按契约实现；Jacobian 由公式解析求导得到（对称）：

```
d = k1 + 2·k2·r²
J00 = S + 2x²d + 2p1 y + 6p2 x
J01 = 2xy d + 2p1 x + 2p2 y
J11 = S + 2y²d + 6p1 y + 2p2 x        (S = 1 + k1 r² + k2 r⁴)
```

测试用**独立编写的中心有限差分**参考逐元素核对（不经过被测实现）。

### 纯径向特例：径向导数分段证书（不取首根）

`p1 = p2 = 0` 时映射沿射线退化为标量函数
`g(r) = r(1 + k1 r² + k2 r⁴)`，且

```
g'(r) = 1 + 3 k1 r² + 5 k2 r⁴
```

令 `q = r²`，临界点由二次方程 `5 k2 q² + 3 k1 q + 1 = 0` 的正根给出。
所有盒内临界点把 `[0, R_ray]` 分成若干 **g 严格单调段**，逐段二分
枚举 `g(r) = ±s` 的根——每段至多一根，因此：

- 根的个数是**枚举出来的完备计数**（同时查目标射线方向 `+s` 与折叠
  分支 `−s`，不会"取首根"了事）；
- 恰一根 ⇒ `unique=True`；零根且目标半径超出端点/临界点处的
  `max|g|` ⇒ `proven_no_solution=True`；
- 方盒沿目标射线的可达半径为 `R/max(|ux|,|uy|)`（角点方向可达
  `R√2`）；目标为原点时预像可能是整圆，圆与方盒相交的临界半径为
  `R√2`，每个相交圆用 45° 射线上的一个代表点表示（连续谱不逐点枚举，
  证书文字中明确说明）。

**切根处理**：目标恰在临界值（`g'(rc)=0`，即映射雅可比行列式为零的
奇异点）时为二重切根。实现用窄浮点误差带（见下）识别切根，合并二分
在其两侧可能产生的一对邻根，返回单个临界点候选；误差带内的判定属于
"按误差口径的证书"，带外严格分类。

### 通用情形：多起点阻尼 Newton（Levenberg 式 + Armijo 回退）

- 确定性起点集：`n×n` 网格（默认 5×5）、4 圈同心环（40 点）、目标点
  及其四点微扰、原点（共 71 个起点）。
- 每点解 `(JᵀJ + λI)Δ = −Jᵀf`；λ 自适应阻尼；Armijo 回退保证残差
  单调下降；**硬盒约束**通过缩短步长（而非截断）满足。
- Newton 方向无法取得下降时回退到归一化的 `−Jᵀf` 最速下降方向逃逸
  平坦区/鞍点。
- 收敛阈值 `‖f‖₂ ≤ residual_tol`；候选按位置去重
  （`1e-8·max(1,R)`），按残差升序。
- 在未解析点（含 `‖f‖²` 的伪局部极小）结束的起点计入预算标志；即使
  已找到候选，该标志仍诚实保留。**有限起点+有限迭代不证明完备**，
  certificate 文本与 README 均显式声明。

## 浮点结果的误差口径

### 连续分支跟踪：最近根 + 显式步长上限（不取首根）

逆解返回的 candidates 按残差排序，与几何上的“同一分支”无关；逐帧
取列表第一根会在折叠映射上随机跳支。跟踪入口（distortion.tracking）
因此：

1. 第 0 帧只接受调用方显式给出的未畸变锚点，并做正向校验
   （残差 <= anchor_tol）；锚点必须在闭盒内。
2. 每一帧原求解器照常完整运行（径向上仍是分段证书，通用情形仍是
   多起点 Newton），候选、证书、预算标志全部保留在 TrackStep 上。
3. 在合格候选中选距上一选中点欧氏距离最小者；最小距离超过显式
   max_step 即断开，绝不硬跳。
4. 两个以上候选到上一点的距离差在 tie_tol 内时判 AMBIGUOUS：不选根、
   不参考列表顺序，候选原样保留。
5. 断点之后求解器停止运行，帧状态为 NEEDS_ANCHOR，直到调用方在
   resume_anchors 给出新锚点（同样正向校验）才续跟。

该机制只利用“相邻帧未畸变坐标变化不大”的连续性假设，不改变、也不
声称逆函数唯一；通用路径的完备性限制与径向特例证书逐帧可见。

- 前向映射：直接按 IEEE-754 双精度求值，单份坐标的典型相对误差
  `O(ε)`（ε≈2.22e-16）；系数与坐标在 `|·| ≲ O(1)` 时绝对误差量级
  `1e-15`（测试断言到 1e-12）。
- Newton 候选：默认 `residual_tol = 1e-10`，即每个接受候选都满足
  重投影残差 `≤ 1e-10`（归一化平面单位）；实测正常点残差多在
  `1e-14` 以下。**不用宽容差掩盖边界错误**：盒外/奇异点另有证书与
  专门测试，而非靠放宽残差通过。
- 径向二分：段内收缩到区间宽 `≤ 1e-14·max(1,R)`，根的坐标误差约
  `1e-14·max(1,R)`，残差通常 `≤ 1e-13`。
- 证书的浮点噪声带：多项式求值采用
  `band(v) = 64ε·max(1,|v|)`。仅当目标超出 `max|g| + band` 才判
  证书无解；仅当临界值差距在 `band` 内才按切根合并。即"精确数学结论
  成立到此误差带为止"，误差带边缘的目标不做更强制的断言。
- 去重距离：`1e-8·max(1,R)`（通用路径）/ `1e-11·max(1,R)`（径向
  段间）。彼此比去重距离更近的两个真根会被当作一个候选——这是已知
  分辨率限制（见下）。

## 已知限制

1. 通用（含切向）路径的多根集合是**数值搜索结果，不是完备证明**；
   起点采样未覆盖的吸引盆可能漏根。需要完备性时应令 `p1=p2=0` 使用
   证书路径。
2. 通用路径永不声明"无解"——目标盒内无候选时只表示本次搜索未找到，
   `proven_no_solution` 保持 `False`。
3. 原点在折叠径向下的预像是整个圆（连续谱）；接口返回每个相交圆的
   一个代表点，不枚举连续谱，`unique` 在多圆时为 `False`。
4. 去重具有有限空间分辨率（见误差口径）；两个比去重阈值更近的不同
   真根会计为一个。
5. 输入量级无显式上限（仅要求有限）；极大 `R`/系数下五次多项式求值
   的舍入相应增大，证书噪声带随之放大。
6. 只实现本题公式的 4 系数模型；不做相机标定、畸变系数约定转换或
   行业标准兼容性声明。
7. 批量上限 100 是接口边界而非性能建议；演示用例不穷举该规模。
8. 连续分支跟踪的"最近根 + 步长上限"是**启发式**，不证明所选根对应
   物理上正确的轨迹；帧间真实运动超过 max_step 时会被当作断开处理。
   同距歧义必须由调用方用新锚点解决，库不做任意裁决。

## 仓库结构

```
distortion/__init__.py    # 公开 API
distortion/numeric.py     # 有限实数/整数校验
distortion/model.py       # 前向模型、解析 Jacobian、批量接口
distortion/radial.py      # 纯径向分段单调性证书求解
distortion/inverse.py     # 结果类型、多起点阻尼 Newton、统一入口
distortion/tracking.py    # 连续分支跟踪（最近根 + 显式步长上限 + 断点/锚点）
tests/test_distortion.py  # 独立参考 + 20 个测试
tests/test_tracking.py    # 跟踪测试：唯一分支/折返/同距歧义/错锚点/中断续跟；
                          # 径向一维截面用独立 Decimal 高精度分段找根核对
demo.py                   # 固定输入演示
README.md
```

测试中的期望值全部独立得到（手算解析例、独立临界点二次方程、稠密
符号变化扫描、中心有限差分 Jacobian）；被测核心不用于生成期望值，
distort↔undistort 往返仅作补充断言。
跟踪测试中被选根对照的"一维径向真根集合"由测试文件内独立的
decimal_roots_g_eq（标准库 decimal，50 位有效数字、2 万点扫描 +
180 次二分，按区间斜率选择收敛分支）给出，不调用 distortion.radial。
