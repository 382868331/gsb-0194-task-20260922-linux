# distortion — 透镜畸变映射的可逆区域检查

在归一化平面上对四系数多项式透镜畸变模型做**前向映射**与**有界盒逆解**，
并在纯径向下给出**分段单调性唯一性/无解证书**。仅使用 Python 3.14
标准库，离线可导入，无第三方依赖，不调用任何同类第三方实现替代核心算法。

## 环境与运行

- Python：**3.14.7**（仅用标准库 `math`；3.10+ 亦可运行，但以 3.14.7 为准）
- 依赖：无。安装 Python 后无需 `pip install` 任何包。
- 网络 / 账号 / 硬件 / 数据库 / 中间件 / Docker：均不需要。

准确命令（在仓库根目录执行）：

```bash
python -m unittest discover -s tests -v   # 全部测试（基线回归 + 前向重采样，约 3 秒）
python demo.py                            # 逆解演示（实测 < 1 秒）
python demo_iteration.py                  # 前向重采样演示（实测 < 1 秒，预算 90 秒）
```

如使用隔离环境：python3 -m venv .venv && . .venv/bin/activate 即可；无第三方依赖，激活后上述命令不变。

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
from distortion import DistortionModel, Candidate, InverseResult, finite_real

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

## 前向面积重采样（小灰度图）

forward_resample 把源像素视为**有面积的常值单元**，经畸变映射前向分配到目标格子；每个目标格输出**收到的能量**与**覆盖面积**两个独立网格——不做点采样，也不做事后归一因子。

调用：forward_resample(model, pixels, src_extent, dst_extent, dst_shape, tol=1e-6, max_depth=12, max_quads=200000)。pixels 为行列表（首行在上，<= 64x64，有限实数灰度）；src_extent/dst_extent 为 (x0, x1, y0, y1)；dst_shape 为 (行, 列)，各 <= 512。

ForwardResult 字段：energy / area（目标网格，分开输出）、lost_energy / lost_area（映射到目标窗外的部分，**不**重新分配）、total_energy（源侧 值x像素面积 之和）、subdivision_error（所有叶子实际接受的最大边界偏差）、budget_exhausted、quads_used、certificate。

### 方法

- 每个源像素是源平面上的矩形常值单元；四角经 distort 映射。
- **自适应四叉细分**：比较边中点/中心的真实映射与角点双线性插值的偏差，超过 tol 则四分；该偏差即弯曲像素边界的近似误差指标。
- 每个叶子四边形用 **Sutherland-Hodgman 精确多边形裁剪**对目标窗和每个重叠目标格求交面积；能量 = 值 x 精确交面积。恒等/仿射（全零系数）退化情形无需细分，结果与精确多边形裁剪参考逐格一致（测试中有独立重写的裁剪器核对）。
- **适用范围**：只在调用方已验证的**单值区域**工作（映射不折叠）；折叠区域上有符号面积会相互抵消，本接口不检测也不声称处理该情形。
- **预算语义**：max_quads 或 max_depth 耗尽时照常返回部分结果并置 budget_exhausted=True，此时**不声称**细分误差已满足 tol；certificate 文本明确说明。
- **拒绝语义**：源像素面积 < 1e-24（近零退化）抛 ValueError；NaN/Inf、bool、ragged 行、空图、非法 extent/shape 均按既有错误语义拒绝；全部校验先于任何计算。

### 误差口径（前向）

- 叶子内部：交面积精确到裁剪器舍入（约 1e-15 量级）。
- 弯曲边界：由 subdivision_error 如实报告，收敛时 <= tol；能量账 sum(energy) + lost_energy 收敛于独立稠密积分 值x|det J|（测试用独立有限差分 Jacobian 的稠密中点积分核对到 2e-3 以内，实测约 2e-5）。
- 恒等映射下 sum(energy) + lost_energy == total_energy 到 1e-12。

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
  `certificate` 文本与 README 均显式声明。

## 浮点结果的误差口径

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
8. 前向重采样假定单值（无折叠）区域；折叠区域的面积账不正确且不报警。目标窗与源区域均为轴对齐矩形；不依赖任何图像库或界面。

## 仓库结构

```
distortion/__init__.py    # 公开 API
distortion/numeric.py     # 有限实数/整数校验
distortion/model.py       # 前向模型、解析 Jacobian、批量接口
distortion/radial.py      # 纯径向分段单调性证书求解
distortion/inverse.py     # 结果类型、多起点阻尼 Newton、统一入口
distortion/clip.py        # Sutherland-Hodgman 裁剪与 shoelace 面积
distortion/forward.py     # 前向面积重采样（自适应细分 + 精确裁剪分配）
tests/test_distortion.py  # 基线回归：独立参考 + 20 个测试
tests/test_forward.py     # 前向重采样：独立裁剪/稠密积分参考 + 13 个测试
demo.py                   # 逆解固定输入演示
demo_iteration.py         # 前向重采样离线演示（< 90 秒）
README.md
```

测试中的期望值全部独立得到（手算解析例、独立临界点二次方程、稠密
符号变化扫描、中心有限差分 Jacobian）；被测核心不用于生成期望值，
distort↔undistort 往返仅作补充断言。
