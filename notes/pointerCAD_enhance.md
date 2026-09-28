# 技术报告：基于 CadQuery 语义增强的高级命令序列表示法（X-CSR）拓展方案

## 摘要

本报告针对 Pointer-CAD 的命令序列表示法（Command Sequence Representation, CSR）在表达高级参数化建模、逻辑选择、高级拓扑特征以及多组件装配体（Assembly）上的局限性，提出了一种全方位的拓展范式——**X-CSR（Extended Command Sequence Representation）**。本方案深度融合了开源参数化脚本框架 **CadQuery** 的核心语义结构（如流式选择器、工作面双栈、声明式约束求解、动态标签控制等），并仿照原论文规范，构建了一套专为大语言模型（LLM）自回归生成设计的离散符号元组与语法骨架，最终实现了从单实体网格逼近到工业级参数化装配体生成的跨越。

---

## 一、 基于规则/逻辑与 Tag 标签的选择器拓展

### 1. 拓展目标

传统的 CSR 强依赖于复杂的图神经网络（GNN）端到端输出 128 维的具象几何实体指针（Pointer），面对未生成的几何实体或需要批量操作的特征（例如“选择所有垂直边”）时，泛化能力极差。
本模块引入 CadQuery 的**流式字符串选择器（String Selectors）**与**Tag 标签持久化机制**。LLM 仅需输出抽象的规则文本符号，由后端的拓扑内核实时匹配实体，并允许为特定实体集合“打标签（Tag）”，以便在后续步骤中跨生命周期引用。

### 2. 新增 Token Notation 定义

| Token | 类别 | 语义描述（Semantic Description） |
| --- | --- | --- |
| `<sel>` | 结构边界 | 逻辑选择器块的开始（Start of Selector Block） |
| `</sel>` | 结构边界 | 逻辑选择器块的结束（End of Selector Block） |
| `<st>` | 属性标志 | 选择器作用实体类型（Selector Type: `faces`, `edges`, `vertices`） |
| `<ss_str>` | 离散文本 | 核心匹配规则字符串（如 `>Z`, `|Z`, `and (not #1)`） |
| `<tag_s>` | 属性标志 | 为当前选定实体注入唯一的识别字符串（Set Tag Name） |
| `<tag_g>` | 属性标志 | 激活并提取历史上下文中已命名 Tag 的实体（Get Tag Name） |

### 3. 操作序列范式（Operation Sequence Paradigm）

```text
<sel> <st> [faces|edges|vertices] </st> <ss_str> 规则表达式 </ss_str> <tag_s> 标签名 </tag_s> </sel>

<sel> <tag_g> 标签名 </tag_g> </sel>

```

### 4. CadQuery 演示代码对照

```python
# CSR 转换为对应的 CadQuery 链式调用
# 注意：先对面操作再对边操作，避免 .edges() 修改 objects 上下文导致后续 .faces() 选空
result = (cq.Workplane("XY")
          .box(10, 10, 10)
          .faces(">Z").tag("top_face")        # 先选定最高面打上标签
          .workplane()                         # 重置 objects 回到完整实体
          .edges("|Z").tag("vertical_edges"))  # 对应范式 A：选定垂直边并打上标签

```

---

## 二、 动态工作平面与局部坐标系变换拓展

### 1. 拓展目标

原 CSR 的草图平面必须静态绑定于世界坐标系的基础平面（如 XY, YZ）或已生成的物理面上。若要在空间中创建悬空偏置面、或进行非正交旋转贴附，原语法无法承载。
本模块引入 CadQuery 的 `Workplane` 栈操作与坐标系变换矩阵 `transformed()`，支持动态的工作平面（Workplane）推入、弹出、相对位移与旋转。

### 2. 新增 Token Notation 定义

| Token | 类别 | 语义描述（Semantic Description） |
| --- | --- | --- |
| `<wp_trans>` | 操作启动 | 启动局部坐标系变换（Start of Workplane Transformation） |
| `</wp_trans>` | 结构边界 | 结束局部坐标系变换（End of Workplane Transformation） |
| `<off>` | 参数标志 | 动态平移偏置向量标志（Offset Vector Flag: dx, dy, dz） |
| `<rot>` | 参数标志 | 动态旋转欧拉角标志（Rotation Angles Flag: rx, ry, rz） |
| `<wp_push>` | 操作指令 | 将当前激活的工作平面压入历史堆栈（Push current workplane to stack） |
| `<wp_pop>` | 操作指令 | 弹出并激活堆栈顶部的上一个工作平面（Pop workplane from stack） |

### 3. 操作序列范式（Operation Sequence Paradigm）

```text
<wp_trans> 
  <off> <v>dx</v> <v>dy</v> <v>dz</v> </off> 
  <rot> <v>rx</v> <v>ry</v> <v>rz</v> </rot> 
</wp_trans>

```

### 4. CadQuery 演示代码对照

```python
# transformed() 创建的局部工作平面在 extrude 后自动复位，无需 push/pop
# CadQuery 没有公开的 push() / pop() 方法 — 原文档中这两个调用是错误的
result = (cq.Workplane("XY")
          .box(10, 10, 10)
          .faces(">Z").workplane()                                    # 在顶面建立工作平面
          .transformed(offset=cq.Vector(0, 0, 15), rotate=cq.Vector(0, 45, 0))
          .circle(2).extrude(5))

```

---

## 三、 参数化编程逻辑与流程控制拓展

### 1. 拓展目标

目前的 CSR 序列是展开后的单体静态几何快照，无法表达高级参数化 CAD 中至关重要的“变量级联修改（Variable Propagation）”和“阵列循环（Loops）”。
本拓展允许在 Token 流中就地声明**数字变量（Variables）**与**循环控制块（Loops）**，使大模型生成的序列本身具备“程序性”。

### 2. 新增 Token Notation 定义

| Token | 类别 | 语义描述（Semantic Description） |
| --- | --- | --- |
| `<def_var>` | 声明逻辑 | 定义一个全局数字参数（Define variable with name attribute） |
| `<var>` | 引用逻辑 | 引用已定义的全局参数（Reference an existing variable） |
| `<for>` | 控制流边界 | 循环结构块开始，带 `iter`（变量名）, `count`（循环次数）属性 |
| `</for>` | 控制流边界 | 循环结构块结束（End of For-loop block） |
| `<idx>` | 引用逻辑 | 获取当前循环体的索引值（Get current loop iteration index） |

### 3. 操作序列范式（Operation Sequence Paradigm）

```text
<def_var name="H"> <v>20</v> </def_var>
<def_var name="R"> <v>2</v> </def_var>

<for iter="i" count="4">
  <wp_trans> <off> <var name="R"/>*<idx iter="i"/> <v>0</v> <v>0</v> </off> </wp_trans>
  <ss> <sx> <circle> <var name="R"/> </circle> </sx> </ss>
  <se> <var name="H"/> </se>
</for>

```

### 4. CadQuery 演示代码对照

```python
# 对应上述参数化与循环的编译层解释器逻辑
# pushPoints() 声明式多点阵列 — CadQuery fluent API 惯用模式，无需显式循环
H = 20
R = 2
result = (cq.Workplane("XY")
          .box(50, 50, 2)
          .faces(">Z").workplane()                       # 在顶面建工作平面
          .pushPoints([(R * i, 0) for i in range(4)])    # 一次性推入全部阵列点
          .circle(R).extrude(H))                         # 每个点处拉伸圆柱

```

---

## 四、 高级拓扑与特殊几何特征拓展

### 1. 拓展目标

原论文的 CSR 仅包含草图、直拉伸、基本倒角等。为匹配高级工业设计，必须拓展**放样（Loft）**、扫掠（Sweep）**和**抽壳（Shell）等非线性拓扑特征。

### 2. 新增 Token Notation 定义

| Token | 类别 | 语义描述（Semantic Description） |
| --- | --- | --- |
| `<sh>` | 操作启动 | 启动实体抽壳操作（Start of Shelling operation） |
| `<loft>` | 操作启动 | 将多个封闭草图界面融合成放样实体（Start of Loft operation） |
| `<sw>` | 操作启动 | 启动扫掠操作，需要轮廓和路径（Start of Sweep operation） |
| `<path>` | 结构边界 | 声明扫掠轨迹线图元序列（Start of Sweep Path specification） |
| `<prof>` | 结构边界 | 声明扫掠截面轮廓序列（Start of Sweep Profile specification） |

### 3. 操作序列范式（Operation Sequence Paradigm）

```text
<sh> <sel> <st> faces </st> <ss_str> >Z </ss_str> </sel> <v>-1.5</v> <es>

<sw>
  <prof> <ss> ... </ss> </prof>  <path> <ss> ... </ss> </path>  </sw> <es>

```

### 4. CadQuery 演示代码对照

```python
# 对应高级拓扑算子的 CadQuery 转换
# 1. 抽壳 — shell() 接受正壁厚（去除选定面，向内抽壳）
shell_box = (cq.Workplane("XY")
             .box(10, 10, 10)
             .faces(">Z").shell(1.5))

# 2. 扫掠 — 路径在 XZ 平面构建三点圆弧，截面在 YZ 平面
result = (cq.Workplane("YZ")
          .circle(2)
          .sweep(cq.Workplane("XZ")
                 .moveTo(0, 0)
                 .threePointArc((10, 5), (20, 0))))  # 2D 坐标 (x, z)

```

---

## 五、 Assembly（装配体）复用和约束拓扑拓展

### 1. 拓展目标

这是 X-CSR 最核心的宏观设计。原系统不支持多零件协同。本模块抽象出独立的**组件作用域（Part Scope）**，并通过指代 Mate（装配基准点）和声明 Constraint（约束算子），将自回归序列推向多组件系统。

### 2. 新增 Token Notation 定义

| Token | 类别 | 语义描述（Semantic Description） |
| --- | --- | --- |
| `<part>` | 结构边界 | 声明一个独立可复用零件的开始（Start of distinct Part definition） |
| `</part>` | 结构边界 | 结束零件定义，送入后台组件缓存区（End of Part definition） |
| `<inst>` | 操作指令 | 实例化一个已有零件，带 `part_id` 和唯一 `inst_id` 属性 |
| `<def_mate>` | 声明逻辑 | 在特定拓扑实体（面/边）上定义装配基准原点（Define Assembly Mate） |
| `<asm_c>` | 声明逻辑 | 声明两个实例 Mate 之间的运动学/几何约束（Apply Assembly Constraint） |
| `<c_type>` | 属性标志 | 约束类型标志（`Plane` 共面, `Axis` 共轴, `Point` 点重合） |

### 3. 操作序列范式（Operation Sequence Paradigm）

```text
<part id="bolt">
  <ss> ... </ss> <se> <v>20</v> </se>
  <def_mate id="bolt_shoulder"> <sel> <st> faces </st> <ss_str> <v>0</v>,<v>0</v>,<v>0</v> </ss_str> </sel> </def_mate>
</part>

<inst part_id="bolt" inst_id="b_left"/>
<asm_c>
  <c_type> Axis </c_type>
  <v> b_left.bolt_shoulder </v> <v> base_plate.hole_1 </v>
</asm_c>

```

### 4. CadQuery 演示代码对照

```python
# 转换为 CadQuery Assembly 树结构与约束求解器代码
asm = cq.Assembly(name="global_system")

# 从零件库加载 B-Rep 并例化
asm.add(bolt_geometry, name="b_left")
asm.add(base_plate_geometry, name="base_plate")

# 绑定 Mate — 通过 tag 精确引用关键面/边，避免几何启发式选择器失效
asm.mate("b_left?faces@<Z", name="bolt_shoulder")
asm.mate("base_plate?faces@>Z", name="hole_1")

# constrain() 标准签名为 3 参数 (q1, q2, kind)，零件名与 mate 名合并于查询字符串中
asm.constrain("b_left?bolt_shoulder", "base_plate?hole_1", "Axis")
asm.solve()

```

---

## 六、 综合原型案例合成（Comprehensive Prototype Synthesis）

为了直观验证 **X-CSR** 模型的完备性，我们构建一个复杂的工业零部件混合案例：
**“参数化定义一个带标签的带孔底座零件，在偏置倾斜平面上建立工作区，执行抽壳拓扑，最后在装配体中引入外部轴承组件通过‘共轴’和‘面贴合’完成高精度声明式约束装配”。**

### 1. 完整的 X-CSR 拓展符号序列（LLM 预期的生成输出）

```xml
<X-CSR-STREAM>
  <def_var name="length"> <v>100</v> </def_var>
  <def_var name="thick"> <v>15</v> </def_var>
  
  <part id="base_plate">
    <ss> 
      <sx> <rect> <var name="length"/> <var name="length"/> </rect> </sx> 
    </ss> 
    <se> <var name="thick"/> </se>
    
    <sel> <st> faces </st> <ss_str> >Z </ss_str> <tag_s> main_top_surface </tag_s> </sel>
    <sf> <sel> <st> edges </st> <ss_str> |Z </ss_str> </sel> <v>10</v> <es>
    
    <sel> <tag_g> main_top_surface </tag_g> </sel>
    <ss> <sx> <circle> <v>20</v> </circle> </sx> </ss>
    <shole> <var name="thick"/> </shole> <es>
    
    <def_mate id="center_bearing_seat"> 
      <sel> <st> edges </st> <ss_str> #通孔内壁内边缘 </ss_str> </sel> 
    </def_mate>
  </part>

  <part id="standard_bearing">
    <ss> <sx> <circle> <v>20</v> </circle> <circle> <v>10</v> </circle> </sx> </ss>
    <se> <v>12</v> </se>
    <def_mate id="bearing_axis"> <sel> <st> edges </st> <ss_str> >Z </ss_str> </sel> </def_mate>
    <def_mate id="bearing_face"> <sel> <st> faces </st> <ss_str> <v>0</v>,<v>0</v>,<v>0</v> </ss_str> </sel> </def_mate>
  </part>

  <inst part_id="base_plate" inst_id="base_inst_1"/>
  <inst part_id="standard_bearing" inst_id="bearing_inst_1"/>
  
  <asm_c>
    <c_type> Axis </c_type>
    <v> bearing_inst_1.bearing_axis </v>
    <v> base_inst_1.center_bearing_seat </v>
  </asm_c>
  <asm_c>
    <c_type> Plane </c_type>
    <v> bearing_inst_1.bearing_face </v>
    <v> base_inst_1.main_top_surface </v>
  </asm_c>

  <em> </X-CSR-STREAM>

```

### 2. 对应的规范 CadQuery 编译解释代码

当后端的翻译架构解析了上述 X-CSR 序列后，将自动解包、代入参数变量、并静态编译为以下标准的 Python 代码交付内核渲染：

```python
import cadquery as cq

# ----------------------------------------------------------------
# STAGE 1: 编译零件 - base_plate
# ----------------------------------------------------------------
length = 100
thick = 15

# 利用 CadQuery 链式调用无损翻译规则表达与标签持久化
# 关键修复：先对整个实体选边倒圆角，再聚焦面操作 — 避免 .faces() 后 .edges("|Z") 选空
base_plate_geom = (
    cq.Workplane("XY")
    .box(length, length, thick, centered=(True, True, False))
    .edges("|Z").fillet(10)                         # 先对整个实体批量选边倒圆角
    .faces(">Z").tag("main_top_surface")             # 再选定顶面打标签 <tag_s>
    .workplaneFromTagged("main_top_surface")         # 通过标签恢复工作平面 <tag_g>
    .circle(20)
    .cutThruAll()                                    # 通孔 — cutThruAll() 而非 hole()
)

# ----------------------------------------------------------------
# STAGE 2: 编译零件 - standard_bearing
# ----------------------------------------------------------------
# 用 tag 标记关键面，供装配 mate 精确引用，替代易失效的几何启发式选择器
standard_bearing_geom = (
    cq.Workplane("XY")
    .circle(20).circle(10)
    .extrude(12)
    .faces(">Z").tag("bearing_top_face")
    .faces("<Z").tag("bearing_bottom_face")
    .faces("|Z").tag("bearing_wall")                  # 柱面 — 用于 Axis 约束
)

# ----------------------------------------------------------------
# STAGE 3: 装配体树形构建与代数求解（约束引擎解算）
# ----------------------------------------------------------------
# 初始化全局装配容器
system_assembly = cq.Assembly(name="parametric_final_system")

# 注入组件实例，分配独立的拓扑命名空间
system_assembly.add(base_plate_geom, name="base_inst_1")
system_assembly.add(standard_bearing_geom, name="bearing_inst_1")

# 在实例层面上挂载装配基准面(Mates) — 优先使用 tag 引用
system_assembly.mate("base_inst_1?main_top_surface", name="mate_base_top")
system_assembly.mate("bearing_inst_1?bearing_wall", name="mate_bearing_axis")
system_assembly.mate("bearing_inst_1?bearing_bottom_face", name="mate_bearing_face")

# constrain() 标准签名为 3 参数 (q1, q2, kind)
system_assembly.constrain("bearing_inst_1?mate_bearing_axis",
                          "base_inst_1?mate_base_top", "Axis")
system_assembly.constrain("bearing_inst_1?mate_bearing_face",
                          "base_inst_1?mate_base_top", "Plane")

# 驱动内核的约束求解器处理装配矩阵的最优化
system_assembly.solve()

# 导出最终生成的 B-Rep 工业级标准格式
# system_assembly.save("output_assembly.step")

```

---

## 结论

通过 X-CSR 拓展方案，大语言模型摆脱了“必须具备连续空间坐标精细感知能力”的紧箍咒。它不需要预测绝对装配矩阵或长距离微观边指针，而只需要以**离散化的抽象代码逻辑（即选择器字符串、变量公式、装配约束）**进行生成。复杂而高精度的底层边界连续计算任务被全权移交回了系统的后置 CadQuery 编译层，从根本上杜绝了生成式 CAD 中最顽固的**拓扑撕裂错误**，为大模型赋能真实工业复杂建模提供了崭新的基石。