# 作业

1. 实现一个四元数类
2. 实现一个位姿类
3. 实现一个坐标转换，要求把给定坐标系下的位姿转换到指定坐标系下的位姿

为统一起见
- 这里的位姿中的姿态全用**欧拉角**的形式。
- 坐标系变换使用四元数的方式。

---

## 欧拉角 ↔ 四元数 转换公式

### 一、欧拉角 → 四元数

假设欧拉角为：
- **Roll** ($\phi$)：绕 X 轴旋转
- **Pitch** ($\theta$)：绕 Y 轴旋转
- **Yaw** ($\psi$)：绕 Z 轴旋转

转换为四元数 $q = (w, x, y, z)$ 的公式为：

$$
\begin{aligned}
c_\phi &= \cos\left(\frac{\phi}{2}\right), \quad s_\phi = \sin\left(\frac{\phi}{2}\right) \\
c_\theta &= \cos\left(\frac{\theta}{2}\right), \quad s_\theta = \sin\left(\frac{\theta}{2}\right) \\
c_\psi &= \cos\left(\frac{\psi}{2}\right), \quad s_\psi = \sin\left(\frac{\psi}{2}\right)
\end{aligned}
$$

$$
\begin{aligned}
w &= c_\phi c_\theta c_\psi + s_\phi s_\theta s_\psi \\
x &= s_\phi c_\theta c_\psi - c_\phi s_\theta s_\psi \\
y &= c_\phi s_\theta c_\psi + s_\phi c_\theta s_\psi \\
z &= c_\phi c_\theta s_\psi - s_\phi s_\theta c_\psi
\end{aligned}
$$

> 这是 **ZYX 顺序**（Yaw → Pitch → Roll）对应的转换

### 二、四元数 → 欧拉角

已知四元数 $q = (w, x, y, z)$，求对应欧拉角（Roll, Pitch, Yaw）：

$$
\begin{aligned}
\phi &= \text{Roll} = \arctan2\left(2(w x + y z),\ 1 - 2(x^2 + y^2)\right) \\
\theta &= \text{Pitch} = \arcsin\left(2(w y - z x)\right) \\
\psi &= \text{Yaw} = \arctan2\left(2(w z + x y),\ 1 - 2(y^2 + z^2)\right)
\end{aligned}
$$

⚠️ 注意：
- 若 $|\theta| = 90^\circ$，会遇到**万向锁**（Pitch 处于极值时，Roll 和 Yaw 混淆）。

---

## 坐标变换：带姿态（四元数）的点从 A 坐标系变换到 B 坐标系

### ✅ 已知

- 点 p 在 **A 坐标系**中的位姿：
  - 平移向量：$\mathbf{t}_{PA}$（点 p 的位置在 A 中的坐标）
  - 姿态四元数：$\mathbf{q}_{PA}$（点 p 的朝向相对于 A）

- **A 相对于 B 坐标系**的位姿：
  - 平移向量：$\mathbf{t}_{BA}$（A 的原点在 B 中的位置）
  - 旋转四元数：$\mathbf{q}_{BA}$（从 A 旋转到 B，也就是A 相对于 B 的姿态）

### 🎯 目标

求点在 **B 坐标系**中的位姿：

- $\mathbf{t}_{PB}$：点 p 在 B 中的位置  
- $\mathbf{q}_{PB}$：点 p 相对于 B 的姿态

### 🧠 解法公式

#### 1. 姿态（四元数）变换：

$$
\mathbf{q}_{PB} = \mathbf{q}_{BA} \cdot \mathbf{q}_{PA}
$$

说明：先将点的方向从点 p 坐标系变换到 A，再从 A 变换到 B。

#### 2. 位置（向量）变换：

$$
\mathbf{t}_{PB} = \mathbf{q}_{BA} \cdot \mathbf{t}_{PA} \cdot \mathbf{q}_{BA}^{-1} + \mathbf{t}_{BA}
$$

说明：

- 把 $\mathbf{t}_{PA}$ 视为纯四元数 $(0, x, y, z)$；
- 使用三明治公式进行四元数旋转；
- 最后加上平移向量。


#### 3. 扩展内容

现假设我知道点 p 在 A 坐标系中的位姿，但此时不知道 A 坐标系相对于 B 坐标系的位姿。知道的是 B 坐标系相对于 A 坐标系的位姿，那么此时就需要进行逆变换。

✅ **公式（逆变换）**：

1. **四元数取共轭（也就是逆）**：

$$
\mathbf{q}_{A}^{B} = \mathbf{q}_{B}^{A^{-1}} = \mathbf{q}_{B}^{A*}
$$

其中 * 表示四元数共轭。

2. **平移项的逆**：

$$
\mathbf{t}_{A}^{B} = -\mathbf{q}_{A}^{B} \cdot \mathbf{t}_{B}^{A}
$$

意思是：先把 B 原点在 A 中的位置  $\mathbf{t}_{B}^{A}$ 用反方向旋转回 B 坐标系，然后取负。


---

## 📝 小贴士

- 一定要弄清楚是 谁相对于谁，谁在谁下。例如：A 相对于 B等价于在 A 在 B 下。
- 四元数乘法不满足交换律，注意顺序。
- 单位四元数的逆就是其共轭（即实部不变，虚部取负）。
- 你也可以使用 4x4 齐次矩阵来实现同样的变换效果，不过最好还是统一用四元数实现变换，有余力再用用齐次变换矩阵。
- 加油加油再加油！  