# 作业

1. 实现一个四元数类
2. 实现一个位姿类
3. 实现一个坐标转换，要求把给定坐标系下的位姿转换到指定坐标系下的位姿

为统一起见
- 这里的位姿中的姿态全用**欧拉角**的形式。

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

- 点在 **A 坐标系**中的位姿：
  - 平移向量：$\mathbf{t}_{PA}$（点的位置在 A 中的坐标）
  - 姿态四元数：$\mathbf{q}_{PA}$（点的朝向相对于 A）

- **A 相对于 B 坐标系**的位姿：
  - 平移向量：$\mathbf{t}_{BA}$（A 的原点在 B 中的位置）
  - 旋转四元数：$\mathbf{q}_{BA}$（从 A 旋转到 B）

### 🎯 目标

求点在 **B 坐标系**中的位姿：

- $\mathbf{t}_{PB}$：点在 B 中的位置  
- $\mathbf{q}_{PB}$：点相对于 B 的姿态

### 🧠 解法公式

#### 1. 姿态（四元数）变换：

$$
\mathbf{q}_{PB} = \mathbf{q}_{BA} \cdot \mathbf{q}_{PA}
$$

说明：先将点的方向从点坐标系变换到 A，再从 A 变换到 B。

#### 2. 位置（向量）变换：

$$
\mathbf{t}_{PB} = \mathbf{q}_{BA} \cdot \mathbf{t}_{PA} \cdot \mathbf{q}_{BA}^{-1} + \mathbf{t}_{BA}
$$

说明：

- 把 $\mathbf{t}_{PA}$ 视为纯四元数 $(0, x, y, z)$；
- 使用三明治公式进行四元数旋转；
- 最后加上平移向量。

### 🧩 位姿合成的整体表达

将点的位姿看作一个整体变换 $T_{PA} = (\mathbf{q}_{PA}, \mathbf{t}_{PA})$，那么：

$$
T_{PB} = T_{BA} \circ T_{PA}
$$

即：

- 姿态合成：$\mathbf{q}_{PB} = \mathbf{q}_{BA} \cdot \mathbf{q}_{PA}$
- 位置合成：$\mathbf{t}_{PB} = \mathbf{q}_{BA} \cdot \mathbf{t}_{PA} \cdot \mathbf{q}_{BA}^{-1} + \mathbf{t}_{BA}$

---

## 📝 小贴士

- 四元数乘法不满足交换律，注意顺序。
- 单位四元数的逆就是其共轭（即实部不变，虚部取负）。
- 你也可以使用 4x4 齐次矩阵来实现同样的效果，不过四元数旋转在数值上更稳定、更高效。
- 加油！