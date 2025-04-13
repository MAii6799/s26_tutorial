# 视觉第三课（第二部分）：四元数等😉

author：[Cinjay Jiang](https://github.com/knot41)

date：2025/4/13

---

## 一、什么是四元数？🤔

在上一部分，我们讲了旋转矩阵是如何表示**姿态（朝向）**的。它很直观，也很好用——但有些**小问题**，就比如旋转矩阵是 $3\times3$，有 9 个元素，然而朝向感觉只需要 3 个自由度。四元数能够解决旋转矩阵用9个元素表示三个自由度的冗余情况。

### 基本性质

四元数是一种扩展了复数的数学概念。四元数的定义如下：

$$
q = w + xi + yj + zk
$$

其中 $w, x, y, z$ 是实数，$i, j, k$ 是虚数单位，满足如下关系：

$$
i^2 = j^2 = k^2 = ijk = -1
$$

四元数的加法和乘法定义如下：

$$
q_1 + q_2 = (w_1 + w_2) + (x_1 + x_2)i + (y_1 + y_2)j + (z_1 + z_2)k
$$

$$
q_1 \cdot q_2 = (w_1w_2 - x_1x_2 - y_1y_2 - z_1z_2) + (w_1x_2 + x_1w_2 + y_1z_2 - z_1y_2)i + (w_1y_2 - x_1z_2 + y_1w_2 + z_1x_2)j + (w_1z_2 + x_1y_2 - y_1x_2 + z_1w_2)k
$$

四元数的模定义如下：

$$
|q| = \sqrt{w^2 + x^2 + y^2 + z^2}
$$

四元数的共轭定义如下：

$$
q^* = w - xi - yj - zk
$$

四元数的逆定义如下：

$$
q^{-1} = \frac{q^*}{|q|^2}
$$


### 四元数表示旋转

一个单位四元数（模长为1）可以表示**绕任意轴旋转任意角度**。

模长不为1的话，旋转后的结果向量会被**缩放**。

如果要绕单位轴 $\hat{\mathbf{n}} = [n_x, n_y, n_z]^T$ 旋转角度 $\theta$，那么对应的四元数是：

$$
\mathbf{q} = \begin{bmatrix}
\cos(\theta/2) \\
n_x \cdot \sin(\theta/2) \\
n_y \cdot \sin(\theta/2) \\
n_z \cdot \sin(\theta/2)
\end{bmatrix}
$$

> 注：是 $\theta/2$，不是 $\theta$ ，别搞错了！

### 四元数的使用

四元数的主要用途就是：**旋转点、表示姿态**等。

#### 旋转一个向量 $\mathbf{v}$：

1. 把向量 $\mathbf{v}$ 变成四元数形式：$\mathbf{v}_q = [0, v_x, v_y, v_z]$
2. 用四元数旋转它：

$$
\mathbf{v}' = \mathbf{q} \cdot \mathbf{v}_q \cdot \mathbf{q}^{-1}
$$

其中 $\mathbf{q}^{-1}$ 是四元数的共轭（因为它是单位四元数）：

$$
\mathbf{q}^{-1} = \begin{bmatrix} w \\ -x \\ -y \\ -z \end{bmatrix}
$$

### 四元数和旋转矩阵的相互转化

我们可以把四元数转成旋转矩阵，或者反过来。

#### 四元数 → 旋转矩阵：

设 $\mathbf{q} = [w, x, y, z]^T$，则对应的旋转矩阵为：

$$
\mathbf{R} = \begin{bmatrix}
1 - 2(y^2 + z^2) & 2(xy - wz) & 2(xz + wy) \\
2(xy + wz) & 1 - 2(x^2 + z^2) & 2(yz - wx) \\
2(xz - wy) & 2(yz + wx) & 1 - 2(x^2 + y^2)
\end{bmatrix}
$$

（虽然有点复杂，但可以交给库来做！😋）

#### 旋转矩阵 → 四元数：

设旋转矩阵为：

$$
\mathbf{R} = \begin{bmatrix}
r_{11} & r_{12} & r_{13} \\
r_{21} & r_{22} & r_{23} \\
r_{31} & r_{32} & r_{33}
\end{bmatrix}
$$

我们先计算一个**迹（Trace）**：

$$
\text{tr} = r_{11} + r_{22} + r_{33}
$$

然后判断：

- 如果 $\text{tr} > 0$：

  $$
  S = \sqrt{\text{tr} + 1.0} \cdot 2 \\
  w = 0.25 \cdot S \\
  x = \frac{r_{32} - r_{23}}{S} \\
  y = \frac{r_{13} - r_{31}}{S} \\
  z = \frac{r_{21} - r_{12}}{S}
  $$

- 否则根据最大对角元素选择计算方式（略复杂，不展开）

---

## 二、什么是欧拉角？🤔

- 我们在生活中描述旋转时常说“先绕X轴转个角度，再绕Y轴，再绕Z轴”，这就是**欧拉角**的基本思想。
- 欧拉角用**三个角度**（通常是 yaw、pitch、roll）来表示一个物体的空间姿态。

### 常见定义方式：
- **Z-Y-X顺序**（航空航天常用）：
  - **Roll**：绕X轴旋转（翻滚）
  - **Pitch**：绕Y轴旋转（俯仰）
  - **Yaw**：绕Z轴旋转（偏航）

Z-Y-X旋转顺序：先 **Yaw** → 再 **Pitch** → 最后 **Roll**

### 欧拉角旋转过程:

用一个三步过程描述：
1. **第一步：绕Z轴旋转 $\gamma$（Yaw）**
2. **第二步：绕Y轴旋转 $\beta$（Pitch）**
3. **第三步：绕X轴旋转 $\alpha$（Roll）**

最终的旋转矩阵为：

$$
R = R_x(\alpha) \cdot R_y(\beta) \cdot R_z(\gamma)
$$

> 注：旋转的实际顺序是从右往左乘！

### 和旋转矩阵的关系

每个轴的基本旋转矩阵为：

- 绕X轴（Roll）：
  $$
  R_x(\alpha) = \begin{bmatrix}
  1 & 0 & 0 \\
  0 & \cos\alpha & -\sin\alpha \\
  0 & \sin\alpha & \cos\alpha
  \end{bmatrix}
  $$

- 绕Y轴（Pitch）：
  $$
  R_y(\beta) = \begin{bmatrix}
  \cos\beta & 0 & \sin\beta \\
  0 & 1 & 0 \\
  -\sin\beta & 0 & \cos\beta
  \end{bmatrix}
  $$

- 绕Z轴（Yaw）：
  $$
  R_z(\gamma) = \begin{bmatrix}
  \cos\gamma & -\sin\gamma & 0 \\
  \sin\gamma & \cos\gamma & 0 \\
  0 & 0 & 1
  \end{bmatrix}
  $$

*小小的提问环节*：枪口坐标系到相机坐标系的转换按照欧拉角的形式应该怎么转呢？

## 三、什么是旋转向量？🤔

旋转向量是一种**简洁表示空间旋转**的方式，它将一个旋转拆解为：

- 一根旋转轴 + 一个旋转角度  
- 并将两者合并成一个**三维向量**

### 定义

旋转向量 $\vec{r}$ 是一个三维向量，定义如下：

$$
\vec{r} = \theta \cdot \hat{u}
$$

- $\theta$：绕轴旋转的角度（弧度）
- $\hat{u}$：单位旋转轴（$\|\hat{u}\| = 1$）
- $\vec{r}$ 的方向是旋转轴，模长是旋转角度


### 和旋转矩阵的转换（Rodrigues 公式）

给定旋转向量 $\vec{r} = \theta \cdot \hat{u}$，对应的旋转矩阵为：

$$
R = I + \sin\theta \cdot [\hat{u}]_\times + (1 - \cos\theta) \cdot [\hat{u}]_\times^2
$$

其中：

- $I$ 是单位矩阵  
- $[\hat{u}]_\times$ 是**旋转轴的叉乘矩阵**（反对称矩阵）：

$$
[\hat{u}]_\times =
\begin{bmatrix}
0 & -u_z & u_y \\
u_z & 0 & -u_x \\
-u_y & u_x & 0
\end{bmatrix}
$$

> 别慌，笔者也不怎么理解，还好opencv有Rodrigues函数 🥹

### 和四元数的转换

从旋转向量 $\vec{r} = \theta \cdot \hat{u}$ 得到四元数：

$$
q = \left[\cos\left(\frac{\theta}{2}\right),\ \hat{u} \cdot \sin\left(\frac{\theta}{2}\right)\right]
$$

这真的好简洁~，要是所有公式都这么**清爽**就好了😭
