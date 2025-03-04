---
title: "3D Rigid Motion"
categories: tech
tags: [Aerial Robotics]
use_math: true
---

三维空间刚体运动，参考自《视觉SLAM十四讲》

## 1. 旋转矩阵 Rotation Matrix

### 1.1 点，向量和坐标系

首先对点进行定义，点是空间中的一个位置，是一个基本元素，没有方向，没有大小，没有体积。

向量可以看成点指向点的连线，有方向，有大小。

坐标系的作用在，只有当坐标系指定后，谈论点的位置和向量的位置才有意义。 

三维空间中某个点的坐标可以用 $\mathbb{R}^3$ 来描述。假设在线性空间内，找到了该空间的一组基 $(\boldsymbol{e}_1, \boldsymbol{e}_2, \boldsymbol{e}_3)$ ，那么任意向量 $\boldsymbol{a}$ 在这组基下就有一个坐标：

$$
\boldsymbol{a} = [\boldsymbol{e}_1, \boldsymbol{e}_2, \boldsymbol{e}_3] 
\begin{bmatrix}
a_1 \\
a_2 \\
a_3
\end{bmatrix} \tag{1}
$$

这里 $(a_1, a_2, a_3)$ 就是 $\boldsymbol{a}$ 在基 $(\boldsymbol{e}_1, \boldsymbol{e}_2, \boldsymbol{e}_3)$ 下的坐标。

向量内积可以描述为：

$$
\boldsymbol{a} \cdot \boldsymbol{b} = \boldsymbol{a}^T \boldsymbol{b} = \sum_{i=1}^{3} a_i b_i \tag{2}
$$

**外积与反对称矩阵**

向量外积的结果是垂直于两个向量的向量，表示为：

$$
\boldsymbol{a} \times \boldsymbol{b} =
\begin{bmatrix}
a_2 b_3 - a_3 b_2 \\
a_3 b_1 - a_1 b_3 \\
a_1 b_2 - a_2 b_1
\end{bmatrix}=
\begin{bmatrix}
0 & -a_3 & a_2 \\
a_3 & 0 & -a_1 \\
-a_2 & a_1 & 0
\end{bmatrix} \boldsymbol{b}
=
\boldsymbol{a}^{\wedge} \boldsymbol{b} \tag{3}
$$

这里 $\boldsymbol{a}^{\wedge}$ 就是 $\boldsymbol{a}$ 的反对称矩阵。向量外积的大小为 $\| \boldsymbol{a} \times \boldsymbol{b} \| = \| \boldsymbol{a} \| \| \boldsymbol{b} \| \sin \theta$ ，其中 $\theta$ 是 $\boldsymbol{a}$ 和 $\boldsymbol{b}$ 之间的夹角。

### 1.2 坐标系间的欧氏变换

欧氏变换由平移和旋转组成。

对于一个向量 $\boldsymbol{a}$ ， 它有两个坐标系下的坐标为 $[a_1, a_2, a_3]^T$ 和 $[b_1, b_2, b_3]^T$ ，那么这两个坐标之间的关系可以表示为：

$$
[\boldsymbol{e}_1, \boldsymbol{e}_2, \boldsymbol{e}_3]
\begin{bmatrix}
a_1 \\
a_2 \\
a_3
\end{bmatrix} = 
[\boldsymbol{e}_1', \boldsymbol{e}_2', \boldsymbol{e}_3']
\begin{bmatrix}
b_1 \\
b_2 \\
b_3
\end{bmatrix} \tag{4}
$$

进而有：

$$
\begin{bmatrix}
a_1 \\
a_2 \\
a_3
\end{bmatrix} = 

\begin{bmatrix}
\boldsymbol{e}_1^T \boldsymbol{e}_1' & \boldsymbol{e}_1^T \boldsymbol{e}_2' & \boldsymbol{e}_1^T \boldsymbol{e}_3' \\
\boldsymbol{e}_2^T \boldsymbol{e}_1' & \boldsymbol{e}_2^T \boldsymbol{e}_2' & \boldsymbol{e}_2^T \boldsymbol{e}_3' \\
\boldsymbol{e}_3^T \boldsymbol{e}_1' & \boldsymbol{e}_3^T \boldsymbol{e}_2' & \boldsymbol{e}_3^T \boldsymbol{e}_3'
\end{bmatrix}

\begin{bmatrix}
b_1 \\
b_2 \\
b_3
\end{bmatrix}

=
\boldsymbol{R}_b^a \boldsymbol{b} \tag{5}
$$

可以看出旋转矩阵由两组基之间的内积组成，并满足一些性质，比如是正交矩阵，行列式为1。

可以将 $n$ 维旋转矩阵的集合用李群定义如下：

$$
SO(n) = \{ \boldsymbol{R} \in \mathbb{R}^{n \times n} | \boldsymbol{R}^T \boldsymbol{R} = \boldsymbol{I}, \text{det}(\boldsymbol{R}) = 1 \} \tag{6}
$$

其中， $SO(n)$ 是 $n$ 维特殊正交群 (Special Orthogonal Group)。特别地， $SO(3)$ 就指三维空间的旋转。

由于旋转矩阵的性质，可以看出旋转矩阵的逆矩阵就是它的转置矩阵，即

$$
\boldsymbol{R}^T \boldsymbol{R} = \boldsymbol{I} \Rightarrow \boldsymbol{R}^T = \boldsymbol{R}^{-1} \tag{7}
$$

因此若已知坐标系 $a$ 到 $b$ 的旋转矩阵 $\boldsymbol{R}_b^a$ ，那么坐标系 $b$ 到 $a$ 的旋转矩阵 $\boldsymbol{R}_a^b$ 就是 $\boldsymbol{R}_b^a$ 的逆矩阵和转置矩阵。

则完整的欧氏变换可以写为：

$$
\boldsymbol{a} = \boldsymbol{R}_b^a \boldsymbol{b} + \boldsymbol{t}_b^a \tag{8}
$$

其中 $\boldsymbol{t}_b^a$ 是坐标系 $b$ 到 $a$ 的平移向量。它指的是坐标系 $a$ 的原点指向坐标系 $b$ 的原点的向量。

### 1.3 变换矩阵与齐次坐标

通过引入齐次坐标和变换矩阵 $\boldsymbol{T}$ (Transformation Matrix)，可以将平移和旋转统一起来。有利于多次变换的组合。比如从坐标系 $b$ 到 $a$ 的变换可以表示为：

$$
\begin{bmatrix}
\boldsymbol{a} \\
1
\end{bmatrix} =
\begin{bmatrix}
\boldsymbol{R}_b^a & \boldsymbol{t}_b^a \\
0 & 1
\end{bmatrix}
\begin{bmatrix}
\boldsymbol{b} \\
1
\end{bmatrix} = 
\boldsymbol{T}_b^a
\begin{bmatrix}
\boldsymbol{b} \\
1
\end{bmatrix} \tag{9}
$$

这种矩阵在李群中称为特殊欧氏群 （Special Euclidean Group） $SE(3)$ ，定义如下：

$$
SE(3) = \left\{ \boldsymbol{T} =
\begin{bmatrix}
\boldsymbol{R} & \boldsymbol{t} \\
0 & 1
\end{bmatrix} \Bigg| \boldsymbol{R} \in SO(3), \, \boldsymbol{t} \in \mathbb{R}^3 \right\} \tag{10}
$$

与 $SO(3)$ 类似，该矩阵的逆表示一个反向的变换，即：

$$
\boldsymbol{T}^{-1} =
\begin{bmatrix}
\boldsymbol{R}^T & -\boldsymbol{R}^T \boldsymbol{t} \\
0 & 1
\end{bmatrix} \tag{11}
$$


## 2. 旋转向量和欧拉角 Rotation Vector and Euler Angle

旋转矩阵能够表示旋转，但是不够紧凑，并且有正交和行列式为1的约束。因此有了其他的表示方法。

### 2.1 旋转向量

任意旋转都可用一个旋转轴+旋转角来刻画。

假设一个用旋转矩阵表示的旋转 $\boldsymbol{R}$ 。如果用旋转向量来描述，则假设旋转轴为一个单位长度的向量 $\boldsymbol{n}$ ，旋转角为 $\theta$ ，则旋转矩阵可以表示为：

$$
\boldsymbol{R} = \cos \theta \boldsymbol{I} + (1 - \cos \theta) \boldsymbol{n} \boldsymbol{n}^T + \sin \theta \boldsymbol{n}^{\wedge}  \tag{12}
$$

即罗德里格斯公式 (Rodrigues' Formula) 。其中 $\boldsymbol{n}^{\wedge}$ 是 $\boldsymbol{n}$ 的反对称矩阵。

对于旋转角 $\theta$ ，可以通过旋转矩阵的迹来计算：

$$
\text{tr} (\boldsymbol{R}) = \cos \theta \text{tr} (\boldsymbol{I}) + (1 - \cos \theta) \text{tr} (\boldsymbol{n} \boldsymbol{n}^T) + \sin \theta \text{tr} (\boldsymbol{n}^{\wedge}) = 1 + 2 \cos \theta \tag{13}
$$

### 2.2 欧拉角

欧拉角很常用，就不细说了。

总结一下问题，在 pitch 接近 $\pm \frac{\pi}{2}$ 时，roll 和 yaw 会出现耦合，这就是万向锁问题 (Gimbal Lock) 。解决方法是用四元数。

注意欧拉角的旋转顺序，比如 ZYX 就是先绕 Z 轴旋转，再绕 Y 轴旋转，最后绕 X 轴旋转。常用的 rpy 角的旋转顺序就是 ZYX 。

## 3. 四元数 Quaternion

四元数的提出解决了紧凑性和奇异性的问题。

### 3.1 四元数的定义

可以将四元数类比为复数来理解，如欧拉公式：

$$
e^{i \theta} = \cos \theta + i \sin \theta
$$

是一个单位长度的复数。在二维情况下，旋转可以用单位复数来表示。在三维情况下，可以用单位四元数来表示。

四元数的定义如下：

$$
\boldsymbol{q} = q_w + q_x i + q_y j + q_z k = [q_w, q_x, q_y, q_z]^T \tag{14}
$$

其中 $i, j, k$ 是虚数单位，且满足：

$$
\begin{cases}
    i^2 = j^2 = k^2 = ijk = -1 \\
    ij = k, \, jk = i, \, ki = j \\
    ji = -k, \, kj = -i, \, ik = -j\\
\end{cases} \tag{15}
$$

四元数也可以用一个标量和矢量来表示：

$$
\boldsymbol{q} = [q_w, \boldsymbol{q}_v] \tag{16}
$$

### 3.2 四元数的运算

现有两个四元数 $\boldsymbol{q}_a$ 和 $\boldsymbol{q}_b$ ，它们的运算如下：

加减法：

$$
\boldsymbol{q}_a \pm \boldsymbol{q}_b = [q_{a_w} \pm q_{b_w}, \boldsymbol{q}_{a_v} \pm \boldsymbol{q}_{b_v}] \tag{17}
$$

乘法：

$$
\boldsymbol{q}_a \boldsymbol{q}_b = [q_{a_w} q_{b_w} - \boldsymbol{q}_{a_v}^T \boldsymbol{q}_{b_v}, q_{a_w} \boldsymbol{q}_{b_v} + q_{b_w} \boldsymbol{q}_{a_v} + \boldsymbol{q}_{a_v} \times \boldsymbol{q}_{b_v}] \tag{18}
$$

模长：

$$
\| \boldsymbol{q} \| = \sqrt{q_w^2 + q_x^2 + q_y^2 + q_z^2} \tag{19}
$$

共轭：

$$
\boldsymbol{q}^* = [q_w, -\boldsymbol{q}_v] \tag{20}
$$

逆：

$$
\boldsymbol{q}^{-1} = \frac{\boldsymbol{q}^*}{\| \boldsymbol{q} \|^2} \tag{21}
$$

### 3.3 用四元数表示旋转

把一个三维空间点用虚四元数表示：

$$
\boldsymbol{p}=[0,x,y,z]^T=[0,\boldsymbol{v}]^T \tag{22}
$$

其经过旋转之后变为 $\boldsymbol{p}'$ ，用旋转矩阵描述为 $\boldsymbol{p}'=\boldsymbol{R}\boldsymbol{p}$ ，用四元数来表示此旋转为：

$$
\boldsymbol{p}'=\boldsymbol{q}\boldsymbol{p}\boldsymbol{q}^{-1} \tag{23} 
$$

最后把虚部提出，就得到旋转后的点。

### 3.4 四元数到其他旋转表示的转换

通过推导（没看懂）可以得到旋转矩阵与表示相同旋转的四元数 $\boldsymbol{q}=[w, \boldsymbol{v}]^T$ 之间的关系：

$$
\boldsymbol{R} = \boldsymbol{v}\boldsymbol{v}^T + w^2\boldsymbol{I} + 2w\boldsymbol{v}^{\wedge}
+ (\boldsymbol{v}^{\wedge})^2 \tag{24}
$$

等式两边取迹，可以得到：

$$
\text{tr}(\boldsymbol{R}) = 4w^2 - 1 \tag{25}
$$

结合式 (13) 可以得到四元数的旋转角：

$$
\theta = \arccos \left( \frac{\text{tr}(\boldsymbol{R}) + 1}{2} \right) = \arccos (2w^2 - 1) \tag{26}
$$

$$
\theta = 2 \arccos w \tag{27}
$$

至于旋转轴，其实就是四元数的虚部 $\boldsymbol{v}$ ，但需要处理模长，使四元数为单位四元数。

$$
\begin{cases}
    \theta = 2 \arccos w \\
    [n_x, n_y, n_z]^T = [q_x, q_y, q_z]^T / \sin (\theta/2) 
\end{cases} \tag{28}
$$

之后根据元素一一对应即可。

### 3.5 四元数表示旋转矩阵的推导 (gpt)

我们来推导四元数与旋转矩阵之间的关系，首先需要理解旋转矩阵和四元数的基本定义，并使用四元数表示旋转。

#### 1. 四元数与旋转的基本概念

旋转矩阵 $ \boldsymbol{R} $ 是一个 $ 3 \times 3 $ 的正交矩阵，用于描述三维空间中物体的旋转，满足条件：

$$
\boldsymbol{R}^T \boldsymbol{R} = \boldsymbol{I}, \quad \text{det}(\boldsymbol{R}) = 1
$$

四元数 $ \boldsymbol{q} $ 是一个四维复数的扩展，通常用来表示旋转，包含一个标量部分 $ q_w $ 和一个三维向量部分 $ \boldsymbol{q}_v = [q_x, q_y, q_z]^T $，表示旋转轴的方向和旋转角度。四元数表示的旋转可以使用以下形式：

$$
\boldsymbol{q} = [q_w, q_x, q_y, q_z]^T = \cos\left(\frac{\theta}{2}\right) + \sin\left(\frac{\theta}{2}\right) \boldsymbol{n} \quad \text{其中} \quad \boldsymbol{n} = \left[ \frac{q_x}{\sin(\theta / 2)}, \frac{q_y}{\sin(\theta / 2)}, \frac{q_z}{\sin(\theta / 2)} \right]
$$

其中，$ \theta $ 是旋转角，$ \boldsymbol{n} $ 是旋转轴单位向量。

#### 2. 四元数到旋转矩阵的推导

通过四元数可以得到旋转矩阵。考虑旋转轴单位向量 $ \boldsymbol{n} = [n_x, n_y, n_z]^T $，旋转角为 $ \theta $，四元数 $ \boldsymbol{q} $ 的表示为：

$$
\boldsymbol{q} = \left[\cos\left(\frac{\theta}{2}\right), n_x \sin\left(\frac{\theta}{2}\right), n_y \sin\left(\frac{\theta}{2}\right), n_z \sin\left(\frac{\theta}{2}\right) \right]
$$

我们可以用 $ \boldsymbol{q} $ 来推导旋转矩阵 $ \boldsymbol{R} $。

##### 2.1 四元数和旋转矩阵的关系

旋转矩阵 $ \boldsymbol{R} $ 和四元数 $ \boldsymbol{q} = [w, \boldsymbol{v}]^T $ 之间的关系可以表示为：

$$
\boldsymbol{R} = \boldsymbol{v} \boldsymbol{v}^T + w^2 \boldsymbol{I} + 2w \boldsymbol{v}^{\wedge} + (\boldsymbol{v}^{\wedge})^2
$$

其中，$ \boldsymbol{v} = [q_x, q_y, q_z]^T $ 是四元数的虚部，$ w = \cos\left(\frac{\theta}{2}\right) $，$ \boldsymbol{v}^{\wedge} $ 是 $ \boldsymbol{v} $ 的反对称矩阵，定义为：

$$
\boldsymbol{v}^{\wedge} = \begin{bmatrix}
0 & -v_z & v_y \\
v_z & 0 & -v_x \\
-v_y & v_x & 0
\end{bmatrix}
$$

##### 2.2 旋转矩阵的推导

通过推导，我们得到的旋转矩阵 $ \boldsymbol{R} $ 与四元数 $ \boldsymbol{q} = [w, q_x, q_y, q_z]^T $ 的关系如下：

$$
\boldsymbol{R} = \begin{bmatrix}
1 - 2(q_y^2 + q_z^2) & 2(q_x q_y - q_z w) & 2(q_x q_z + q_y w) \\
2(q_x q_y + q_z w) & 1 - 2(q_x^2 + q_z^2) & 2(q_y q_z - q_x w) \\
2(q_x q_z - q_y w) & 2(q_y q_z + q_x w) & 1 - 2(q_x^2 + q_y^2)
\end{bmatrix}
$$

这个旋转矩阵能够描述通过四元数 $ \boldsymbol{q} $ 表示的旋转。

#### 3. 旋转矩阵到四元数的推导

为了将旋转矩阵转换为四元数，首先我们利用旋转矩阵的迹（tr）来计算四元数的标量部分 $ w $，然后根据四元数的关系得到旋转轴的方向。

##### 3.1 计算四元数的标量部分

给定旋转矩阵 $ \boldsymbol{R} $，四元数的标量部分 $ w $ 可以通过矩阵的迹计算：

$$
w = \frac{1}{2} \sqrt{1 + \text{tr}(\boldsymbol{R})}
$$

其中，矩阵的迹是其对角线元素之和，即：

$$
\text{tr}(\boldsymbol{R}) = R_{11} + R_{22} + R_{33}
$$

##### 3.2 计算四元数的虚部

四元数的虚部 $ \boldsymbol{v} = [q_x, q_y, q_z]^T $ 可以通过旋转矩阵的元素计算出来。具体来说：

$$
q_x = \frac{R_{32} - R_{23}}{4w}, \quad q_y = \frac{R_{13} - R_{31}}{4w}, \quad q_z = \frac{R_{21} - R_{12}}{4w}
$$

通过这些计算，我们能够从旋转矩阵恢复出四元数。

#### 4. 总结

通过推导我们得到了旋转矩阵和四元数之间的关系：

1. **从四元数到旋转矩阵**：使用四元数的虚部和标量部分按照特定的公式转换为旋转矩阵。
2. **从旋转矩阵到四元数**：利用旋转矩阵的迹计算四元数的标量部分 $ w $，并使用矩阵元素计算虚部 $ \boldsymbol{v} $。

这种推导关系使得四元数成为表示旋转的一个有效工具，具有比旋转矩阵更紧凑的优点，同时避免了欧拉角的万向锁问题。

