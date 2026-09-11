+++
title = 'FastDivMod 的起源与推导'
date = 2026-09-11T08:51:29-07:00
draft = false
tags = []
categories = []
description = ""
+++

## 1 算法起源

FastDivmod 是快速求商与余数的工程封装名称。其算法基础是 division by invariant integers，即针对常数或运行时保持不变的除数，用预计算乘数与移位替代重复整数除法。

| 时间 | 代表工作 | 意义 |
|---|---|---|
| 1973、1976 年 | 固定整数除数的快速除法研究 | 早期针对特定常数的算法 |
| 1991 年 | Robert Alverson，Integer Division Using Reciprocals | 使用倒数和整数乘法实现除法 |
| 1994 年 | Torbjörn Granlund、Peter L. Montgomery，Division by Invariant Integers using Multiplication | 系统讨论任意非零常数及运行时不变除数，并介绍 GCC 实现 |

1994 年的 PLDI 论文是经典算法来源，但不是整个思想的最早起点；论文自己引用了更早的研究。这些文献也不能确定 FastDivmod 这一具体类名最早出现在哪个代码仓库。[论文原文](https://gmplib.org/~tege/divcnst-pldi94.pdf)

## 2 从定点倒数理解乘法除法

设真正的商为 q，余数为 r，则：

$$n=qd+r,\qquad 0\le r<d.$$

选择缩放倍数 S，用整数 M 表示倒数的近似值 M/S。为了让最终的缩放操作成为右移，取：

$$S=2^k.$$

希望计算：

$$\hat q=\left\lfloor\frac{nM}{S}\right\rfloor.$$

这里用 q 表示真正的商，使用带帽的商表示候选计算结果，避免在证明前就假定两者相等。

例如，除以 3 时用 334/1000 近似 1/3。对于 0 到 99 的整数，乘以 0.334 再向下取整会得到正确整数商。但 n=500 时得到 167，真正的整数商为 166。这个例子说明，近似精度必须与输入范围配合。

## 3 为什么选择向上取整

为保证近似结果不低于真正的 n/d，希望：

$$\frac{M}{S}\ge\frac1d\quad\Longleftrightarrow\quad M\ge\frac Sd.$$

满足这个约束的最小整数为：

$$\boxed{M=\left\lceil\frac Sd\right\rceil=\left\lceil\frac{2^k}{d}\right\rceil.}$$

这就是 magic number 公式的来源。向上取整是本算法变体的选择；也存在配合修正操作的向下取整算法。

由上取整定义：

$$\frac Sd\le M<\frac Sd+1.$$

乘以正数 d：

$$S\le Md<S+d.$$

定义整数误差 e=Md−S，就得到：

$$Md=S+e,\qquad 0\le e<d.$$

## 4 误差展开式逐步推导

从 Md=S+e 出发，先解出 M：

$$M=\frac{S+e}{d}.$$

代入候选结果取整前的表达式：

$$\frac{nM}{S}=\frac nS\cdot\frac{S+e}{d}
=\frac{n(S+e)}{dS}
=\frac{nS}{dS}+\frac{ne}{dS}
=\frac nd+\frac{ne}{dS}.$$

再将 n=qd+r 代入 n/d：

$$\frac nd=\frac{qd+r}{d}=q+\frac rd.$$

合并即得：

$$\boxed{\frac{nM}{S}=q+\frac rd+\frac{ne}{dS}.}$$

三部分依次为真正的整数商、原有的小数部分、近似倒数引入的非负误差。

## 5 从正确商的区间推导误差条件

要使候选商等于真正的商，取整前的值必须位于同一个整数区间：

$$\hat q=q\quad\Longleftrightarrow\quad q\le\frac{nM}{S}<q+1.$$

因为余数和误差非负，下界自动成立。只需保证上界：

$$\frac rd+\frac{ne}{dS}<1.$$

两侧乘以正数 dS：

$$rS+ne<dS\quad\Longleftrightarrow\quad\boxed{ne<(d-r)S.}$$

这是在上述定义下，针对某个输入 n 的精确正确性条件。但预计算时不希望依赖每个输入的余数 r。

因为 r≤d−1，所以 d−r≥1。用最小间隔代替实际间隔，可以采用更强而更容易验证的条件：

$$\boxed{ne<S=2^k.}$$

它是充分条件，不是必要条件：即使不满足，某些输入仍可能算对，只是这条保守证明不再提供保证。

## 6 如何选取 k

假设所有输入均满足：

$$0\le n<2^B.$$

选择：

$$L=\lceil\log_2d\rceil,\qquad k=B+L.$$

因为 e<d≤2^L，因此：

$$ne<2^B\cdot2^L=2^{B+L}=2^k.$$

于是能够保证规定范围内每个输入都得到正确结果。该选择是简单的充分方案，不一定是某个除数所需的最小 k。

CUTLASS 常见非负 int 索引的上限为 2³¹，所以取 B=31，得到：

$$k=31+\lceil\log_2d\rceil.$$

在 2≤d≤2³¹−1 的范围内，令 L=ceil(log₂d)，有 d>2^(L−1)，从而 2^(31+L)/d<2³²；再利用整数 d≥2^(L−1)+1，可保证上取整后的 M 仍能放入 uint32_t。d=1 单独处理。

## 7 从数学表达式到 CUDA 指令

对非负整数，右移 k 位等于除以 2^k 后向下取整，因此：

$$\boxed{q=\left\lfloor\frac{nM}{2^k}\right\rfloor=(nM)\gg k.}$$

这里的乘积必须完整保留需要的高位，不能先用 32 位乘法截断再右移。

以除以 10 为例：

$$L=4,\quad k=35,\quad M=\left\lceil\frac{2^{35}}{10}\right\rceil=3435973837.$$

M 的十六进制表示为 0xCCCCCCCD。可写为：

```cpp
uint32_t q = (uint64_t(n) * 0xCCCCCCCDu) >> 35;
uint32_t r = n - q * 10;
```

__umulhi 返回两个 32 位无符号整数乘积的高 32 位：

$$\operatorname{umulhi}(n,M)=\left\lfloor\frac{nM}{2^{32}}\right\rfloor.$$

对整数乘积，先右移 32 位再右移 3 位，与一次右移 35 位相同。因此代码可改为：

```cpp
uint32_t q = __umulhi(n, 0xCCCCCCCDu) >> 3;
uint32_t r = n - q * 10;
```

一般情况下，设备端右移量为 k−32=L−1。余数公式直接来自 n=qd+r，两侧减去 qd 即得 r=n−qd。

## 8 性能意义与适用边界

当除数是一次 kernel 调用中保持不变的张量维度时，可以在 host 端计算 M 和移位量，再让大量线程复用。初始化本身仍可能需要除法，但它的成本被后续大量计算摊薄。[CUTLASS 源码](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/fast_math.h)

本推导面向非负输入和正除数。d=0 不合法，d=1 可直接返回原数和零余数。完整 uint32_t 范围、负数或更宽输入需要相应的参数、位宽处理或修正算法。某些除数的参数可能在更大范围内也成立，但不能据此扩大整个简化实现的保证范围。

当除数每次变化时，预计算成本可能抵消收益；当除数在编译时已知时，编译器通常可以自行进行相关优化，显式对象的主要价值在于运行时确定但反复使用的除数。

## 参考资料

1. Granlund 与 Montgomery，1994，Division by Invariant Integers using Multiplication：https://gmplib.org/~tege/divcnst-pldi94.pdf
2. NVIDIA CUTLASS，include/cutlass/fast_math.h：https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/fast_math.h
3. Yifei Li，Classic Round-Up Variant of Fast Unsigned Division by Constants Algorithm and Full Proof：https://arxiv.org/pdf/2412.03680

