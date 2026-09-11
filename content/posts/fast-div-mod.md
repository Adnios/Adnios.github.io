+++
title = 'FastDivMod 的起源与推导'
date = 2026-09-11T08:51:29-07:00
draft = false
tags = []
categories = []
description = ""
+++

本文的 `FastDivmod` 指的是 **CUDA / CUTLASS 里的 `FastDivmod`，它的算法根源是“用预计算的倒数乘法实现整数除法”**，历史可以追溯到 GPU 出现之前。 这里要区分：`FastDivmod` 是代码里的封装名称，背后的算法通常叫 **division by invariant integers** 或 **magic-number division**。

> Torbjörn Granlund, Peter L. Montgomery: Division by Invariant Integers using Multiplication. PLDI 1994: 61-72

[1994 年的 PLDI 论文](https://gmplib.org/~tege/divcnst-pldi94.pdf)是经典算法来源，但不是整个思想的最早起点；论文自己引用了更早的研究。

它解决的问题是：同一个除数 `d` 被反复使用时，先为 `d` 算好一个整数乘数和移位量，之后通过乘法、移位以及必要的修正，得到**精确的整数商**。余数再算：

```cpp
r = n - q * d;
```

[CUTLASS](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/fast_math.h#L368) 把这一思路封装成了 `FastDivmod`。它的 32 位实现核心可以概括为：

```cpp
// 预先根据 d 算好 multiplier 和 shift_right
q = (d == 1) ? n : (__umulhi(n, multiplier) >> shift_right);
r = n - q * d;
```

`__umulhi` 取两个 32 位无符号整数乘积的高 32 位。这个简化版本有其输入范围约束，不能直接推广成任意有符号整数除法。此外 CutLASS 推荐：**除数在整个 grid 中不变时，在 host 端预计算，再作为 kernel 参数传进去**。

```cpp
/// Construct the FastDivmod object, in host code ideally.
///
/// This precomputes some values based on the divisor and is computationally expensive.
```

我们先用一个十进制例子理解，再对应到 CUDA。

**1. 为什么乘近似倒数，也能得到精确整数商？**

假设要计算 `n / 3`，且 `n` 是 `0～99` 的整数。

用 `0.334` 近似 `1/3`，可以这样算：

$$
q=\left\lfloor n\times0.334\right\rfloor
=\left\lfloor\frac{n\times334}{1000}\right\rfloor
$$

例如：

| n  | 真正的 n/3 | n×0.334 | 向下取整 |
| -- | ------: | ------: | ---: |
| 8  |  2.666… |   2.672 |    2 |
| 9  |       3 |   3.006 |    3 |
| 11 |  3.666… |   3.674 |    3 |

虽然小数部分有误差，**只要误差没有让结果跨过下一个整数，商就完全正确。**

但输入范围很重要：`n = 500` 时，`500 × 0.334 = 167`，真正的整数商却是 `166`。因此要根据输入上限选择倒数精度。

**2. 计算机用 \(2^k\) 作为缩放倍数**

这样最后的除法就能用右移完成。预计算：

$$
M=\left\lceil\frac{2^k}{d}\right\rceil
$$

运行时计算：

$$
\boxed{q=(nM)\gg k}
$$

这里的 `M` 就是 **magic number**。它表示放大 \(2^k\) 倍后、向上取整的倒数。

向上取整可以保证估计结果不会低于真正的 \(n/d\)；再选择足够大的 `k`，保证它不会高到下一个整数。

**3. 怎样保证误差足够小？**

写成：

$$
n=qd+r,\quad 0\le r<d
$$

又因为 `M` 向上取整：

$$
Md=2^k+e,\quad 0\le e<d
$$

先从上式解出：

$$
M=\frac{2^k+e}{d}
$$

代入：

$$
\begin{aligned}
\frac{nM}{2^k}
&=\frac{n(2^k+e)}{d\,2^k}\\
&=\frac nd+\frac{ne}{d\,2^k}
\end{aligned}
$$

又因为真正的整数除法满足：

$$
n=qd+r
\quad\Rightarrow\quad
\frac nd=q+\frac rd
$$

于是：

$$
\frac{nM}{2^k}
=q+\underbrace{\frac rd}_{原有小数部分}
+\underbrace{\frac{ne}{d\,2^k}}_{近似误差}
$$

最坏情况下，原有小数部分是 \((d-1)/d\)，距离下一个整数还有 \(1/d\)。

因此，只要：

$$
\frac{ne}{d\,2^k}<\frac1d
\quad\Longleftrightarrow\quad
\boxed{ne<2^k}
$$

就能保证向下取整后仍然是 `q`。这是一个充分条件。

对于 CUTLASS 常见的非负 `int` 索引，即 \(n<2^{31}\)，可以取：

$$
L=\lceil\log_2d\rceil,\qquad k=31+L
$$

由于 $e<d\le2^L$，自然有：

$$
ne<2^{31}\cdot2^L=2^k
$$

这就是那套参数选择背后的数学依据。

**4. 对应到 CUDA 代码**

以除以 `10` 为例：

$$
k=31+\lceil\log_2 10\rceil=35
$$

$$
M=\left\lceil\frac{2^{35}}{10}\right\rceil
=3435973837=\texttt{0xCCCCCCCD}
$$

于是可以精确计算：

```cpp
uint32_t q = (uint64_t(n) * 0xCCCCCCCDu) >> 35;
uint32_t r = n - q * 10;
```

CUDA 的 `__umulhi` 直接取 32 位整数乘积的高 32 位，相当于先右移 32 位，所以代码变成：

```cpp
uint32_t q = __umulhi(n, 0xCCCCCCCDu) >> 3;
uint32_t r = n - q * 10;
```

对于不同的除数，提前计算对应的 `M` 和移位量即可。完整无符号范围、负数等情况，可能需要其他参数或额外修正。

**它快的关键是复用预计算结果。** 比如一个 kernel 中所有线程都用同一个张量维度 `d` 做索引转换：只需初始化一次，之后每次求商和余数都用乘法、移位和减法完成。
