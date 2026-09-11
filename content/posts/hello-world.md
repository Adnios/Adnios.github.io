+++
title = 'Hello World'
date = 2026-09-11T23:00:00+08:00
draft = false
tags = ['blog']
categories = ['杂记']
description = '博客的第一篇文章'
+++

博客搭建完成，基于 [Hugo](https://gohugo.io/) 与 [PaperMod](https://github.com/adityatelange/hugo-PaperMod) 主题，托管在 GitHub Pages。

## 写作流程

```bash
hugo new posts/my-post.md   # 新建文章
hugo server -D              # 本地预览，含草稿
git add . && git commit -m "post: my post" && git push
```

推送到 `main` 分支后，GitHub Actions 会自动构建并发布。

## 代码高亮测试

```cuda
__global__ void axpy(int n, float a, const float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}
```
