# 论文 LaTeX 模板说明

这是一个基于 `elsarticle` 的通用 SCI 论文模板，适合投稿 Pattern Recognition、Neurocomputing、Expert Systems with Applications 等期刊，也可以改成 IEEEtran 用于会议。

## 文件结构

```
paper/
├── main.tex              # 主文件
├── references.bib        # 参考文献
├── figures/              # 图片目录
│   ├── network_architecture.pdf   # 网络结构图（待替换）
│   └── gradcam_examples.pdf       # Grad-CAM 可视化（待替换）
└── README.md
```

## 编译方式

### 方式 1：本地编译（推荐 TeX Live / MiKTeX）

```bash
cd paper
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

或使用 `latexmk`：

```bash
latexmk -pdf main.tex
```

### 方式 2：Overleaf

1. 把整个 `paper/` 文件夹打包成 zip。
2. 上传到 Overleaf。
3. 选择编译器为 `pdfLaTeX` 或 `XeLaTeX`。

## 需要替换的内容

- `[Dataset Name]`：你的数据集名称。
- `[number]`、`[xxx / xxx / xxx]`：数据集数量。
- `XXXNet`：给你的网络起个名字。
- `[xx.x\%]`：填入实际实验结果。
- 作者信息、单位、邮箱、基金号。
- `figures/network_architecture.pdf` 和 `figures/gradcam_examples.pdf`：换成你自己的图。
- `\citep{sample-ref}`：替换成真实文献。

## 切换到期刊格式

### 单栏预印本（默认）
```latex
\documentclass[preprint,12pt,a4paper]{elsarticle}
```

### 双栏审稿版
```latex
\documentclass[review]{elsarticle}
```

### 最终出版版
```latex
\documentclass[final]{elsarticle}
```

## 切换为 IEEE 会议格式

如果需要投 IEEE 会议，把 `main.tex` 开头改为：

```latex
\documentclass[conference]{IEEEtran}
```

并去掉 `elsarticle` 特有的 `frontmatter`、`highlights`、`keywords` 环境，使用 IEEE 的 `\IEEEtitleabstractindextext`。

## 下一步建议

1. 把 `main.tex` 中的 `xx.x` 替换为真实实验数值。
2. 用 TikZ 或 PPT 画一张清晰的网络结构图，导出 PDF 放到 `figures/`。
3. 运行 Grad-CAM 生成可视化图。
4. 补充 `references.bib` 中你实际引用的文献。
