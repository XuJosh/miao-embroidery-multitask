# 中文 SCI 论文 LaTeX 模板

## 文件说明

- `chinese_sci_template.tex`：通用学术论文字模板，包含标准 SCI 论文结构。
- `references.bib`：示例参考文献库。

## 使用方法

1. 确保已安装 TeX Live / MiKTeX，并支持 `xelatex` 编译。
2. 在终端中执行：

```bash
xelatex chinese_sci_template.tex
bibtex chinese_sci_template
xelatex chinese_sci_template.tex
xelatex chinese_sci_template.tex
```

3. 将 `[请在此处填写...]` 替换为你的实际内容。
4. 根据目标期刊要求调整：
   - 页面边距（`geometry`）
   - 参考文献样式（`\bibliographystyle{}`）
   - 行号（`\linenumbers` 可删除）
   - 字体大小与行距

## 标准结构

1. 标题与作者信息
2. 摘要（Abstract）与关键词（Keywords）
3. 引言（Introduction）
4. 相关工作（Related Work）
5. 方法（Methodology）
6. 实验（Experiments）
7. 结果与讨论（Results and Discussion）
8. 结论（Conclusion）
9. 致谢（Acknowledgments，可选）
10. 参考文献（References）
11. 附录（Appendix，可选）

## 提示

- 投稿前请下载目标期刊的官方 LaTeX 模板（如 Elsevier、IEEE、Springer 等），并将本模板内容迁移过去。
- 多数 SCI 期刊要求使用期刊提供的 `bst` 参考文献样式文件。
