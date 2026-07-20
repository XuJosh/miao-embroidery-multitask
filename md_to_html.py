import os

# Read both markdown files
md_paths = [
    'D:/song/kimi_DEMO/evaluation_metrics_explanation.md',
    'D:/song/kimi_DEMO/arcface_dynamic_loss_explanation.md',
]

html_parts = []
html_parts.append("""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>ArcFace 与评估标准说明</title>
<script>
MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']],
    displayMath: [['$$', '$$'], ['\\[', '\\]']]
  }
};
</script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
<style>
body { font-family: "Segoe UI", "Microsoft YaHei", sans-serif; max-width: 900px; margin: 40px auto; padding: 20px; line-height: 1.8; color: #333; }
h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
h2 { color: #34495e; margin-top: 35px; }
h3 { color: #555; }
table { border-collapse: collapse; width: 100%; margin: 20px 0; }
th, td { border: 1px solid #ddd; padding: 10px; text-align: left; }
th { background: #f4f6f8; }
blockquote { border-left: 4px solid #3498db; padding-left: 15px; color: #666; }
code { background: #f4f4f4; padding: 2px 6px; border-radius: 3px; }
</style>
</head>
<body>
""")

for path in md_paths:
    if not os.path.exists(path):
        continue
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    # Very simple markdown-to-HTML conversion
    lines = content.split('\n')
    in_code = False
    for line in lines:
        if line.startswith('```'):
            in_code = not in_code
            if not in_code:
                html_parts.append('</code></pre>')
            else:
                html_parts.append('<pre><code>')
            continue
        if in_code:
            html_parts.append(line.replace('<', '&lt;').replace('>', '&gt;') + '\n')
            continue
        # Headers
        if line.startswith('# '):
            html_parts.append(f'<h1>{line[2:]}</h1>')
        elif line.startswith('## '):
            html_parts.append(f'<h2>{line[3:]}</h2>')
        elif line.startswith('### '):
            html_parts.append(f'<h3>{line[4:]}</h3>')
        elif line.startswith('---'):
            html_parts.append('<hr>')
        elif line.startswith('|') and '|' in line[1:]:
            # Table handling: skip separator lines, process header and data rows
            if not line.startswith('|---') and not line.startswith('|:--'):
                cells = [c.strip() for c in line.split('|')[1:-1]]
                if cells:
                    tag = 'th' if not html_parts[-1].startswith('<table') else 'td'
                    # Simple approach: first table row is header
                    if not html_parts[-1].startswith('<table'):
                        html_parts.append('<table>')
                    row_html = '<tr>' + ''.join(f'<{tag}>{c}</{tag}>' for c in cells) + '</tr>'
                    html_parts.append(row_html)
        elif line.startswith('• ') or line.startswith('- '):
            html_parts.append(f'<li>{line[2:]}</li>')
        elif line.strip() == '':
            html_parts.append('</ul><ul>' if html_parts and html_parts[-1].startswith('<li>') else '<p></p>')
        else:
            html_parts.append(f'<p>{line}</p>')

html_parts.append("""
</body>
</html>
""")

html_content = '\n'.join(html_parts)
html_path = 'D:/song/kimi_DEMO/evaluation_arcface_metrics.html'
with open(html_path, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f'HTML saved to: {html_path}')
