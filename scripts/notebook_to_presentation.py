#!/usr/bin/env python3
"""
Notebook to Presentation Converter

Converts IMU_Project.ipynb to a styled HTML presentation with:
- Separate versioned CSS file
- Versioned HTML file (IMU_Presentation_v{N}.html)
- Same structure as the existing presentation format

Usage:
    python scripts/notebook_to_presentation.py

Each run creates a new version (v1, v2, etc.) to prevent overwriting.
"""

import json
import os
import re
import shutil
from pathlib import Path
import base64

# Project paths
PROJECT_DIR = Path(__file__).parent.parent
NOTEBOOK_PATH = PROJECT_DIR / 'IMU_Project.ipynb'
PRESENTATION_DIR = PROJECT_DIR / 'IMU_Presentation'
CSS_TEMPLATE_PATH = PRESENTATION_DIR / 'test.css'


def get_next_version(presentation_dir: Path) -> int:
    """Find existing versioned files and return the next version number."""
    existing_versions = set()
    
    if not presentation_dir.exists():
        presentation_dir.mkdir(parents=True)
        return 1
    
    for f in presentation_dir.iterdir():
        # Match HTML files: IMU_Presentation_v1.html, IMU_Presentation_v2.html, etc.
        if match := re.match(r'IMU_Presentation_v(\d+)\.html', f.name):
            existing_versions.add(int(match.group(1)))
        # Match CSS files: presentation_v1.css, presentation_v2.css, etc.
        elif match := re.match(r'presentation_v(\d+)\.css', f.name):
            existing_versions.add(int(match.group(1)))
    
    return max(existing_versions, default=0) + 1


def markdown_to_html(md_content: str) -> str:
    """Convert markdown to HTML using pure Python (no external dependencies)."""
    lines = md_content.split('\n')
    result = []
    in_list = False
    in_code_block = False
    
    for line in lines:
        # Handle code blocks
        if line.strip().startswith('```'):
            if in_code_block:
                result.append('</code></pre>')
                in_code_block = False
            else:
                result.append('<pre><code>')
                in_code_block = True
            continue
        
        if in_code_block:
            result.append(line)
            continue
        
        # Headers
        if line.startswith('#### '):
            result.append(f'<h4>{process_inline_markdown(line[5:])}</h4>')
        elif line.startswith('### '):
            result.append(f'<h3>{process_inline_markdown(line[4:])}</h3>')
        elif line.startswith('## '):
            result.append(f'<h2>{process_inline_markdown(line[3:])}</h2>')
        elif line.startswith('# '):
            result.append(f'<h1>{process_inline_markdown(line[2:])}</h1>')
        # Numbered lists
        elif re.match(r'^\d+\.\s', line):
            if not in_list:
                result.append('<ol>')
                in_list = 'ol'
            content = re.sub(r'^\d+\.\s', '', line)
            result.append(f'<li>{process_inline_markdown(content)}</li>')
        # Bullet lists
        elif line.strip().startswith('- ') or line.strip().startswith('* '):
            if not in_list:
                result.append('<ul>')
                in_list = 'ul'
            content = re.sub(r'^[\s]*[-*]\s', '', line)
            result.append(f'<li>{process_inline_markdown(content)}</li>')
        else:
            # Close any open list
            if in_list:
                result.append(f'</{in_list}>')
                in_list = False
            # Regular paragraph
            if line.strip():
                result.append(f'<p>{process_inline_markdown(line)}</p>')
    
    # Close any remaining list
    if in_list:
        result.append(f'</{in_list}>')
    
    return '\n'.join(result)


def process_inline_markdown(text: str) -> str:
    """Process inline markdown elements like bold, italic, code."""
    # Bold
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    # Italic
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
    # Inline code
    text = re.sub(r'`(.+?)`', r'<code>\1</code>', text)
    return text


def process_cell_outputs(outputs: list) -> str:
    """Process code cell outputs to HTML."""
    html_parts = []
    
    for output in outputs:
        output_type = output.get('output_type', '')
        
        if output_type == 'stream':
            # Text output (stdout/stderr)
            text = ''.join(output.get('text', []))
            html_parts.append(f'''
            <div class="jp-OutputArea-child">
              <div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain" tabindex="0">
                <pre>{text}</pre>
              </div>
            </div>''')
        
        elif output_type in ('execute_result', 'display_data'):
            data = output.get('data', {})
            
            # HTML output (tables, etc.)
            if 'text/html' in data:
                html_content = ''.join(data['text/html'])
                html_parts.append(f'''
                <div class="jp-OutputArea-child">
                  <div class="jp-RenderedHTMLCommon jp-RenderedHTML jp-OutputArea-output" data-mime-type="text/html" tabindex="0">
                    {html_content}
                  </div>
                </div>''')
            
            # Image output
            elif 'image/png' in data:
                img_data = data['image/png']
                if isinstance(img_data, list):
                    img_data = ''.join(img_data)
                html_parts.append(f'''
                <div class="jp-OutputArea-child">
                  <div class="jp-RenderedImage jp-OutputArea-output" tabindex="0">
                    <img alt="Output" src="data:image/png;base64,{img_data}" />
                  </div>
                </div>''')
            
            # Plain text fallback
            elif 'text/plain' in data:
                text = ''.join(data['text/plain'])
                html_parts.append(f'''
                <div class="jp-OutputArea-child">
                  <div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain" tabindex="0">
                    <pre>{text}</pre>
                  </div>
                </div>''')
    
    return '\n'.join(html_parts)


def convert_notebook(notebook_path: Path) -> str:
    """Convert notebook to HTML body content."""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    cells = notebook.get('cells', [])
    html_cells = []
    
    # Add initial line breaks like the original
    html_cells.append('<br>' * 6)
    
    for cell in cells:
        cell_type = cell.get('cell_type', '')
        source = ''.join(cell.get('source', []))
        
        if cell_type == 'markdown':
            # Markdown cell
            html_content = markdown_to_html(source)
            html_cells.append(f'''
    <div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
      <div class="jp-Cell-inputWrapper" tabindex="0">
        <div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
        </div>
        <div class="jp-InputArea jp-Cell-inputArea">
          <div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
            {html_content}
          </div>
        </div>
      </div>
    </div>''')
        
        elif cell_type == 'code':
            # Code cell - only show outputs (no source code)
            outputs = cell.get('outputs', [])
            if outputs:
                output_html = process_cell_outputs(outputs)
                html_cells.append(f'''
    <div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noInput">
      <div class="jp-Cell-outputWrapper">
        <div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
        </div>
        <div class="jp-OutputArea jp-Cell-outputArea">
          {output_html}
        </div>
      </div>
    </div>''')
    
    return '\n'.join(html_cells)


def generate_html(body_content: str, css_filename: str) -> str:
    """Generate complete HTML document."""
    return f'''<!DOCTYPE html>

<html lang="en">

<head>
  <link rel="stylesheet" href="{css_filename}">
  <meta charset="utf-8" />
  <meta content="width=device-width, initial-scale=1.0" name="viewport" />
  <title>IMU_Project</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.1.10/require.min.js"></script>

  <!-- Load mathjax -->
  <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/latest.js?config=TeX-AMS_CHTML-full,Safe"> </script>
  <!-- MathJax configuration -->
  <script type="text/x-mathjax-config">

    init_mathjax = function() {{
        if (window.MathJax) {{
        // MathJax loaded
            MathJax.Hub.Config({{
                TeX: {{
                    equationNumbers: {{
                    autoNumber: "AMS",
                    useLabelIds: true
                    }}
                }},
                tex2jax: {{
                    inlineMath: [ ['$','$'], ["\\\\(","\\\\)"] ],
                    displayMath: [ ['$$','$$'], ["\\\\[","\\\\]"] ],
                    processEscapes: true,
                    processEnvironments: true
                }},
                displayAlign: 'center',
                messageStyle: 'none',
                CommonHTML: {{
                    linebreaks: {{
                    automatic: true
                    }}
                }}
            }});

            MathJax.Hub.Queue(["Typeset", MathJax.Hub]);
        }}
    }}
    init_mathjax();
    </script>
  <!-- End of mathjax configuration -->
  <script type="module">
    document.addEventListener("DOMContentLoaded", async () => {{
      const diagrams = document.querySelectorAll(".jp-Mermaid > pre.mermaid");
      if (!diagrams.length) {{
        return;
      }}
      const mermaid = (await import("https://cdnjs.cloudflare.com/ajax/libs/mermaid/10.7.0/mermaid.esm.min.mjs")).default;
      mermaid.initialize({{
        maxTextSize: 100000,
        maxEdges: 100000,
        startOnLoad: false,
        fontFamily: window.getComputedStyle(document.body).getPropertyValue("--jp-ui-font-family"),
        theme: "default",
      }});
    }});
  </script>
</head>

<body class="jp-Notebook" data-jp-theme-light="true" data-jp-theme-name="JupyterLab Light">
  <main>
    {body_content}
  </main>
</body>

</html>
'''


def copy_css(template_path: Path, output_path: Path):
    """Copy CSS template to versioned output."""
    if template_path.exists():
        shutil.copy(template_path, output_path)
        print(f"  CSS copied from template: {template_path.name}")
    else:
        # Create a minimal CSS if template doesn't exist
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('''/* Modern IMU Project Presentation Styling */

:root {
    --primary-color: #2563eb;
    --secondary-color: #7c3aed;
    --accent-color: #10b981;
    --dark-bg: #1e293b;
    --light-bg: #f8fafc;
    --text-dark: #0f172a;
    --text-light: #64748b;
    --card-bg: #ffffff;
    --gradient-1: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    background-attachment: fixed;
    color: var(--text-dark);
    line-height: 1.6;
    margin: 0;
    padding: 20px;
}

main {
    max-width: 1200px;
    margin: 0 auto;
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    padding: 40px;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
}

h1, h2, h3 {
    background: var(--gradient-1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 700;
}

.jp-Cell { margin-bottom: 30px; }

.jp-RenderedText pre {
    background: #f8fafc;
    padding: 15px;
    border-radius: 8px;
    overflow-x: auto;
    border-left: 4px solid var(--primary-color);
}

.jp-RenderedImage img {
    max-width: 100%;
    border-radius: 10px;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
}

table {
    width: 100%;
    border-collapse: collapse;
    margin: 20px 0;
    background: white;
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
}

th {
    background: var(--gradient-1);
    color: white;
    padding: 15px;
    text-align: left;
}

td {
    padding: 12px 15px;
    border-bottom: 1px solid #e2e8f0;
}
''')
        print(f"  CSS generated with default styles")


def main():
    """Main conversion function."""
    print("\n" + "=" * 60)
    print("  Notebook to Presentation Converter")
    print("=" * 60)
    
    # Check notebook exists
    if not NOTEBOOK_PATH.exists():
        print(f"\nError: Notebook not found at: {NOTEBOOK_PATH}")
        return 1
    
    # Get next version number
    version = get_next_version(PRESENTATION_DIR)
    print(f"\n  Generating version: v{version}")
    
    # Output files
    html_filename = f"IMU_Presentation_v{version}.html"
    css_filename = f"presentation_v{version}.css"
    html_output_path = PRESENTATION_DIR / html_filename
    css_output_path = PRESENTATION_DIR / css_filename
    
    print(f"  Notebook: {NOTEBOOK_PATH.name}")
    print(f"  Output HTML: {html_filename}")
    print(f"  Output CSS: {css_filename}")
    
    # Convert notebook
    print("\n  Processing notebook...")
    body_content = convert_notebook(NOTEBOOK_PATH)
    
    # Generate HTML
    print("  Generating HTML...")
    html_content = generate_html(body_content, css_filename)
    
    # Write HTML
    with open(html_output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"  HTML saved: {html_output_path}")
    
    # Copy/generate CSS
    print("  Generating CSS...")
    copy_css(CSS_TEMPLATE_PATH, css_output_path)
    print(f"  CSS saved: {css_output_path}")
    
    print("\n" + "=" * 60)
    print(f"  Success! Presentation v{version} created.")
    print("=" * 60 + "\n")
    
    return 0


if __name__ == '__main__':
    exit(main())
