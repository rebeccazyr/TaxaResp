#!/usr/bin/env python3
"""Render taxonomy connected tree text into a standalone HTML viewer.

Input format example:
  Field of study (ROOT_FOS)
    - Computer science (41008148)
      - Algorithm (11413529)
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize taxonomy tree txt as HTML")
    parser.add_argument("input", help="Path to *_personal_taxonomy_connected_tree.txt")
    parser.add_argument(
        "-o",
        "--output",
        help="Output HTML path (default: same path with .html)",
    )
    parser.add_argument(
        "--title",
        default="Personal Taxonomy Viewer",
        help="Title shown in HTML",
    )
    return parser.parse_args()


def parse_tree_lines(lines: List[str]) -> Dict:
    root = {"name": "ROOT", "children": []}
    stack: List[Dict] = [root]

    for raw in lines:
        line = raw.rstrip("\n")
        if not line.strip():
            continue

        stripped = line.lstrip(" ")
        leading_spaces = len(line) - len(stripped)

        if stripped.startswith("- "):
            depth = leading_spaces // 2
            label = stripped[2:].strip()
        else:
            depth = 0
            label = stripped.strip()

        node = {"name": label, "children": []}

        while len(stack) > depth + 1:
            stack.pop()

        parent = stack[-1]
        parent["children"].append(node)
        stack.append(node)

    # If the file starts with a single root line, use it directly.
    if len(root["children"]) == 1:
        return root["children"][0]
    return root


def render_html(tree: Dict, page_title: str) -> str:
    tree_json = json.dumps(tree, ensure_ascii=False)
    safe_title = html.escape(page_title)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{safe_title}</title>
  <style>
    :root {{
      --bg: #f6f7fb;
      --panel: #ffffff;
      --text: #1f2937;
      --muted: #6b7280;
      --line: #d1d5db;
      --accent: #0f766e;
      --accent-bg: #ccfbf1;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 24px;
      background: radial-gradient(circle at top right, #e0f2fe, var(--bg) 35%);
      color: var(--text);
      font-family: "IBM Plex Sans", "Segoe UI", Arial, sans-serif;
    }}
    .wrap {{
      max-width: 1200px;
      margin: 0 auto;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.06);
      overflow: hidden;
    }}
    .head {{
      padding: 16px 18px;
      border-bottom: 1px solid var(--line);
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
    }}
    h1 {{
      font-size: 18px;
      margin: 0;
      font-weight: 700;
    }}
    .muted {{
      color: var(--muted);
      font-size: 13px;
    }}
    .btn {{
      border: 1px solid var(--line);
      background: #fff;
      color: var(--text);
      padding: 6px 10px;
      border-radius: 8px;
      font-size: 13px;
      cursor: pointer;
    }}
    .btn:hover {{ border-color: #9ca3af; }}
    .body {{ padding: 14px 18px 22px; overflow: auto; }}
    ul.tree, ul.tree ul {{
      list-style: none;
      margin: 0;
      padding-left: 18px;
      position: relative;
    }}
    ul.tree ul::before {{
      content: "";
      position: absolute;
      left: 6px;
      top: 0;
      bottom: 8px;
      width: 1px;
      background: var(--line);
    }}
    li.node {{
      margin: 6px 0;
      position: relative;
      padding-left: 8px;
    }}
    li.node::before {{
      content: "";
      position: absolute;
      left: -12px;
      top: 12px;
      width: 12px;
      height: 1px;
      background: var(--line);
    }}
    .row {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 6px;
      border-radius: 7px;
    }}
    .row:hover {{ background: #f8fafc; }}
    .toggle {{
      width: 18px;
      height: 18px;
      border: 1px solid var(--line);
      border-radius: 4px;
      font-size: 11px;
      line-height: 16px;
      text-align: center;
      cursor: pointer;
      user-select: none;
      color: var(--muted);
      background: #fff;
    }}
    .toggle.hidden {{
      visibility: hidden;
      pointer-events: none;
    }}
    .label {{
      font-size: 13px;
      white-space: nowrap;
    }}
    .collapsed > ul {{ display: none; }}
    .root-chip {{
      background: var(--accent-bg);
      color: var(--accent);
      border: 1px solid #99f6e4;
      font-size: 12px;
      padding: 2px 8px;
      border-radius: 999px;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="head">
      <h1>{safe_title}</h1>
      <span class="root-chip">taxonomy tree</span>
      <span class="muted">Click +/- to fold/unfold branches.</span>
      <button id="expandAll" class="btn">Expand all</button>
      <button id="collapseAll" class="btn">Collapse all</button>
    </div>
    <div class="body">
      <div id="tree"></div>
    </div>
  </div>
  <script>
    const data = {tree_json};

    function makeNode(node) {{
      const li = document.createElement("li");
      li.className = "node";

      const row = document.createElement("div");
      row.className = "row";

      const btn = document.createElement("span");
      btn.className = "toggle";
      row.appendChild(btn);

      const label = document.createElement("span");
      label.className = "label";
      label.textContent = node.name || "";
      row.appendChild(label);

      li.appendChild(row);

      const children = node.children || [];
      if (children.length === 0) {{
        btn.classList.add("hidden");
      }} else {{
        btn.textContent = "−";
        const ul = document.createElement("ul");
        for (const child of children) {{
          ul.appendChild(makeNode(child));
        }}
        li.appendChild(ul);
        btn.addEventListener("click", () => {{
          const collapsed = li.classList.toggle("collapsed");
          btn.textContent = collapsed ? "+" : "−";
        }});
      }}
      return li;
    }}

    function render() {{
      const host = document.getElementById("tree");
      const ul = document.createElement("ul");
      ul.className = "tree";
      ul.appendChild(makeNode(data));
      host.appendChild(ul);
    }}

    function setAll(collapsed) {{
      const nodes = document.querySelectorAll("li.node");
      nodes.forEach((li) => {{
        const btn = li.querySelector(":scope > .row > .toggle");
        const hasChildren = !!li.querySelector(":scope > ul");
        if (!hasChildren || !btn || btn.classList.contains("hidden")) return;
        li.classList.toggle("collapsed", collapsed);
        btn.textContent = collapsed ? "+" : "−";
      }});
    }}

    document.getElementById("expandAll").addEventListener("click", () => setAll(false));
    document.getElementById("collapseAll").addEventListener("click", () => setAll(true));
    render();
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path = Path(args.output) if args.output else input_path.with_suffix(".html")

    lines = input_path.read_text(encoding="utf-8").splitlines()
    tree = parse_tree_lines(lines)
    html_text = render_html(tree, args.title)
    output_path.write_text(html_text, encoding="utf-8")

    print(f"Generated HTML: {output_path}")


if __name__ == "__main__":
    main()
