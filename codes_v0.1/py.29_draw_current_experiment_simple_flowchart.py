#!/usr/bin/env python3
"""Create the two-stage current-experiment flowchart as editable draw.io and SVG."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import html
import xml.etree.ElementTree as ET


ROOT = Path("/Users/yaoruntian/LiLab/phage/phage_designer_v0.1")
OUT = ROOT / "results" / "current_experiment_simple_flowchart"
DRAWIO = OUT / "current_experiment_simple_flowchart.drawio"
SVG = OUT / "current_experiment_simple_flowchart.svg"
PNG = OUT / "current_experiment_simple_flowchart.png"
PDF = OUT / "current_experiment_simple_flowchart.pdf"

W, H = 1600, 900

BG = "#FFFFFF"
REGION1 = "#F7FFF3"
REGION2 = "#F4FCFB"
PALE_GREEN = "#EEFFE6"
PALE_TEAL = "#E8FAF7"
MID_TEAL = "#0BDACD"
DEEP_BLUE = "#163179"
ARROW = "#1D6F78"
MUTED = "#6B7C8F"
REGION_STROKE = "#B8DCCF"
TEXT = "#17324D"


@dataclass(frozen=True)
class Node:
    id: str
    x: int
    y: int
    w: int
    h: int
    label: str
    fill: str
    bold: bool = False
    font_size: int = 16


NODES = [
    Node("s1_input", 80, 155, 150, 70, "蛋白功能", PALE_GREEN, True),
    Node("s1_ref", 270, 155, 170, 70, "参考蛋白检索", PALE_TEAL),
    Node("s1_g", 480, 155, 170, 70, "ΦX174 G 蛋白", PALE_GREEN, True),
    Node("s1_esm", 690, 145, 185, 90, "ESM2-650M\n掩码采样", MID_TEAL, True),
    Node("s1_gen", 915, 145, 170, 90, "3,000 条\nG-like 蛋白", PALE_TEAL, True),
    Node("s1_filter", 1125, 145, 185, 90, "多层蛋白筛选", PALE_GREEN, True),
    Node("s1_rank", 1350, 145, 160, 90, "综合排序", PALE_TEAL, True),
    Node("s1_counts", 1125, 275, 385, 58, "473 条  →  100 条  →  10 条", PALE_GREEN, True, 17),
    Node("s2_protein", 1233, 525, 170, 70, "10 条候选蛋白", PALE_GREEN, True),
    Node("s2_rt", 993, 515, 190, 90, "Evo2 约束\n反向翻译", MID_TEAL, True),
    Node("s2_100cds", 753, 525, 180, 70, "100 条合法 CDS", PALE_TEAL),
    Node("s2_score", 523, 515, 180, 90, "Evo2 似然排序", PALE_GREEN, True),
    Node("s2_10cds", 303, 525, 170, 70, "10 条优选 CDS", PALE_TEAL, True),
    Node("s2_template", 80, 680, 200, 70, "ΦX174 模板基因组", PALE_GREEN, True),
    Node("s2_merge", 365, 670, 190, 90, "替换 G CDS", MID_TEAL, True),
    Node("s2_genomes", 650, 670, 190, 90, "10 条设计基因组", PALE_TEAL, True),
    Node("s2_validate", 960, 670, 230, 90, "多层计算验证", PALE_GREEN, True),
    Node("s2_evidence", 1270, 670, 220, 90, "综合证据表", PALE_TEAL, True),
]


EDGES = [
    ("e01", "s1_input", "s1_ref", ""),
    ("e02", "s1_ref", "s1_g", ""),
    ("e03", "s1_g", "s1_esm", ""),
    ("e04", "s1_esm", "s1_gen", ""),
    ("e05", "s1_gen", "s1_filter", ""),
    ("e06", "s1_filter", "s1_rank", ""),
    ("e07", "s1_rank", "s1_counts", ""),
    ("e08", "s1_counts", "s2_protein", ""),
    ("e09", "s2_protein", "s2_rt", ""),
    ("e10", "s2_rt", "s2_100cds", ""),
    ("e11", "s2_100cds", "s2_score", ""),
    ("e12", "s2_score", "s2_10cds", ""),
    ("e13", "s2_template", "s2_merge", ""),
    ("e14", "s2_10cds", "s2_merge", ""),
    ("e15", "s2_merge", "s2_genomes", ""),
    ("e16", "s2_genomes", "s2_validate", ""),
    ("e17", "s2_validate", "s2_evidence", ""),
]


def style(parts: list[str]) -> str:
    return ";".join(parts) + ";"


def add_vertex(root: ET.Element, node: Node) -> None:
    font_color = "#FFFFFF" if node.fill == MID_TEAL else TEXT
    node_style = style([
        "rounded=1",
        "arcSize=14",
        "whiteSpace=wrap",
        "html=1",
        "align=center",
        "verticalAlign=middle",
        f"fillColor={node.fill}",
        f"strokeColor={DEEP_BLUE}",
        "strokeWidth=2",
        "fontFamily=Arial",
        f"fontSize={node.font_size}",
        f"fontColor={font_color}",
        f"fontStyle={1 if node.bold else 0}",
        "spacing=8",
    ])
    value = "<b>" + node.label.replace("\n", "<br>") + "</b>" if node.bold else node.label.replace("\n", "<br>")
    cell = ET.SubElement(root, "mxCell", {
        "id": node.id,
        "value": value,
        "style": node_style,
        "vertex": "1",
        "parent": "1",
    })
    ET.SubElement(cell, "mxGeometry", {
        "x": str(node.x), "y": str(node.y), "width": str(node.w), "height": str(node.h), "as": "geometry"
    })


def add_region(root: ET.Element, rid: str, x: int, y: int, w: int, h: int, fill: str) -> None:
    cell = ET.SubElement(root, "mxCell", {
        "id": rid,
        "value": "",
        "style": style([
            "rounded=1", "arcSize=10", "html=1", "container=1", "pointerEvents=0",
            "dashed=1", "dashPattern=12 8", "fillColor=none",
            f"strokeColor={REGION_STROKE}", "strokeWidth=2",
        ]),
        "vertex": "1", "parent": "1",
    })
    ET.SubElement(cell, "mxGeometry", {
        "x": str(x), "y": str(y), "width": str(w), "height": str(h), "as": "geometry"
    })


def add_text(root: ET.Element, tid: str, x: int, y: int, w: int, h: int, value: str, size: int, bold: bool = False) -> None:
    cell = ET.SubElement(root, "mxCell", {
        "id": tid,
        "value": ("<b>" + value + "</b>") if bold else value,
        "style": style([
            "text", "html=1", "strokeColor=none", "fillColor=none", "whiteSpace=wrap",
            "overflow=visible", "align=left", "verticalAlign=middle", "fontFamily=Arial",
            f"fontSize={size}", f"fontColor={DEEP_BLUE}", f"fontStyle={1 if bold else 0}",
        ]),
        "vertex": "1", "parent": "1",
    })
    ET.SubElement(cell, "mxGeometry", {
        "x": str(x), "y": str(y), "width": str(w), "height": str(h), "as": "geometry"
    })


def add_edge(root: ET.Element, eid: str, source: str, target: str, label: str) -> None:
    cell = ET.SubElement(root, "mxCell", {
        "id": eid,
        "value": label,
        "style": style([
            "edgeStyle=orthogonalEdgeStyle", "rounded=1", "orthogonalLoop=1", "jettySize=auto",
            "html=1", f"strokeColor={ARROW}", "strokeWidth=2", "endArrow=classic", "endFill=1",
            "fontFamily=Arial", "fontSize=12", f"fontColor={MUTED}", "labelBackgroundColor=#FFFFFF",
        ]),
        "edge": "1", "parent": "1", "source": source, "target": target,
    })
    ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})


def build_drawio() -> None:
    mxfile = ET.Element("mxfile", {
        "host": "app.diagrams.net", "agent": "Codex", "version": "26.0.9", "pages": "1"
    })
    diagram = ET.SubElement(mxfile, "diagram", {"id": "current-experiment-simple", "name": "Current experiment"})
    model = ET.SubElement(diagram, "mxGraphModel", {
        "dx": "1600", "dy": "900", "grid": "1", "gridSize": "10", "guides": "1", "tooltips": "1",
        "connect": "1", "arrows": "1", "fold": "1", "page": "1", "pageScale": "1",
        "pageWidth": str(W), "pageHeight": str(H), "math": "0", "shadow": "0",
    })
    root = ET.SubElement(model, "root")
    ET.SubElement(root, "mxCell", {"id": "0"})
    ET.SubElement(root, "mxCell", {"id": "1", "parent": "0"})

    add_region(root, "stage1_region", 40, 40, 1520, 360, REGION1)
    add_region(root, "stage2_region", 40, 430, 1520, 420, REGION2)
    add_text(root, "stage1_title", 70, 55, 620, 46, "阶段一｜G-like 蛋白生成与筛选", 26, True)
    add_text(root, "stage2_title", 70, 445, 720, 46, "阶段二｜CDS生成、模板构建与验证", 26, True)
    add_text(root, "s1_mask_rates", 655, 245, 255, 30, "掩码率 5% · 10% · 20% · 30%", 14, False)

    for node in NODES:
        add_vertex(root, node)
    for edge in EDGES:
        add_edge(root, *edge)

    ET.indent(mxfile, space="  ")
    ET.ElementTree(mxfile).write(DRAWIO, encoding="utf-8", xml_declaration=True)


def svg_text(x: float, y: float, label: str, size: int, color: str, bold: bool = False, anchor: str = "middle") -> str:
    lines = label.split("\n")
    line_h = size * 1.35
    start_y = y - (len(lines) - 1) * line_h / 2
    weight = "700" if bold else "500"
    tspans = "".join(
        f'<tspan x="{x}" y="{start_y + i * line_h:.1f}">{html.escape(line)}</tspan>'
        for i, line in enumerate(lines)
    )
    return (
        f'<text text-anchor="{anchor}" font-family="Arial, PingFang SC, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{color}">{tspans}</text>'
    )


def node_anchor(node: Node, side: str) -> tuple[float, float]:
    if side == "left":
        return node.x, node.y + node.h / 2
    if side == "right":
        return node.x + node.w, node.y + node.h / 2
    if side == "top":
        return node.x + node.w / 2, node.y
    return node.x + node.w / 2, node.y + node.h


def build_svg() -> None:
    lookup = {n.id: n for n in NODES}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        '<defs><marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L12,6 L0,12 z" fill="#1D6F78"/></marker></defs>',
        f'<rect width="{W}" height="{H}" fill="{BG}"/>',
        f'<rect x="40" y="40" width="1520" height="360" rx="20" fill="none" stroke="{REGION_STROKE}" stroke-width="2" stroke-dasharray="12 8"/>',
        f'<rect x="40" y="430" width="1520" height="420" rx="20" fill="none" stroke="{REGION_STROKE}" stroke-width="2" stroke-dasharray="12 8"/>',
        svg_text(70, 86, "阶段一｜G-like 蛋白生成与筛选", 26, DEEP_BLUE, True, "start"),
        svg_text(70, 476, "阶段二｜CDS生成、模板构建与验证", 26, DEEP_BLUE, True, "start"),
    ]

    # Explicit routes reproduce the reference's top chain and bottom fan-in.
    routes = {
        "e01": [(230,190),(270,190)], "e02": [(440,190),(480,190)], "e03": [(650,190),(690,190)],
        "e04": [(875,190),(915,190)], "e05": [(1085,190),(1125,190)], "e06": [(1310,190),(1350,190)],
        "e07": [(1430,235),(1430,258),(1318,258),(1318,275)],
        "e08": [(1318,333),(1318,525)],
        "e09": [(1233,560),(1183,560)], "e10": [(993,560),(933,560)], "e11": [(753,560),(703,560)],
        "e12": [(523,560),(473,560)], "e13": [(280,715),(365,715)],
        "e14": [(388,595),(388,635),(460,635),(460,670)], "e15": [(555,715),(650,715)], "e16": [(840,715),(960,715)],
        "e17": [(1190,715),(1270,715)],
    }
    for eid, _, _, label in EDGES:
        pts = routes[eid]
        pts_s = " ".join(f"{x},{y}" for x, y in pts)
        parts.append(f'<polyline points="{pts_s}" fill="none" stroke="{ARROW}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round" marker-end="url(#arrow)"/>')
    parts.append(svg_text(782.5, 263, "掩码率 5% · 10% · 20% · 30%", 14, DEEP_BLUE, False))

    for node in NODES:
        text_color = "#FFFFFF" if node.fill == MID_TEAL else TEXT
        parts.append(
            f'<rect x="{node.x}" y="{node.y}" width="{node.w}" height="{node.h}" rx="14" '
            f'fill="{node.fill}" stroke="{DEEP_BLUE}" stroke-width="2"/>'
        )
        parts.append(svg_text(node.x + node.w / 2, node.y + node.h / 2 + node.font_size * 0.35, node.label, node.font_size, text_color, node.bold))

    parts.append('</svg>')
    SVG.write_text("\n".join(parts), encoding="utf-8")


def build_static_exports() -> None:
    """Render PNG/PDF from the same explicit geometry used by the draw.io source."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    from matplotlib.font_manager import FontProperties

    font = FontProperties(fname="/System/Library/Fonts/PingFang.ttc")
    routes = {
        "e01": [(230,190),(270,190)], "e02": [(440,190),(480,190)], "e03": [(650,190),(690,190)],
        "e04": [(875,190),(915,190)], "e05": [(1085,190),(1125,190)], "e06": [(1310,190),(1350,190)],
        "e07": [(1430,235),(1430,258),(1318,258),(1318,275)],
        "e08": [(1318,333),(1318,525)],
        "e09": [(1233,560),(1183,560)], "e10": [(993,560),(933,560)], "e11": [(753,560),(703,560)],
        "e12": [(523,560),(473,560)], "e13": [(280,715),(365,715)],
        "e14": [(388,595),(388,635),(460,635),(460,670)], "e15": [(555,715),(650,715)], "e16": [(840,715),(960,715)],
        "e17": [(1190,715),(1270,715)],
    }

    fig, ax = plt.subplots(figsize=(16, 9), dpi=200)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")

    for x, y, w, h in [(40, 40, 1520, 360), (40, 430, 1520, 420)]:
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0,rounding_size=20",
            linewidth=2, edgecolor=REGION_STROKE, facecolor="none",
            linestyle=(0, (8, 6)), zorder=0,
        ))

    ax.text(70, 86, "阶段一｜G-like 蛋白生成与筛选", color=DEEP_BLUE, fontsize=26,
            fontproperties=font, fontweight="bold", va="center", ha="left", zorder=5)
    ax.text(70, 476, "阶段二｜CDS生成、模板构建与验证", color=DEEP_BLUE, fontsize=26,
            fontproperties=font, fontweight="bold", va="center", ha="left", zorder=5)

    for eid, _, _, label in EDGES:
        pts = routes[eid]
        if len(pts) > 2:
            xs, ys = zip(*pts[:-1])
            ax.plot(xs, ys, color=ARROW, linewidth=2.5, solid_joinstyle="round", zorder=1)
        ax.add_patch(FancyArrowPatch(
            pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=18,
            linewidth=2.5, color=ARROW, shrinkA=0, shrinkB=0, zorder=2,
        ))
    ax.text(782.5, 260, "掩码率 5% · 10% · 20% · 30%", color=DEEP_BLUE, fontsize=14,
            fontproperties=font, va="center", ha="center", zorder=5)

    for node in NODES:
        ax.add_patch(FancyBboxPatch(
            (node.x, node.y), node.w, node.h,
            boxstyle="round,pad=0,rounding_size=14",
            linewidth=2, edgecolor=DEEP_BLUE, facecolor=node.fill, zorder=3,
        ))
        text_color = "#FFFFFF" if node.fill == MID_TEAL else TEXT
        ax.text(
            node.x + node.w / 2, node.y + node.h / 2, node.label,
            color=text_color, fontsize=node.font_size, fontproperties=font,
            fontweight="bold" if node.bold else "normal",
            va="center", ha="center", linespacing=1.25, zorder=4,
        )

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(PNG, dpi=200, facecolor=BG, bbox_inches=None, pad_inches=0)
    fig.savefig(PDF, facecolor=BG, bbox_inches=None, pad_inches=0)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    build_drawio()
    build_svg()
    build_static_exports()
    print(DRAWIO)
    print(SVG)
    print(PNG)
    print(PDF)


if __name__ == "__main__":
    main()
