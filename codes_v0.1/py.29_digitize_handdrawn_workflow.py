#!/usr/bin/env python3
"""Digitize IMG_3487 into an editable Draw.io scientific workflow."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path("/Users/yaoruntian/LiLab/phage/phage_designer_v0.1")
OUT_DIR = ROOT / "results" / "handdrawn_workflow_digitized"
OUT_FILE = OUT_DIR / "phage_design_workflow_digitized.drawio"

FONT = "Arial"
TEXT = "#203040"
MUTED = "#6B7C8C"
NAVY = "#163179"
TEAL = "#0BDACD"
PALE = "#EEFFE6"
PALE_TEAL = "#D9F7F2"
WHITE = "#FFFFFF"
WARM = "#FFF4E8"


def add_vertex(root, cid, value, x, y, w, h, style, parent="1"):
    cell = ET.SubElement(
        root,
        "mxCell",
        {"id": cid, "value": value, "style": style, "vertex": "1", "parent": parent},
    )
    ET.SubElement(
        cell,
        "mxGeometry",
        {"x": str(x), "y": str(y), "width": str(w), "height": str(h), "as": "geometry"},
    )
    return cell


def add_edge(
    root,
    cid,
    source,
    target,
    label,
    *,
    dashed=False,
    exit_xy=(1.0, 0.5),
    entry_xy=(0.0, 0.5),
    points=None,
):
    dash = "dashed=1;dashPattern=8 6;" if dashed else ""
    style = (
        "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;"
        "endArrow=block;endFill=1;endSize=10;strokeWidth=2.4;"
        f"strokeColor={NAVY};fontColor={TEXT};fontSize=13;fontFamily={FONT};"
        f"labelBackgroundColor={WHITE};exitX={exit_xy[0]};exitY={exit_xy[1]};"
        f"entryX={entry_xy[0]};entryY={entry_xy[1]};{dash}"
    )
    cell = ET.SubElement(
        root,
        "mxCell",
        {
            "id": cid,
            "value": label,
            "style": style,
            "edge": "1",
            "parent": "1",
            "source": source,
            "target": target,
        },
    )
    geo = ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})
    if points:
        arr = ET.SubElement(geo, "Array", {"as": "points"})
        for px, py in points:
            ET.SubElement(arr, "mxPoint", {"x": str(px), "y": str(py)})
    return cell


def card(fill, stroke=NAVY, *, align="center", key=False, font_size=15):
    return (
        "rounded=1;whiteSpace=wrap;html=1;arcSize=16;"
        f"align={align};verticalAlign=middle;spacing=14;"
        f"fillColor={fill};strokeColor={stroke};strokeWidth={3 if key else 2};"
        f"fontColor={TEXT};fontFamily={FONT};fontSize={font_size};shadow=0;"
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mxfile = ET.Element(
        "mxfile",
        {"host": "app.diagrams.net", "agent": "Codex", "version": "26.0.9", "pages": "1"},
    )
    diagram = ET.SubElement(mxfile, "diagram", {"id": "digitized-phage-workflow", "name": "Workflow"})
    model = ET.SubElement(
        diagram,
        "mxGraphModel",
        {
            "dx": "2200",
            "dy": "1240",
            "grid": "1",
            "gridSize": "10",
            "guides": "1",
            "tooltips": "1",
            "connect": "1",
            "arrows": "1",
            "fold": "1",
            "page": "1",
            "pageScale": "1",
            "pageWidth": "2200",
            "pageHeight": "1240",
            "math": "0",
            "shadow": "0",
        },
    )
    root = ET.SubElement(model, "root")
    ET.SubElement(root, "mxCell", {"id": "0"})
    ET.SubElement(root, "mxCell", {"id": "1", "parent": "0"})

    text_title = (
        "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;"
        f"fontFamily={FONT};fontSize=28;fontStyle=1;fontColor={NAVY};"
    )
    add_vertex(
        root,
        "title",
        "基于蛋白功能与 ΦX174 模板的噬菌体序列设计流程",
        220,
        25,
        1760,
        52,
        text_title,
    )
    add_vertex(
        root,
        "subtitle",
        "蛋白功能控制 → G-like 蛋白生成与筛选 → 同义 CDS 设计 → 模板组合 → 分层验证",
        360,
        76,
        1480,
        30,
        text_title.replace("fontSize=28", "fontSize=15").replace("fontStyle=1", "fontStyle=0").replace(NAVY, MUTED),
    )

    lane_style = (
        "text;html=1;strokeColor=none;fillColor=none;whiteSpace=wrap;overflow=visible;"
        "align=left;verticalAlign=middle;"
        f"fontColor={NAVY};fontFamily={FONT};fontSize=21;fontStyle=1;"
    )
    add_vertex(root, "protein_lane", "A · G-like 蛋白生成与筛选", 65, 125, 520, 38, lane_style)
    add_vertex(root, "dna_lane", "B · CDS 设计与模板组合", 65, 630, 520, 38, lane_style)

    # Protein generation and screening lane.
    add_vertex(root, "function_input", "<b>蛋白功能输入</b><br><br>示例：<i>spike G protein</i>", 85, 255, 180, 125, card(PALE))
    add_vertex(root, "retrieval", "<b>功能检索</b><br><br>检索同功能的<br>天然参考蛋白", 315, 255, 205, 125, card(PALE_TEAL))
    add_vertex(root, "reference_protein", "<b>真实参考序列</b><br><br>ΦX174 G protein<br><b>175 aa</b>", 570, 255, 210, 125, card(PALE))

    esm_label = (
        "<b><font color=\"#163179\">ESM2-650M masked-LM</font></b><br><br>"
        "掩码并重采样参考序列<br>"
        "<table cellpadding=\"4\" cellspacing=\"2\" border=\"0\" width=\"100%\">"
        "<tr><td><b>5%</b></td><td>750条</td></tr>"
        "<tr><td><b>10%</b></td><td>750条</td></tr>"
        "<tr><td><b>20%</b></td><td>750条</td></tr>"
        "<tr><td><b>30%</b></td><td>750条</td></tr>"
        "</table>"
    )
    add_vertex(root, "esm2_generator", esm_label, 830, 195, 300, 245, card(PALE_TEAL, TEAL, key=True, font_size=14))
    add_vertex(
        root,
        "generated_pool",
        "<b>G-like 蛋白候选</b><br><br><font color=\"#163179\"><b>3,000条</b></font><br>175 aa · 序列唯一<br>5/10/20/30% 四组",
        1180,
        235,
        230,
        165,
        card(PALE),
    )

    screen_label = (
        "<b>独立序列与结构筛选</b><br><br>完整性 · MMseqs2<br>PHROG/HMM · 保守位点<br>ESMFold · TM-like / RMSD<br><br>"
        "<table cellpadding=\"3\" cellspacing=\"1\" border=\"0\" width=\"100%\">"
        "<tr><td>5%</td><td><b>334</b></td><td>10%</td><td><b>125</b></td></tr>"
        "<tr><td>20%</td><td><b>14</b></td><td>30%</td><td><b>0</b></td></tr>"
        "</table>"
    )
    add_vertex(root, "filter_stack", screen_label, 1460, 195, 290, 245, card(PALE_TEAL, NAVY, font_size=14))
    add_vertex(
        root,
        "ranking",
        "<b>综合排序与分层选择</b><br><br><b>473</b> 条通过结构筛选（15.77%）<br>↓<br><b>Top 100</b> 质量与多样性候选<br>↓<br><b>10</b> 条分层候选",
        1800,
        210,
        270,
        215,
        card(PALE, TEAL, key=True, font_size=15),
    )

    add_edge(root, "e01", "function_input", "retrieval", "功能文本")
    add_edge(root, "e02", "retrieval", "reference_protein", "天然同源蛋白")
    add_edge(root, "e03", "reference_protein", "esm2_generator", "参考序列")
    add_edge(root, "e04", "esm2_generator", "generated_pool", "4 × 750")
    add_edge(root, "e05", "generated_pool", "filter_stack", "3,000条")
    add_edge(root, "e06", "filter_stack", "ranking", "473条")

    # Genome / CDS lane. Template branch flows left-to-right.
    add_vertex(root, "template_input", "<b>ΦX174 模板 prompt</b><br><br>参考基因组骨架<br>5,386 nt", 90, 725, 220, 120, card(PALE))
    add_vertex(root, "evo2_backbone", "<b>Evo2</b><br><br>模板条件下的<br>基因组骨架生成", 365, 715, 250, 140, card(PALE_TEAL, NAVY, key=True))
    add_vertex(root, "backbone_pool", "<b>ΦX174-like 骨架</b><br><br>seq 1 … seq 10<br><b>10条</b>", 670, 725, 225, 120, card(PALE))

    # Protein-to-CDS branch intentionally runs right-to-left, matching the sketch.
    add_vertex(root, "selected_proteins", "<b>10条 G-like 蛋白</b><br><br>来自 Top 100<br>分层抽样", 1810, 925, 230, 115, card(PALE))
    add_vertex(root, "reverse_translate", "<b>Evo2 受约束反向翻译</b><br><br>每条蛋白生成10个<br>合法同义 CDS", 1495, 910, 260, 145, card(PALE_TEAL, NAVY, key=True, font_size=14))
    add_vertex(root, "cds_pool", "<b>100条候选 CDS</b><br><br>10 × 10<br>翻译结果严格一致", 1205, 925, 235, 115, card(PALE))

    add_vertex(
        root,
        "combine",
        "<b>骨架 × G CDS</b><br><br>组合 / 嵌入<br><font color=\"#163179\"><b>10 × 10</b></font><br>保持蛋白翻译一致",
        955,
        760,
        210,
        205,
        card("#C9F7EF", TEAL, key=True),
    )
    add_vertex(root, "candidate_genomes", "<b>候选完整基因组</b><br><br>10个骨架 × 10条CDS<br><b>100条</b>", 1225, 735, 235, 125, card(PALE))
    add_vertex(
        root,
        "computational_evidence",
        "<b>计算证据整合</b><br><br>序列与功能 · 结构与界面<br>mRNA起始区 · Evo2似然<br><b>候选优先级排序</b>",
        1515,
        715,
        275,
        165,
        card(PALE_TEAL),
    )
    add_vertex(
        root,
        "wet_lab",
        "<b>湿实验验证</b><br><br>合成 · 表达 · 装配<br>感染性与宿主范围",
        1850,
        735,
        215,
        125,
        card(WARM, "#B8752A"),
    )

    add_edge(root, "e07", "template_input", "evo2_backbone", "模板条件")
    add_edge(root, "e08", "evo2_backbone", "backbone_pool", "生成")
    add_edge(root, "e09", "backbone_pool", "combine", "10条骨架", entry_xy=(0.0, 0.35))
    add_edge(
        root,
        "e10",
        "ranking",
        "selected_proteins",
        "分层选择",
        exit_xy=(0.85, 1.0),
        entry_xy=(0.85, 0.0),
        points=[(2100, 590), (2100, 895), (2045, 895)],
    )
    add_edge(root, "e11", "selected_proteins", "reverse_translate", "蛋白序列", exit_xy=(0.0, 0.5), entry_xy=(1.0, 0.5))
    add_edge(root, "e12", "reverse_translate", "cds_pool", "100条CDS", exit_xy=(0.0, 0.5), entry_xy=(1.0, 0.5))
    add_edge(root, "e13", "cds_pool", "combine", "同义CDS", exit_xy=(0.0, 0.5), entry_xy=(1.0, 0.78))
    add_edge(root, "e14", "combine", "candidate_genomes", "10×10组合", exit_xy=(1.0, 0.35), entry_xy=(0.0, 0.5))
    add_edge(root, "e15", "candidate_genomes", "computational_evidence", "100条候选")
    add_edge(root, "e16", "computational_evidence", "wet_lab", "需实验验证", dashed=True)

    note_style = (
        "rounded=1;whiteSpace=wrap;html=1;arcSize=12;align=center;verticalAlign=middle;"
        f"fillColor={WHITE};strokeColor=#9EB1C0;strokeWidth=1.5;fontColor={MUTED};"
        f"fontFamily={FONT};fontSize=14;"
    )
    add_vertex(
        root,
        "boundary_note",
        "<b>解释边界：</b>计算流程仅用于生成、筛选和候选优先级排序；不能单独证明噬菌体可感染、可装配或具有实验可行性。",
        300,
        1105,
        1600,
        65,
        note_style,
    )

    legend_style = note_style.replace("align=center", "align=left").replace("fontSize=14", "fontSize=13")
    add_vertex(
        root,
        "legend",
        "<b>图例</b>　浅绿：输入/候选　　浅青：模型与筛选　　青色边框：核心接口　　实线：计算数据流　　虚线：进入实验验证",
        350,
        1180,
        1500,
        42,
        legend_style,
    )

    tree = ET.ElementTree(mxfile)
    ET.indent(tree, space="  ")
    tree.write(OUT_FILE, encoding="utf-8", xml_declaration=True)
    print(OUT_FILE)


if __name__ == "__main__":
    main()
