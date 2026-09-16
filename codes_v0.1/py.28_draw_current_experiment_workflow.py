#!/usr/bin/env python3
"""Create an editable Draw.io overview of the current ΦX174 G-like workflow."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path("/Users/yaoruntian/LiLab/phage/phage_designer_v0.1")
OUT_DIR = ROOT / "results" / "current_experiment_workflow"
OUT_FILE = OUT_DIR / "current_experiment_workflow.drawio"

FONT = "Arial"
TEXT = "#203040"
EDGE = "#263238"


def add_vertex(
    root: ET.Element,
    cell_id: str,
    value: str,
    x: float,
    y: float,
    width: float,
    height: float,
    style: str,
    *,
    parent: str = "1",
) -> ET.Element:
    cell = ET.SubElement(
        root,
        "mxCell",
        {"id": cell_id, "value": value, "style": style, "vertex": "1", "parent": parent},
    )
    ET.SubElement(
        cell,
        "mxGeometry",
        {"x": str(x), "y": str(y), "width": str(width), "height": str(height), "as": "geometry"},
    )
    return cell


def add_edge(
    root: ET.Element,
    cell_id: str,
    source: str,
    target: str,
    value: str,
    *,
    dashed: bool = False,
    parent: str = "1",
    exit_xy: tuple[float, float] = (1.0, 0.5),
    entry_xy: tuple[float, float] = (0.0, 0.5),
    points: list[tuple[float, float]] | None = None,
) -> ET.Element:
    dash = "dashed=1;dashPattern=7 5;" if dashed else ""
    style = (
        "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;"
        "endArrow=block;endFill=1;strokeWidth=2.2;"
        f"strokeColor={EDGE};fontColor={TEXT};fontSize=13;fontFamily={FONT};labelBackgroundColor=#FFFFFF;"
        f"exitX={exit_xy[0]};exitY={exit_xy[1]};exitDx=0;exitDy=0;"
        f"entryX={entry_xy[0]};entryY={entry_xy[1]};entryDx=0;entryDy=0;{dash}"
    )
    cell = ET.SubElement(
        root,
        "mxCell",
        {
            "id": cell_id,
            "value": value,
            "style": style,
            "edge": "1",
            "parent": parent,
            "source": source,
            "target": target,
        },
    )
    geometry = ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})
    if points:
        array = ET.SubElement(geometry, "Array", {"as": "points"})
        for x, y in points:
            ET.SubElement(array, "mxPoint", {"x": str(x), "y": str(y)})
    return cell


def card_style(fill: str, stroke: str, *, accent: bool = False) -> str:
    sw = 2.8 if accent else 2.0
    return (
        "rounded=1;whiteSpace=wrap;html=1;arcSize=14;align=left;verticalAlign=top;"
        "spacingTop=16;spacingLeft=16;spacingRight=14;spacingBottom=12;"
        f"fillColor={fill};strokeColor={stroke};strokeWidth={sw};fontColor={TEXT};"
        f"fontFamily={FONT};fontSize=15;shadow=0;"
    )


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    mxfile = ET.Element(
        "mxfile",
        {"host": "app.diagrams.net", "agent": "Codex", "version": "26.0.9", "pages": "1"},
    )
    diagram = ET.SubElement(mxfile, "diagram", {"id": "phix174-current-workflow", "name": "Current workflow"})
    model = ET.SubElement(
        diagram,
        "mxGraphModel",
        {
            "dx": "2200",
            "dy": "1100",
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
            "pageHeight": "1100",
            "math": "0",
            "shadow": "0",
        },
    )
    root = ET.SubElement(model, "root")
    ET.SubElement(root, "mxCell", {"id": "0"})
    ET.SubElement(root, "mxCell", {"id": "1", "parent": "0"})

    title_style = (
        "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;"
        f"fontFamily={FONT};fontSize=30;fontStyle=1;fontColor=#172B3A;"
    )
    add_vertex(
        root,
        "figure_title",
        "ΦX174 G-like蛋白引导的噬菌体设计与多层计算验证",
        250,
        26,
        1700,
        48,
        title_style,
    )
    add_vertex(
        root,
        "figure_subtitle",
        "当前已完成流程：蛋白显式生成 → 合法CDS设计 → 模板基因组替换 → 计算证据整合",
        390,
        76,
        1420,
        30,
        title_style.replace("fontSize=30", "fontSize=16").replace("fontStyle=1", "fontStyle=0").replace("#172B3A", "#526575"),
    )

    stages = [
        (
            "stage_input",
            "<b>01 · 参考输入</b><br><br>ΦX174参考基因组<br><b>5,386 nt</b><br><br>天然G刺突蛋白<br><b>175 aa</b>",
            50,
            150,
            270,
            230,
            "#E8F2F5",
            "#63758A",
            False,
        ),
        (
            "stage_esm2",
            "<b>02 · G-like蛋白生成</b><br><br><b>ESM2-650M masked-LM</b><br>掩码率 5% / 10% / 20% / 30%<br>每组750条<br><br><b>3,000条唯一序列</b>",
            380,
            150,
            300,
            230,
            "#F1D7D4",
            "#B44948",
            True,
        ),
        (
            "stage_screen",
            "<b>03 · 独立蛋白筛选</b><br><br>完整性 · MMseqs2<br>PHROG-1483 · 74个保守位点<br>ESMFold单体结构<br><br><b>3,000 → 1,856 → 473</b><br>最终保留率 <b>15.77%</b>",
            740,
            150,
            310,
            230,
            "#EAF0F6",
            "#63758A",
            False,
        ),
        (
            "stage_rank",
            "<b>04 · 质量与多样性排序</b><br><br>序列质量 + 结构相似性<br>保守性 + 新颖度<br><br>候选池 <b>473 → 100</b><br>分层选取 <b>10条蛋白</b>",
            1110,
            150,
            280,
            230,
            "#EDE9F4",
            "#7B6A9A",
            False,
        ),
        (
            "stage_reverse",
            "<b>05 · Evo2受约束反向翻译</b><br><br>仅在同义密码子空间采样<br>保持未突变位点原始密码子<br>Evo2进行上下文似然排序<br><br><b>蛋白翻译100%一致</b>",
            1450,
            150,
            310,
            230,
            "#F1D7D4",
            "#B44948",
            True,
        ),
        (
            "stage_replace",
            "<b>06 · ΦX174模板替换</b><br><br>仅替换G CDS<br>其余模板序列保持不变<br><br><b>10条完整基因组</b><br>每条 <b>5,386 nt</b>",
            1820,
            150,
            300,
            230,
            "#EAF0F6",
            "#63758A",
            False,
        ),
    ]
    for sid, label, x, y, w, h, fill, stroke, accent in stages:
        add_vertex(root, sid, label, x, y, w, h, card_style(fill, stroke, accent=accent))

    add_edge(root, "edge_01", "stage_input", "stage_esm2", "参考G")
    add_edge(root, "edge_02", "stage_esm2", "stage_screen", "3,000条")
    add_edge(root, "edge_03", "stage_screen", "stage_rank", "473条")
    add_edge(root, "edge_04", "stage_rank", "stage_reverse", "10条蛋白")
    add_edge(root, "edge_05", "stage_reverse", "stage_replace", "10条CDS")

    group_style = (
        "rounded=1;dashed=1;dashPattern=8 6;whiteSpace=wrap;html=1;container=1;"
        "align=left;verticalAlign=top;spacingTop=13;spacingLeft=18;"
        "fillColor=#FFFFFF;strokeColor=#63758A;strokeWidth=2;"
        f"fontFamily={FONT};fontSize=20;fontStyle=1;fontColor=#263746;"
    )
    add_vertex(
        root,
        "validation_group",
        "并行多层计算验证（10条候选）",
        60,
        485,
        2080,
        355,
        group_style,
    )

    validation_cards = [
        (
            "val_seq",
            "<b>A · 序列与功能证据</b><br><br>合法ORF与精确翻译<br>MMseqs2 identity / coverage<br>PHROG-1483功能域<br>保守位点保持率<br><br><b>10/10通过基础序列约束</b>",
            45,
            70,
            450,
            230,
            "#F4EEDC",
            "#9A7B3F",
        ),
        (
            "val_structure",
            "<b>B · 结构与装配代理</b><br><br>ESMFold单体 · G5 Boltz-2<br>TM-align至天然36CQ<br>ipTM · PAE · 界面接触<br><br><font color=\"#A64034\"><b>F5/G5/J5：单卡5090 OOM</b><br>当前F1/G1/J1仅为局部代理</font>",
            555,
            70,
            450,
            230,
            "#FFF4F2",
            "#C65D4B",
        ),
        (
            "val_rna",
            "<b>C · G起始区mRNA</b><br><br>ViennaRNA，240 nt窗口<br>ΔMFE相对ΦX174<br>起始区平均可及性变化<br>碱基配对结构距离<br><br><b>评估翻译起始环境变化</b>",
            1065,
            70,
            450,
            230,
            "#EAF0F6",
            "#63758A",
        ),
        (
            "val_genome",
            "<b>D · 全基因组环境</b><br><br>Microviridae微调Evo2-7B<br>全基因组平均log-likelihood/nt<br>相对ΦX174及119条天然变体<br>z-score与经验百分位<br><br><b>评估模板上下文自然性</b>",
            1575,
            70,
            450,
            230,
            "#EAF0F6",
            "#63758A",
        ),
    ]
    for sid, label, x, y, w, h, fill, stroke in validation_cards:
        add_vertex(root, sid, label, x, y, w, h, card_style(fill, stroke), parent="validation_group")

    add_edge(
        root,
        "edge_06",
        "stage_replace",
        "validation_group",
        "完整基因组 + G蛋白",
        exit_xy=(0.5, 1.0),
        entry_xy=(0.88, 0.0),
    )

    output_style = (
        "rounded=1;whiteSpace=wrap;html=1;arcSize=14;align=center;verticalAlign=middle;"
        "spacing=12;fillColor=#E5F1E3;strokeColor=#5A8A55;strokeWidth=2.8;"
        f"fontColor={TEXT};fontFamily={FONT};fontSize=17;"
    )
    add_vertex(
        root,
        "integrated_output",
        "<b>综合证据输出</b>　10/10记录完整 · 61字段证据表 · 候选优先级与风险分层<br>"
        "<font color=\"#456B42\"><b>仅用于候选排序；不标记“viable”，不声称感染性、装配能力或实验可行性</b></font>",
        430,
        915,
        1340,
        100,
        output_style,
    )
    add_edge(
        root,
        "edge_07",
        "validation_group",
        "integrated_output",
        "整合四类证据",
        exit_xy=(0.5, 1.0),
        entry_xy=(0.5, 0.0),
    )

    legend_style = (
        "rounded=1;whiteSpace=wrap;html=1;arcSize=12;align=left;verticalAlign=middle;"
        "spacingLeft=14;fillColor=#FFFFFF;strokeColor=#C5CED6;strokeWidth=1.5;"
        f"fontColor=#435463;fontFamily={FONT};fontSize=13;"
    )
    add_vertex(
        root,
        "legend",
        "<b>图例</b><br>红色边框：蛋白→DNA接口核心步骤<br>虚线边框：并行验证组<br>橙红卡片：资源限制/替代分析<br>实线箭头：数据流",
        1810,
        900,
        310,
        130,
        legend_style,
    )

    tree = ET.ElementTree(mxfile)
    ET.indent(tree, space="  ")
    tree.write(OUT_FILE, encoding="utf-8", xml_declaration=True)
    print(OUT_FILE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
