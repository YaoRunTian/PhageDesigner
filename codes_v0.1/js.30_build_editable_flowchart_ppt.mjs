import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const SKILL_DIR = "/Users/yaoruntian/.codex/plugins/cache/openai-primary-runtime/presentations/26.905.11957/skills/presentations";
const RUNTIME_PYTHON = "/Users/yaoruntian/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3";
const workspaceDir = "/Users/yaoruntian/LiLab/phage/phage_designer_v0.1/results/current_experiment_simple_flowchart";
const TMP_DIR = path.join(workspaceDir, "ppt_live_build");
const FINAL_DIR = path.join(workspaceDir, "editable_ppt");
const FINAL_PPTX = path.join(FINAL_DIR, "current_experiment_editable_flowchart_v2.pptx");

const { finalizePresentation } = await import(
  pathToFileURL(path.join(SKILL_DIR, "container_tools/artifact_tool_utils.mjs")).href,
);

await fs.mkdir(TMP_DIR, { recursive: true });
await fs.mkdir(FINAL_DIR, { recursive: true });

const W = 1280;
const H = 720;
const FONT = "PingFang SC";
const BG = "#FFFFFF";
const NAVY = "#163179";
const TEXT = "#17324D";
const TEAL = "#0BDACD";
const ARROW = "#1D6F78";
const MINT = "#EEFFE6";
const CYAN = "#E8FAF7";
const BORDER = "#A9D7CB";
const ORANGE = "#FF8A3D";
const BLUE = "#3D8BFF";
const GREEN = "#22A06B";

const p = Presentation.create({ slideSize: { width: W, height: H } });
const slide = p.slides.add();
slide.background.fill = BG;

function addShape(geometry, left, top, width, height, fill = "none", lineFill = NAVY, lineWidth = 1.8, name = undefined) {
  return slide.shapes.add({
    geometry,
    name,
    position: { left, top, width, height },
    fill,
    line: { style: "solid", fill: lineFill, width: lineWidth },
    shadow: "shadow-none",
  });
}

function addText(left, top, width, height, text, size = 18, bold = false, color = TEXT, alignment = "center", name = undefined) {
  const box = slide.shapes.add({
    geometry: "textbox",
    name,
    position: { left, top, width, height },
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  box.text = text;
  box.text.style = {
    typeface: FONT,
    fontSize: size,
    bold,
    color,
    alignment,
    verticalAlignment: "middle",
    autoFit: "shrinkText",
    insets: { top: 2, bottom: 2, left: 4, right: 4 },
  };
  return box;
}

function addBox(id, left, top, width, height, label, fill, bold = false, color = TEXT, size = 18) {
  const box = addShape("roundRect", left, top, width, height, fill, NAVY, 1.8, id);
  box.text = label;
  box.text.style = {
    typeface: FONT,
    fontSize: size,
    bold,
    color,
    alignment: "center",
    verticalAlignment: "middle",
    autoFit: "shrinkText",
    insets: { top: 5, bottom: 5, left: 6, right: 6 },
  };
  return box;
}

function addLine(x1, y1, x2, y2, color = NAVY, width = 1.8, name = undefined) {
  const left = Math.min(x1, x2);
  const top = Math.min(y1, y2);
  return slide.shapes.add({
    geometry: "line",
    name,
    position: {
      left,
      top,
      width: Math.abs(x2 - x1),
      height: Math.abs(y2 - y1),
      horizontalFlip: x2 < x1,
      verticalFlip: y2 < y1,
    },
    fill: "none",
    line: { style: "solid", fill: color, width },
  });
}

function connect(a, b, fromSide, toSide, kind = "straight") {
  return slide.shapes.connect(a, b, {
    kind,
    fromSide,
    toSide,
    line: { style: "solid", fill: ARROW, width: 2 },
    tail: { type: "triangle", width: "sm", length: "sm" },
  });
}

function addProteinIcon(x, y, scale = 1, name = "protein") {
  const pts = [[0,20],[12,5],[27,17],[42,2],[56,18],[70,7]];
  for (let i = 0; i < pts.length - 1; i++) {
    addLine(x + pts[i][0] * scale, y + pts[i][1] * scale, x + pts[i+1][0] * scale, y + pts[i+1][1] * scale, NAVY, 2, `${name}-bond-${i}`);
  }
  const colors = [TEAL, BLUE, TEAL, ORANGE, TEAL, BLUE];
  pts.forEach((pt, i) => addShape("ellipse", x + (pt[0]-4)*scale, y + (pt[1]-4)*scale, 8*scale, 8*scale, colors[i], NAVY, 1, `${name}-aa-${i}`));
}

function addDatabaseIcon(x, y) {
  // Build the database stack from export-safe native shapes so every part
  // remains editable in PowerPoint.
  addShape("rect", x, y + 6, 42, 34, CYAN, NAVY, 1.6, "icon-database-body");
  addShape("ellipse", x, y, 42, 13, MINT, NAVY, 1.6, "icon-database-top");
  addShape("arc", x, y + 13, 42, 13, "none", NAVY, 1.2, "icon-database-row-1");
  addShape("arc", x, y + 26, 42, 13, "none", NAVY, 1.2, "icon-database-row-2");
  addShape("ellipse", x + 30, y + 25, 24, 24, "none", NAVY, 2, "icon-search-lens");
  addLine(x + 49, y + 44, x + 61, y + 56, NAVY, 3, "icon-search-handle");
}

function addNetworkIcon(x, y) {
  const cols = [0, 26, 52];
  const rows = [0, 18, 36];
  for (const a of rows) for (const b of rows) {
    addLine(x + cols[0] + 5, y + a + 5, x + cols[1] + 5, y + b + 5, "#5A91C7", 0.8);
    addLine(x + cols[1] + 5, y + a + 5, x + cols[2] + 5, y + b + 5, "#5A91C7", 0.8);
  }
  cols.forEach((cx, ci) => rows.forEach((ry, ri) => addShape("ellipse", x + cx, y + ry, 10, 10, ci === 1 ? TEAL : BLUE, NAVY, 0.8, `icon-net-${ci}-${ri}`)));
  ["M", "A", "MASK", "T"].forEach((t, i) => {
    const w = t === "MASK" ? 34 : 22;
    const bx = x - 3 + i * 27;
    const tile = addBox(`token-${i}`, bx, y - 28, w, 20, t, CYAN, true, NAVY, 9);
    tile.line = { style: "solid", fill: "#9EC9EE", width: 0.8 };
  });
}

function addFunnelIcon(x, y) {
  addShape("trapezoid", x, y, 64, 34, CYAN, NAVY, 1.6, "icon-funnel-top");
  addShape("rect", x + 27, y + 32, 10, 18, TEAL, NAVY, 1.2, "icon-funnel-stem");
}

function addClipboardIcon(x, y) {
  addShape("roundRect", x, y + 5, 46, 52, CYAN, NAVY, 1.6, "icon-clipboard");
  addShape("roundRect", x + 13, y, 20, 10, MINT, NAVY, 1.2, "icon-clipboard-tab");
  [0,1,2].forEach(i => {
    const yy = y + 17 + i * 12;
    addLine(x + 8, yy + 3, x + 13, yy + 8, GREEN, 2);
    addLine(x + 13, yy + 8, x + 20, yy, GREEN, 2);
    addLine(x + 25, yy + 4, x + 39, yy + 4, NAVY, 1.2);
  });
}

function addDnaIcon(x, y, scale = 1, name = "dna") {
  const left = [[0,0],[10,8],[0,16],[10,24],[0,32],[10,40]];
  const right = left.map(([px,py]) => [28-px, py]);
  for (let i=0;i<left.length-1;i++) {
    addLine(x+left[i][0]*scale, y+left[i][1]*scale, x+left[i+1][0]*scale, y+left[i+1][1]*scale, BLUE, 1.7, `${name}-l-${i}`);
    addLine(x+right[i][0]*scale, y+right[i][1]*scale, x+right[i+1][0]*scale, y+right[i+1][1]*scale, TEAL, 1.7, `${name}-r-${i}`);
  }
  [4,12,20,28,36].forEach((yy,i) => addLine(x+5*scale, y+yy*scale, x+23*scale, y+yy*scale, i===2?ORANGE:NAVY, 1, `${name}-rung-${i}`));
}

function addChartIcon(x, y) {
  addLine(x, y+42, x, y, NAVY, 1.5);
  addLine(x, y+42, x+54, y+42, NAVY, 1.5);
  const pts = [[4,35],[16,28],[28,31],[40,14],[53,8]];
  for (let i=0;i<pts.length-1;i++) addLine(x+pts[i][0],y+pts[i][1],x+pts[i+1][0],y+pts[i+1][1],BLUE,2);
}

function addGenomeIcon(x, y, scale = 1, name = "genome") {
  addShape("ellipse", x, y, 46*scale, 46*scale, "none", NAVY, 2, `${name}-ring`);
  addShape("arc", x+1*scale, y+1*scale, 44*scale, 44*scale, "none", ORANGE, 4, `${name}-segment`);
}

function addPhageIcon(x, y) {
  addShape("hexagon", x+16, y, 36, 34, CYAN, NAVY, 1.5, "icon-phage-head");
  addLine(x+34,y+34,x+34,y+74,NAVY,2,"icon-phage-tail");
  addLine(x+22,y+47,x+46,y+47,NAVY,1.5,"icon-phage-cross");
  addLine(x+34,y+64,x+10,y+84,NAVY,2,"icon-phage-leg1");
  addLine(x+34,y+64,x+58,y+84,NAVY,2,"icon-phage-leg2");
  addLine(x+34,y+64,x+22,y+86,NAVY,1.5,"icon-phage-leg3");
  addLine(x+34,y+64,x+46,y+86,NAVY,1.5,"icon-phage-leg4");
}

function addGridIcon(x, y) {
  for (let r=0;r<3;r++) for (let c=0;c<3;c++) addShape("rect",x+c*11,y+r*11,9,9,(r+c)%2?CYAN:BLUE,NAVY,0.4,`grid-${r}-${c}`);
}

// Stage containers and titles.
const region1 = slide.shapes.add({ geometry: "roundRect", name: "stage-1-region", position: { left: 28, top: 18, width: 1224, height: 315 }, fill: "none", line: { style: "dashed", fill: BORDER, width: 1.5 } });
const region2 = slide.shapes.add({ geometry: "roundRect", name: "stage-2-region", position: { left: 28, top: 348, width: 1224, height: 345 }, fill: "none", line: { style: "dashed", fill: BORDER, width: 1.5 } });
region1.sendToBack(); region2.sendToBack();
addText(55, 32, 520, 40, "阶段一  G-like 蛋白生成与筛选", 30, true, NAVY, "left", "stage-1-title");
addText(55, 365, 600, 40, "阶段二  CDS生成、模板构建与验证", 30, true, NAVY, "left", "stage-2-title");

// Stage 1 boxes.
const s1Input = addBox("protein-function", 58, 176, 120, 52, "蛋白功能", MINT, true);
const s1Ref = addBox("reference-search", 211, 176, 132, 52, "参考蛋白检索", CYAN, false);
const s1G = addBox("phix174-g-protein", 376, 176, 126, 52, "ΦX174 G 蛋白", MINT, true);
const s1Esm = addBox("esm2-sampling", 535, 168, 148, 68, "ESM2-650M\n掩码采样", TEAL, true, "#FFFFFF", 18);
const s1Gen = addBox("g-like-proteins", 716, 168, 132, 68, "3,000 条\nG-like 蛋白", CYAN, true);
const s1Filter = addBox("protein-screen", 881, 176, 140, 52, "多层蛋白筛选", MINT, true);
const s1Rank = addBox("integrated-rank", 1054, 176, 128, 52, "综合排序", CYAN, true);
const s1Counts = addBox("ranked-counts", 943, 264, 239, 44, "473 条  →  100 条  →  10 条", MINT, true, TEXT, 16);
addText(535, 236, 148, 26, "掩码率 5% · 10% · 20% · 30%", 12, false, NAVY, "center", "mask-rates");

connect(s1Input,s1Ref,"right","left");
connect(s1Ref,s1G,"right","left");
connect(s1G,s1Esm,"right","left");
connect(s1Esm,s1Gen,"right","left");
connect(s1Gen,s1Filter,"right","left");
connect(s1Filter,s1Rank,"right","left");
connect(s1Rank,s1Counts,"bottom","top","elbow");

// Stage 1 editable flat icons.
addProteinIcon(82, 108, 0.75, "icon-function-protein");
addDatabaseIcon(245, 103);
addProteinIcon(404, 108, 0.75, "icon-g-protein");
addNetworkIcon(579, 112);
addProteinIcon(748, 110, 0.45, "icon-generated-protein-a");
addProteinIcon(785, 110, 0.45, "icon-generated-protein-b");
addFunnelIcon(919, 106);
addClipboardIcon(1093, 102);

// Stage 2 top branch.
const s2Protein = addBox("candidate-proteins", 1000, 483, 140, 52, "10 条候选蛋白", MINT, true);
const s2Rt = addBox("evo2-reverse-translation", 821, 475, 150, 68, "Evo2 约束\n反向翻译", TEAL, true, "#FFFFFF");
const s2Cds100 = addBox("valid-cds-100", 647, 483, 140, 52, "100 条合法 CDS", CYAN, false);
const s2Score = addBox("evo2-likelihood", 470, 475, 142, 68, "Evo2 似然排序", MINT, true);
const s2Cds10 = addBox("selected-cds-10", 296, 483, 140, 52, "10 条优选 CDS", CYAN, true);
connect(s1Counts,s2Protein,"bottom","top","elbow");
connect(s2Protein,s2Rt,"left","right");
connect(s2Rt,s2Cds100,"left","right");
connect(s2Cds100,s2Score,"left","right");
connect(s2Score,s2Cds10,"left","right");

addProteinIcon(1032, 425, 0.62, "icon-candidate-protein");
addDnaIcon(872, 416, 1.0, "icon-rt-dna");
addText(905, 416, 60, 18, "ATG  GAA", 9, true, NAVY, "center", "codon-tiles");
addDnaIcon(699, 422, 0.9, "icon-valid-cds");
addChartIcon(516, 416);
addDnaIcon(330, 420, 0.7, "icon-cds-a");
addDnaIcon(358, 420, 0.7, "icon-cds-b");
addShape("diamond", 410, 428, 20, 20, ORANGE, ORANGE, 1, "icon-quality-star");

// Stage 2 merge and output branch.
const template = addBox("phix174-template", 54, 618, 166, 46, "ΦX174 模板基因组", MINT, true, TEXT, 16);
const merge = addBox("replace-g-cds", 285, 607, 150, 66, "替换 G CDS", TEAL, true, "#FFFFFF");
const genomes = addBox("designed-genomes", 512, 607, 150, 66, "10 条设计基因组", CYAN, true);
const validate = addBox("computational-validation", 744, 607, 170, 66, "多层计算验证", MINT, true);
const evidence = addBox("evidence-table", 986, 607, 166, 66, "综合证据表", CYAN, true);
connect(template,merge,"right","left");
connect(s2Cds10,merge,"bottom","top","elbow");
connect(merge,genomes,"right","left");
connect(genomes,validate,"right","left");
connect(validate,evidence,"right","left");

addPhageIcon(70, 530);
addGenomeIcon(153, 548, 1.0, "icon-template-genome");
addGenomeIcon(312, 548, 0.95, "icon-original-genome");
addGenomeIcon(375, 548, 0.95, "icon-new-genome");
addShape("rightArrow", 355, 561, 20, 10, ARROW, ARROW, 0.5, "icon-replace-arrow");
addGenomeIcon(530, 548, 0.72, "icon-designed-genome-1");
addGenomeIcon(565, 548, 0.72, "icon-designed-genome-2");
addGenomeIcon(600, 548, 0.72, "icon-designed-genome-3");
addProteinIcon(755, 548, 0.45, "icon-validation-protein");
addChartIcon(815, 548);
addGridIcon(878, 553);
addClipboardIcon(1000, 538);
for (let i=0;i<4;i++) addShape("rect",1060+i*13,575-i*9,9,18+i*9, i===3?TEAL:BLUE, NAVY,0.5,`icon-evidence-bar-${i}`);

// Export draft, preview, and validated standalone source deck.
const preview = await p.export({ slide, format: "png", scale: 1.5 });
await fs.writeFile(path.join(TMP_DIR, "editable-flowchart-preview.png"), new Uint8Array(await preview.arrayBuffer()));
const layout = await slide.export({ format: "layout" });
await fs.writeFile(path.join(TMP_DIR, "editable-flowchart.layout.json"), await layout.text());

const candidatePath = path.join(TMP_DIR, "candidate.pptx");
await (await PresentationFile.exportPptx(p)).save(candidatePath);

await finalizePresentation({
  workspaceDir,
  candidatePath,
  finalPath: FINAL_PPTX,
  pythonExecutable: RUNTIME_PYTHON,
  integrityValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_layout_geometry.py"),
  layoutArgs: ["--expected-slide-size-emu", "12192000,6858000", "--validate-heading-fit"],
  explicitTotalSlideCount: 1,
  requiredNativeTableOwnerSlides: [],
  requiredNativeChartOwnerSlides: [],
  fontPolicy: { basis: "design", families: [FONT] },
  verifyArtifactToolImport: true,
  receiptPath: path.join(TMP_DIR, "current_experiment_editable_flowchart_v2.validation.json"),
});

console.log(FINAL_PPTX);
console.log(path.join(TMP_DIR, "editable-flowchart-preview.png"));
