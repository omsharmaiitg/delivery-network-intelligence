// ===========================================================================
//  Network Operations Strategy Memo  ->  outputs/Network_Operations_Strategy_Memo.docx
//  Audience: Head of Network Operations (non-technical). Decisions, not models.
// ===========================================================================
const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, LevelFormat, HeadingLevel, BorderStyle,
  WidthType, ShadingType, PageNumber, ImageRun, VerticalAlign,
} = require("docx");

const ROOT = "/home/claude/delhivery_project";
const FIG = ROOT + "/outputs/figures";

const INK = "1d2733", ACCENT = "c1440e", BLUE = "2c5f8a", LIGHT = "eef2f6",
      SAND = "f6f1e7", LINE = "c9d3dd", GREEN = "2e7d52";

// ---- helpers --------------------------------------------------------------
const P = (children, opts = {}) => new Paragraph({ children, spacing: { after: 120, line: 276 }, ...opts });
const run = (text, o = {}) => new TextRun({ text, font: "Arial", size: 20, ...o });
const H = (text, lvl = HeadingLevel.HEADING_2) =>
  new Paragraph({ heading: lvl, spacing: { before: 220, after: 100 },
    children: [new TextRun({ text, font: "Arial", color: INK })] });

const bullet = (children) => new Paragraph({
  numbering: { reference: "b", level: 0 }, spacing: { after: 90, line: 272 }, children });

const border = { style: BorderStyle.SINGLE, size: 4, color: LINE };
const borders = { top: border, bottom: border, left: border, right: border,
  insideHorizontal: border, insideVertical: border };

function cell(content, { w, fill, bold, color, align, size = 18 } = {}) {
  const items = Array.isArray(content) ? content : [content];
  return new TableCell({
    width: { size: w, type: WidthType.DXA }, borders,
    shading: fill ? { fill, type: ShadingType.CLEAR } : undefined,
    margins: { top: 60, bottom: 60, left: 110, right: 110 },
    verticalAlign: VerticalAlign.CENTER,
    children: items.map(t => new Paragraph({
      alignment: align || AlignmentType.LEFT,
      children: [new TextRun({ text: t, font: "Arial", size, bold: !!bold,
        color: color || INK })] })),
  });
}

function img(file, w, h, caption) {
  const out = [ new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 80, after: 40 },
    children: [ new ImageRun({ type: "png", data: fs.readFileSync(`${FIG}/${file}`),
      transformation: { width: w, height: h },
      altText: { title: caption, description: caption, name: file } }) ] }) ];
  if (caption) out.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 160 },
    children: [new TextRun({ text: caption, font: "Arial", italics: true, size: 16, color: "5a6b7b" })] }));
  return out;
}

// ---- header / footer ------------------------------------------------------
const header = new Header({ children: [ new Paragraph({
  border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: ACCENT, space: 4 } },
  children: [
    new TextRun({ text: "CONFIDENTIAL", font: "Arial", size: 14, bold: true, color: ACCENT }),
    new TextRun({ text: "\tDelhivery — Network Operations", font: "Arial", size: 14, color: "5a6b7b" }),
  ],
  tabStops: [{ type: "right", position: 9360 }],
}) ] });

const footer = new Footer({ children: [ new Paragraph({
  alignment: AlignmentType.CENTER,
  children: [
    new TextRun({ text: "Graph-Based Network Intelligence  ·  Strategy Memo  ·  Page ",
      font: "Arial", size: 14, color: "8a98a6" }),
    new TextRun({ children: [PageNumber.CURRENT], font: "Arial", size: 14, color: "8a98a6" }),
    new TextRun({ text: " of ", font: "Arial", size: 14, color: "8a98a6" }),
    new TextRun({ children: [PageNumber.TOTAL_PAGES], font: "Arial", size: 14, color: "8a98a6" }),
  ] }) ] });

// ---- BLUF box (single-cell shaded table) ----------------------------------
function bluf() {
  const txt = [
    new Paragraph({ spacing: { after: 60 }, children: [
      new TextRun({ text: "BOTTOM LINE", font: "Arial", size: 18, bold: true, color: ACCENT }) ]}),
    new Paragraph({ spacing: { after: 0, line: 270 }, children: [
      run("Our routing engine (OSRM) under-predicts delivery time by about ", {}),
      run("2× network-wide", { bold: true }),
      run(" — actual legs run roughly double the estimate, so ETAs are wrong and capacity planning slips. The delay is not spread evenly: it concentrates in a handful of mega-hubs. ", {}),
      run("Just five facilities account for 31% of all delay in the network, and the top three account for 24%.", { bold: true }),
      run(" We can (a) deploy a graph-based ETA model that is on-target ", {}),
      run("58% of the time vs 4% today", { bold: true, color: GREEN }),
      run(", and (b) by upgrading the top three hubs, cut late deliveries on their corridors by ", {}),
      run("~28%", { bold: true, color: GREEN }),
      run(" and recover an estimated ", {}),
      run("₹1.5 crore/year", { bold: true, color: GREEN }),
      run(" of revenue-at-risk.", {}),
    ]}),
  ];
  return new Table({ width: { size: 9360, type: WidthType.DXA }, columnWidths: [9360],
    borders: { top: { style: BorderStyle.SINGLE, size: 12, color: ACCENT },
      bottom: { style: BorderStyle.SINGLE, size: 12, color: ACCENT },
      left: { style: BorderStyle.SINGLE, size: 12, color: ACCENT },
      right: { style: BorderStyle.SINGLE, size: 12, color: ACCENT } },
    rows: [ new TableRow({ children: [ new TableCell({
      width: { size: 9360, type: WidthType.DXA },
      shading: { fill: SAND, type: ShadingType.CLEAR },
      margins: { top: 130, bottom: 130, left: 160, right: 160 },
      children: txt }) ] }) ] });
}

// ---- hub table ------------------------------------------------------------
function hubTable() {
  const head = ["Bottleneck hub", "Share of\nnetwork delay", "Why it is a chokepoint",
                "Recommended intervention"];
  const W = [2150, 1300, 3050, 2860];
  const rows = [ new TableRow({ tableHeader: true, children: head.map((t, i) =>
    cell(t.replace("\n", " "), { w: W[i], fill: INK, bold: true, color: "ffffff", size: 17 })) }) ];
  const data = [
    ["1. Gurgaon (Bilaspur HB), Haryana", "12.5%",
     "Highest betweenness in the country (0.19); 49 outbound + 45 inbound corridors. The national gateway — almost every long-haul trunk routes through it.",
     "Parallel sortation capacity + dedicated night-dispatch lanes to Bangalore, Kolkata, Hyderabad, Bhiwandi."],
    ["2. Bangalore (Nelamangla H), Karnataka", "6.3%",
     "Southern anchor; 35 out / 36 in corridors. The Gurgaon⇄Bangalore trunk is the single worst lane in the network.",
     "Add a parallel relay on the Gurgaon⇄Bangalore lane; balance inbound dock load."],
    ["3. Bhiwandi (Mankoli HB), Maharashtra", "5.5%",
     "Mumbai gateway with high intrinsic delay (2.25× OSRM) — slow even for its volume, pointing to facility-side dwell.",
     "Facility upgrade: dock automation + staffing to cut handling/dwell time."],
    ["4. Kolkata (Dankuni HB), West Bengal", "3.7%",
     "Eastern gateway and the worst delay ratio of the top hubs (2.59× OSRM).",
     "Facility upgrade + congestion-aware inbound scheduling."],
    ["5. Hyderabad (Shamshabad H), Telangana", "2.8%",
     "Central-south chokepoint (betweenness 0.11) linking north and south flows.",
     "Targeted throughput upgrade; monitor as volumes grow."],
  ];
  data.forEach((r, idx) => {
    const fill = idx < 3 ? SAND : (idx % 2 ? LIGHT : "ffffff");
    rows.push(new TableRow({ children: [
      cell(r[0], { w: W[0], fill, bold: true, size: 17 }),
      cell(r[1], { w: W[1], fill, bold: true, color: ACCENT, align: AlignmentType.CENTER, size: 20 }),
      cell(r[2], { w: W[2], fill, size: 16 }),
      cell(r[3], { w: W[3], fill, size: 16 }),
    ]}));
  });
  return new Table({ width: { size: 9360, type: WidthType.DXA }, columnWidths: W, borders, rows });
}

// ---- build doc ------------------------------------------------------------
const doc = new Document({
  creator: "Network Analytics", title: "Network Operations Strategy Memo",
  styles: { default: { document: { run: { font: "Arial", size: 20 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 30, bold: true, font: "Arial", color: INK },
        paragraph: { spacing: { before: 120, after: 120 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 23, bold: true, font: "Arial", color: ACCENT },
        paragraph: { spacing: { before: 220, after: 100 }, outlineLevel: 1 } },
    ] },
  numbering: { config: [
    { reference: "b", levels: [{ level: 0, format: LevelFormat.BULLET, text: "▪", alignment: AlignmentType.LEFT,
      style: { run: { color: ACCENT }, paragraph: { indent: { left: 460, hanging: 240 } } } }] },
    { reference: "n", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 460, hanging: 280 } } } }] },
  ] },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 },
      margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    headers: { default: header }, footers: { default: footer },
    children: [
      // Title block
      new Paragraph({ spacing: { after: 40 }, children: [
        new TextRun({ text: "STRATEGY MEMO", font: "Arial", size: 40, bold: true, color: INK }) ]}),
      new Paragraph({ spacing: { after: 160 },
        border: { bottom: { style: BorderStyle.SINGLE, size: 14, color: ACCENT, space: 6 } },
        children: [ new TextRun({ text: "Optimizing Delivery ETAs with Graph-Based Network Intelligence",
          font: "Arial", size: 22, color: "5a6b7b" }) ]}),

      // memo metadata
      new Table({ width: { size: 9360, type: WidthType.DXA }, columnWidths: [1500, 7860],
        borders: { top: { style: BorderStyle.NONE }, bottom: { style: BorderStyle.NONE },
          left: { style: BorderStyle.NONE }, right: { style: BorderStyle.NONE },
          insideHorizontal: { style: BorderStyle.NONE }, insideVertical: { style: BorderStyle.NONE } },
        rows: [
          ["TO:", "Head of Network Operations"],
          ["FROM:", "Network Analytics — Graph Intelligence Team"],
          ["RE:", "Where delay concentrates, and the three moves that recover it"],
        ].map(([k, v]) => new TableRow({ children: [
          new TableCell({ width: { size: 1500, type: WidthType.DXA }, margins: { top: 20, bottom: 20 },
            children: [new Paragraph({ children: [new TextRun({ text: k, font: "Arial", size: 18, bold: true, color: "5a6b7b" })] })] }),
          new TableCell({ width: { size: 7860, type: WidthType.DXA }, margins: { top: 20, bottom: 20 },
            children: [new Paragraph({ children: [new TextRun({ text: v, font: "Arial", size: 18, color: INK })] })] }),
        ] })) }),
      new Paragraph({ spacing: { after: 140 }, children: [] }),

      bluf(),
      new Paragraph({ spacing: { after: 120 }, children: [] }),

      H("The problem in one chart"),
      P([
        run("Delhivery promises ETAs from OSRM, a routing engine that assumes open roads and shortest paths. Reality includes congestion, dwell at facilities, and consolidation. The result: across 26,000 trip-legs, actual time is a median "),
        run("2.0× the OSRM estimate", { bold: true }),
        run(", and 95% of legs run more than 20% over. This is not occasional noise — the estimate is structurally low, which is why downstream capacity planning keeps breaking."),
      ]),
      ...img("fig5_osrm_miscalibration.png", 560, 233,
        "Figure 1 — Today's OSRM ETA is systematically ~2× too fast. The fix is not a tweak; it is a different model."),

      H("Where the delay actually is"),
      P([
        run("Treating the network as a graph (facilities as nodes, corridors as edges) lets us see that delay is highly concentrated. Of the network's total excess delay, "),
        run("the top 5 hubs carry 30.8% and the top 3 carry 24.3%", { bold: true }),
        run(". These are the points where fixing one facility moves the whole network."),
      ]),
      ...img("fig1_network_map.png", 430, 452,
        "Figure 2 — The delay map. Circle size = a hub's share of total network delay; red lines = the most delayed corridors. The Gurgaon⇄Bangalore trunk dominates."),

      H("Top 5 bottleneck hubs — and what to do about each"),
      hubTable(),
      new Paragraph({ spacing: { after: 80 }, children: [] }),
      ...img("fig2_bottleneck_hubs.png", 540, 360,
        "Figure 3 — Hub ranking by share of total network delay (top three in orange)."),

      H("The corridors bleeding the most time"),
      P([ run("Three trunk lanes alone lose enormous time versus estimate and should be the first targets for parallel routing / relay points:") ]),
      bullet([ run("Gurgaon ⇄ Bangalore", { bold: true }),
        run(" — ~2,866 min actual vs 1,548 OSRM one way; the two directions together lose ~145,000 excess minutes in just three weeks. This single lane is the clearest case for a parallel relay.") ]),
      bullet([ run("Gurgaon → Kolkata", { bold: true }),
        run(" — 2.03× OSRM, ~55,600 excess minutes. Pair facility scheduling at Kolkata with a dedicated dispatch slot.") ]),
      bullet([ run("Guwahati → Delhi & Gurgaon → Hyderabad / Bhiwandi", { bold: true }),
        run(" — long, high-ratio lanes (2.1–2.5×) where reliability, not distance, is the issue.") ]),

      H("Route type: stop deciding FTL vs Carting by habit"),
      P([
        run("Route-type choice today does not account for a facility's position in the network. Our framework makes the trade-off explicit: full-truck (FTL) runs measurably more reliably (1.92× vs 2.15× OSRM) and its time advantage grows with distance, while Carting is cheaper on short feeders. The data-backed rule:"),
      ]),
      bullet([ run("Long-haul (>400 km): default to FTL", { bold: true }),
        run(" — it is recommended on 100% of these lanes and is actually cheaper per leg once the truck is full, while saving 3–6 hours.") ]),
      bullet([ run("Short feeders (<150 km): default to Carting", { bold: true }),
        run(" — the FTL premium is not justified by the small time saving.") ]),
      bullet([ run("150–400 km: decide by graph position", { bold: true }),
        run(" — shift to FTL when the source is a central hub feeding an SLA-critical trunk (≈31% of these lanes).") ]),

      H("What the three moves are worth"),
      P([
        run("If the top three hubs (Gurgaon, Bangalore, Bhiwandi) are upgraded so that 40% of their excess delay is removed — a conservative target that brings them toward network-median performance — the modelled impact is:"),
      ]),
      bullet([ run("~28% fewer late deliveries", { bold: true, color: GREEN }),
        run(" on the corridors those hubs touch (≈5% across the entire network, from just three sites).") ]),
      bullet([ run("~110,000 hours of delay recovered per year", { bold: true, color: GREEN }),
        run(" — capacity that is currently lost to congestion and dwell.") ]),
      bullet([ run("≈ ₹1.5 crore/year of revenue-at-risk recovered", { bold: true, color: GREEN }),
        run(" (SLA penalties + churn avoided). Figures use explicit, finance-tunable assumptions; the leverage point — three hubs — is the robust finding.") ]),

      H("A faster win available now: better ETAs"),
      P([
        run("Independent of any facility spend, replacing the OSRM estimate with our graph-enhanced ETA model is a software change with immediate effect. On held-out data it is on-target (within 15% of actual) "),
        run("58% of the time versus 4% today", { bold: true, color: GREEN }),
        run(", and its average error falls from 108 minutes to 28. The single most powerful signal is corridor-level history — exactly what a point-to-point engine like OSRM ignores."),
      ]),
      ...img("fig3_model_comparison.png", 540, 245,
        "Figure 4 — The graph-enhanced model beats both today's OSRM and a trip-features-only baseline on accuracy and on the on-target-ETA business metric."),

      H("Recommended actions"),
      new Paragraph({ numbering: { reference: "n", level: 0 }, spacing: { after: 90 },
        children: [ run("Deploy the graph-enhanced ETA model", { bold: true }),
          run(" for promised-time generation and capacity planning (quick win, no capex).") ]}),
      new Paragraph({ numbering: { reference: "n", level: 0 }, spacing: { after: 90 },
        children: [ run("Upgrade Gurgaon first", { bold: true }),
          run(" — it alone is 12.5% of network delay; add parallel sortation and dedicated night-dispatch lanes.") ]}),
      new Paragraph({ numbering: { reference: "n", level: 0 }, spacing: { after: 90 },
        children: [ run("Stand up a parallel relay on Gurgaon⇄Bangalore", { bold: true }),
          run(", the worst single corridor in the network.") ]}),
      new Paragraph({ numbering: { reference: "n", level: 0 }, spacing: { after: 90 },
        children: [ run("Adopt the FTL/Carting rule", { bold: true }),
          run(" (FTL on long-haul and central-hub trunks; Carting on short feeders).") ]}),
      new Paragraph({ numbering: { reference: "n", level: 0 }, spacing: { after: 90 },
        children: [ run("Review Bhiwandi and Kolkata for facility-side dwell", { bold: true }),
          run(" — their delay is high even for their volume, pointing to handling, not roads.") ]}),

      new Paragraph({ spacing: { before: 200 }, border: { top: { style: BorderStyle.SINGLE, size: 4, color: LINE, space: 6 } },
        children: [ new TextRun({ text: "Method in brief: ", font: "Arial", size: 15, bold: true, color: "5a6b7b" }),
          new TextRun({ text: "Directed weighted graph of 1,657 facilities and 2,775 corridors; edge weights = median actual÷OSRM delay ratio, stratified by route type and time of day. Hub ranking blends betweenness centrality, throughput and delay contribution. ETA model = gradient-boosted trees on trip features plus node2vec graph embeddings and corridor history, evaluated on a held-out 20% with leakage controls. All business constants are configurable. Data window ≈ 25 days, Sep–Oct 2018.",
            font: "Arial", size: 15, italics: true, color: "8a98a6" }) ]}),
    ],
  }],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(ROOT + "/outputs/Network_Operations_Strategy_Memo.docx", buf);
  console.log("memo written:", buf.length, "bytes");
});
