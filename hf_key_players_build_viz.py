"""Build the standalone AI World "Key players by organisation type" viz (logo rosettes).

Four rosettes, one per organisation type, each showing the seven accounts of that
type with the most dataset downloads in the last 30 days. Membership and the
metrics are a frozen snapshot (see SNAPSHOT below), taken from the Hugging Face
public API; only the avatars are re-fetched at build time and inlined as base64
so the output stays a single self-contained file.

Colours come from the project-wide org-type palette (the same colour-blind-safe
set used by the other viewer/final-aiw-viz charts), so a reader moving between
the charts keeps one colour per type.

The published file is `viewer/final-aiw-viz/dataset_key_players_by_org_type.html`,
ready to host at viz.aiworld.eu (e.g. aiworld/Story/<viz-slug>/viz.html).

Usage:
  python hf_key_players_build_viz.py                # fetch avatars, write viz
  python hf_key_players_build_viz.py --no-avatars   # reuse avatars already in the viz
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import re
import sys
import urllib.request
from pathlib import Path

from PIL import Image

OUT_HTML = Path("viewer/final-aiw-viz/dataset_key_players_by_org_type.html")
SNAPSHOT = "29 July 2026"
AVATAR_PX = 128
UA = {"User-Agent": "Mozilla/5.0 (CEPS AI World viz build)"}

log = logging.getLogger("key-players")

# ---------------------------------------------------------------------------
# Data. One entry per account; `words` are the on-figure label broken at
# natural boundaries (lines concatenate back to the exact handle).
# dl30 = dataset downloads in the 30 days to the snapshot date.
# ---------------------------------------------------------------------------
GROUPS = [
    ("government", "Governments", [
        dict(handle="NbAiLab", words=["Nb", "Ai", "Lab"], name="Nasjonalbiblioteket AI Lab",
             cc="NO", country="Norway", datasets=40, models=251, followers=387, dl30=1168733,
             known="NB-Whisper · NB-BERT · NPSC",
             blurb="The National Library of Norway's AI Lab, turning the legal-deposit "
                   "archive into Norwegian speech and language models."),
        dict(handle="NetherlandsForensicInstitute", words=["Netherlands", "Forensic", "Institute"],
             name="Netherlands Forensic Institute", cc="NL", country="Netherlands",
             datasets=19, models=5, followers=51, dl30=18132,
             known="RobBERT ST · ARM64BERT · Vuurwerkverkenner",
             blurb="Ministry of Justice and Security forensic lab, open-sourcing the models it "
                   "uses in casework."),
        dict(handle="AgentPublic", words=["Agent", "Public"], name="AgentPublic",
             cc="FR", country="France", datasets=80, models=19, followers=153, dl30=16991,
             known="LEGI · PIAF · Guillaume Tell 7B",
             blurb="Etalab / DINUM, the French state digital agency. Publishes legal corpora "
                   "and a sovereign public-service LLM."),
        dict(handle="HHS-Official", words=["HHS-", "Official"],
             name="Department of Health and Human Services", cc="US", country="United States",
             datasets=414, models=0, followers=32, dl30=5632,
             known="Autism prevalence · COVID-19 weekly",
             blurb="Federal health department, mirroring healthdata.gov and CDC releases "
                   "onto the Hub."),
        dict(handle="NationalLibraryOfScotland", words=["National", "Library", "Of", "Scotland"],
             name="National Library of Scotland", cc="UK", country="United Kingdom",
             datasets=11, models=2, followers=19, dl30=4891,
             known="Chapbook illustrations · Scottish exam papers",
             blurb="Digitised heritage collections released as ML-ready datasets."),
        dict(handle="destatis", words=["destatis"], name="Statistisches Bundesamt",
             cc="DE", country="Germany", datasets=224, models=0, followers=1, dl30=4693,
             known="Preise · Arbeitslose · Kurzarbeiter",
             blurb="The German federal statistical office, publishing official series directly "
                   "as datasets."),
        dict(handle="govtech", words=["govtech"], name="GovTech Singapore - AI Practice",
             cc="SG", country="Singapore", datasets=9, models=7, followers=73, dl30=989,
             known="LionGuard · RabakBench · MinorBench",
             blurb="Safety guardrails and evaluation benchmarks built for Singapore's "
                   "public-sector deployments."),
    ]),
    ("university", "Universities", [
        dict(handle="Helsinki-NLP", words=["Helsinki-", "NLP"], name="Helsinki-NLP Research Group",
             cc="FI", country="Finland", datasets=53, models=1563, followers=1065, dl30=7605230,
             known="OPUS-100 · OPUS-MT (1,563 models)",
             blurb="Home of the OPUS parallel corpora, and a translation model for almost "
                   "every language pair."),
        dict(handle="jhu-clsp", words=["jhu-", "clsp"],
             name="Center for Language and Speech Processing @ JHU", cc="US",
             country="United States", datasets=40, models=53, followers=247, dl30=851005,
             known="mmBERT · JFLEG · Kreyòl-MT",
             blurb="Multilingual encoders and corpora for languages the big models handle "
                   "badly."),
        dict(handle="uoft-cs", words=["uoft-", "cs"], name="University of Toronto Computer Science",
             cc="CA", country="Canada", datasets=2, models=0, followers=100, dl30=558205,
             known="CIFAR-10 · CIFAR-100",
             blurb="Two datasets, both from 2009, both still among the most-pulled on the "
                   "Hub."),
        dict(handle="nyu-mll", words=["nyu-", "mll"], name="NYU Machine Learning for Language",
             cc="US", country="United States", datasets=5, models=12, followers=213, dl30=476491,
             known="GLUE · MultiNLI · BLiMP",
             blurb="Built the benchmark suite that defined the BERT era."),
        dict(handle="behavior-1k", words=["behavior-", "1k"], name="BEHAVIOR-1K",
             cc="US", country="United States", datasets=8, models=0, followers=49, dl30=373225,
             known="BEHAVIOR challenge demos",
             blurb="Stanford's embodied-AI benchmark: 1,000 everyday household activities in "
                   "simulation."),
        dict(handle="stanfordnlp", words=["stanford", "nlp"], name="Stanford NLP",
             cc="US", country="United States", datasets=18, models=118, followers=391, dl30=322701,
             known="IMDb · SST-2 · SHP · CoreNLP",
             blurb="Stanford NLP Group; its sentiment corpora are still default smoke-tests for "
                   "new models."),
        dict(handle="CERN", words=["CERN"], name="CERN - European Organization for Nuclear Research",
             cc="CH", country="Switzerland", datasets=2, models=0, followers=125, dl30=187064,
             known="ColliderML Release 1",
             blurb="The particle-physics lab, now publishing collider ML benchmarks on the Hub."),
    ]),
    ("community", "Communities", [
        dict(handle="IPEC-COMMUNITY", words=["IPEC-", "COMMUNITY"],
             name="IPEC at Shanghai AI Laboratory", cc="CN", country="China",
             datasets=49, models=10, followers=221, dl30=2311565,
             known="droid_lerobot · bridge_orig · EO-Data1.5M",
             blurb="Shanghai AI Lab's IPEC group; re-publishes the major robot datasets in a "
                   "single LeRobot format so they can be trained on together."),
        dict(handle="m-a-p", words=["m-a-p"], name="Multimodal Art Projection",
             cc="CN", country="China", datasets=76, models=212, followers=1102, dl30=1893935,
             known="COIG-CQIA · YuE · ChatMusician",
             blurb="Open research collective spanning Chinese instruction data, code and "
                   "music generation."),
        dict(handle="BangumiBase", words=["Bangumi", "Base"], name="BangumiBase",
             cc="", country="No stated affiliation", datasets=810, models=0, followers=161, dl30=1213963,
             known="One Piece · Steins;Gate · 808 more",
             blurb="One dataset per anime series, a character image archive. Fandom "
                   "infrastructure on the Hub."),
        dict(handle="agibot-world", words=["agibot-", "world"], name="AgiBot World",
             cc="CN", country="China", datasets=11, models=6, followers=568, dl30=404948,
             known="AgiBotWorld Alpha/Beta · GO-1",
             blurb="AgiBot's open real-robot programme: thousands of hours of manipulation "
                   "on one hardware platform."),
        dict(handle="DL3DV", words=["DL3DV"], name="DL3DV", cc="INT", country="International",
             datasets=11, models=0, followers=78, dl30=350735,
             known="DL3DV-10K · DL3DV-Benchmark",
             blurb="Real-world 3D scene capture at scale; the standard benchmark for novel-view "
                   "synthesis."),
        dict(handle="MedOtter", words=["Med", "Otter"], name="MedOtter",
             cc="", country="No stated affiliation", datasets=128, models=106, followers=10, dl30=281449,
             known="BraTS 2023 · 4D-Lung · CT Lymph Nodes",
             blurb="Mirrors clinical imaging archives onto the Hub: CT, MRI and pathology."),
        dict(handle="FLARE-MedFM", words=["FLARE-", "MedFM"],
             name="Fast, Low-resource, Accurate, Robust and Effectual Medical Image Analysis",
             cc="INT", country="International", datasets=11, models=1, followers=67, dl30=47200,
             known="PancancerCTSeg · FLARE-MLLM-2D",
             blurb="MICCAI challenge community building medical image segmentation models "
                   "as a public competition."),
    ]),
    ("non-profit", "Non-profits", [
        dict(handle="allenai", words=["allenai"], name="Ai2", cc="US", country="United States",
             datasets=1275, models=968, followers=6348, dl30=11401946,
             known="Dolma · OLMo · Molmo · C4",
             blurb="Fully-open OLMo stack: weights, data and training code all released "
                   "together."),
        dict(handle="EleutherAI", words=["Eleuther", "AI"], name="EleutherAI",
             cc="US", country="United States", datasets=255, models=972, followers=1356,
             dl30=7458347, known="The Pile · GPT-J-6B · GPT-NeoX-20B",
             blurb="Grassroots collective turned nonprofit lab; built the open pretraining "
                   "corpus much of the field trained on."),
        dict(handle="mteb", words=["mteb"], name="Massive Text Embedding Benchmark",
             cc="INT", country="International", datasets=1654, models=4, followers=1116,
             dl30=1364534, known="MTEB task suite (1,654 sets)",
             blurb="Keeper of the embedding leaderboard that new text-embedding models "
                   "report against."),
        dict(handle="bigcode", words=["bigcode"], name="BigCode", cc="INT",
             country="International", datasets=93, models=69, followers=2121, dl30=1057237,
             known="The Stack v1/v2 · StarCoder · StarCoder2",
             blurb="Open scientific collaboration (ServiceNow + Hugging Face) on "
                   "responsibly-licensed code LLMs."),
        dict(handle="RoboCOIN", words=["Robo", "COIN"], name="BAAI-RoboCOIN",
             cc="CN", country="China", datasets=771, models=0, followers=47, dl30=732235,
             known="Cobot Magic · AgiBot-G1 episodes",
             blurb="BAAI's robot-data consortium: cross-embodiment manipulation episodes pooled "
                   "from many robot platforms."),
        dict(handle="cais", words=["cais"], name="Center for AI Safety",
             cc="US", country="United States", datasets=13, models=8, followers=562, dl30=614571,
             known="Humanity's Last Exam · MMLU · WMDP",
             blurb="Publishes the benchmark exams frontier models are scored on."),
        dict(handle="InternRobotics", words=["Intern", "Robotics"], name="Intern Robotics",
             cc="CN", country="China", datasets=29, models=39, followers=388, dl30=286915,
             known="InternData-A1 · InternVLA · OmniWorld",
             blurb="Shanghai AI Laboratory's embodied-AI arm: robot data and "
                   "vision-language-action policies."),
    ]),
]


# ---------------------------------------------------------------------------
# Avatars
# ---------------------------------------------------------------------------
def fetch_avatar(handle: str) -> str:
    """Return a base64 data URI for the org avatar: centre-cropped square, WEBP."""
    meta_url = f"https://huggingface.co/api/organizations/{handle}/overview"
    with urllib.request.urlopen(urllib.request.Request(meta_url, headers=UA), timeout=30) as r:
        url = json.load(r).get("avatarUrl")
    if not url:
        raise RuntimeError("no avatarUrl")
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
        raw = r.read()

    im = Image.open(io.BytesIO(raw))
    im = im.convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))  # discs sit on white
    im = Image.alpha_composite(bg, im).convert("RGB")
    w, h = im.size
    s = min(w, h)                                          # centre-crop to square ("cover")
    im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    im = im.resize((AVATAR_PX, AVATAR_PX), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "WEBP", quality=82, method=6)
    return "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def existing_avatars() -> dict[str, str]:
    """Recover avatars from a previously built viz so --no-avatars can skip the network."""
    if not OUT_HTML.exists():
        return {}
    m = re.search(r"const DATA = (\[.*?\]);\n", OUT_HTML.read_text(encoding="utf-8"), re.S)
    if not m:
        return {}
    out = {}
    for g in json.loads(m.group(1)):
        for mem in g["members"]:
            if mem.get("avatar"):
                out[mem["handle"]] = mem["avatar"]
    return out


# ---------------------------------------------------------------------------
# Template
# ---------------------------------------------------------------------------
TEMPLATE = r"""<!doctype html><html><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Key players on the Hugging Face Hub</title>
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
  :root { --breakpoint-compact: 500px; }            /* MUST equal BREAKPOINT_COMPACT in JS */
  html,body { margin:0; height:100vh; width:100vw; overflow:hidden; color:#0f172a;
    font-family: system-ui,-apple-system,"Segoe UI",Roboto,Ubuntu,Helvetica,Arial,sans-serif; }
  .viz-grid { display:grid; height:100vh; box-sizing:border-box; padding-bottom:clamp(8px,2vh,18px);
    grid-template-rows:18fr 1fr 1fr; grid-template-columns:1fr 1fr; }   /* rebuilt by applyContainerVisibility() */
  #container1 { grid-column:1 / -1; min-height:0; } #viz { width:100%; height:100%; }
  #container2 { grid-column:1 / -1; }               /* legend */
  #legend { height:100%; display:flex; flex-wrap:wrap; align-items:center; justify-content:center;
    gap:2px 16px; font-size:clamp(9px,1.4vw,13px); font-weight:600; color:#475569; }
  #legend .chip { display:inline-block; width:11px; height:11px; border-radius:2px; margin-right:6px; vertical-align:-1px; }
  #legend .val { font-variant-numeric:tabular-nums; }
  #legend .note { font-weight:600; color:#94a3b8; }
  /* source (left): width-scaled, vertically centred, ellipsised so a long source never wraps the row */
  #container3 { font-size:clamp(5px,1.6vw,16px); display:flex; align-items:center; justify-content:flex-start;
    padding:0 20px; overflow:hidden; min-width:0; }
  #container3 #source-text { font-weight:600; color:#333; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; max-width:100%; }
  #container3 #source-text a { color:#2756d3; text-decoration:underline; }
  /* logo (right): fills the footer row height */
  #container4 { display:flex; justify-content:flex-end; align-items:center; padding:0 20px; overflow:hidden; min-width:0; }
  #container4 a { height:100%; display:flex; align-items:center; }
  #container4 img { height:100%; width:auto; max-height:100%; max-width:100%; object-fit:contain; }
  #tooltip { position:fixed; pointer-events:none; display:none; background:#fff;
    border:1px solid #DBDBDB; padding:8px 10px; font-size:12px; border-radius:4px;
    max-width:350px; box-shadow:0 6px 20px rgba(15,23,42,0.10); border-top-width:3px; border-top-style:solid; }
  #tooltip .tt-cc { font-size:10px; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; }
  #tooltip .tt-title { font-weight:700; font-size:13px; line-height:1.25; margin-top:2px; }
  #tooltip .tt-handle { color:#475569; font-size:11px; }
  #tooltip .tt-blurb { color:#334155; line-height:1.45; margin-top:6px; }
  #tooltip .tt-stats { display:grid; grid-template-columns:repeat(4,1fr); gap:6px;
    margin-top:7px; padding-top:6px; border-top:1px solid #DBDBDB; }
  #tooltip .tt-stats dt { font-size:9px; font-weight:700; letter-spacing:0.04em; text-transform:uppercase; color:#475569; }
  #tooltip .tt-stats dd { margin:1px 0 0; font-size:12px; font-weight:700; font-variant-numeric:tabular-nums; }
  #tooltip .tt-known { margin-top:6px; padding-top:5px; border-top:1px solid #DBDBDB;
    font-size:10px; color:#475569; line-height:1.5; }
  svg text { font-family:inherit; }
  svg .node { cursor:pointer; }
  /* Short iframe / "snapshot" embed: drop ALL chrome (legend, source, logo); chart alone fills the frame. */
  @media (max-height:500px){
    .viz-grid { grid-template-rows:1fr !important; grid-template-columns:1fr !important; padding-bottom:0; }
    #container1 { grid-row:1 !important; grid-column:1 / -1 !important; }
    #container2,#container3,#container4,#container5 { display:none !important; }
  }
</style></head><body>
<div class="viz-grid">
  <div id="container1"><div id="viz"></div></div>
  <div id="container2"><div id="legend"></div></div>
  <div id="container3"><span id="source-text"></span></div>
  <div id="container4"><a href="https://aiworld.eu/" target="_blank" rel="noopener"><img src="https://aiworld.eu/logo-transparent.svg" alt="AI World"></a></div>
</div>
<div id="tooltip"></div>
<script>
"use strict";
// ---- CONFIG: declare everything; nothing implicit ----
const SHOW_CONTAINER_2 = 1, SHOW_CONTAINER_3 = 1, SHOW_CONTAINER_4 = 1, SHOW_CONTAINER_5 = 0;
const SHOW_TOOLTIP = 1;
const BREAKPOINT_COMPACT = 500;                       // px; MUST equal --breakpoint-compact

// One colour per organisation type (colour-blind-safe set reused across the HF project).
const COLORS = { company:"#DE8F05", university:"#CC78BC", "non-profit":"#D55E00",
                 community:"#029E73", classroom:"#CA9161", government:"#0173B2" };
const INK = "#0f172a", MUTED = "#475569";

// Rosette geometry, in units of L (one node diameter). Seven nodes, first at 12 o'clock.
const N_NODES = 7;
const R_RING = 1.15238;                               // node-centre radius
const R_HUB = 0.65238;                                // hub-disc radius
const LABEL_GAP = 0.60;                               // label offset from node centre
const MAX_LABEL_W = 1.45;                             // wrap width for an account label
const FS_LABEL = 0.185, FS_HUB = 0.155, LINE_H = 1.15;// label / hub font size in L; line height in ems
// Bottom of the content box, in L: the two lowest discs, plus a little room for the
// 1.07 hover lift. The top is measured at run time (it depends on how the top label wraps).
const EXT_BOTTOM = Math.sin(64.2857 * Math.PI / 180) * R_RING + 0.5 * 1.07;
const COL_CHOICES = [1, 2, 4];                        // cluster grid arrangements to try

const SNAPSHOT = "__SNAPSHOT__";
const SOURCE_NAME = "CEPS analysis of the Hugging Face Hub API, " + SNAPSHOT;
const SOURCE_HREF = "https://huggingface.co/docs/hub/api";

/* DATA: one object per organisation type, `members` ordered by dl30 desc (the seven
   most-downloaded accounts of that type). dl30 = dataset downloads in the last 30 days;
   `words` are the on-figure label broken at natural boundaries (lines concatenate back
   to the exact handle); `avatar` is the org avatar inlined as a base64 WEBP. */
const DATA = __DATA__;

document.getElementById("source-text").innerHTML =
  'Source: <a href="' + SOURCE_HREF + '" target="_blank" rel="noopener">' + SOURCE_NAME + '</a>';

// Show only the footer rows whose flag is on, then recompute grid rows.
(function applyContainerVisibility(){
  const grid = document.querySelector(".viz-grid"), c = n => document.getElementById("container"+n);
  const c2=c(2), c3=c(3), c4=c(4), c5=c(5);
  if (c2 && !SHOW_CONTAINER_2) c2.style.display="none";
  if (c3 && !SHOW_CONTAINER_3) c3.style.display="none";
  if (c4 && !SHOW_CONTAINER_4) c4.style.display="none";
  if (c5 && !SHOW_CONTAINER_5) c5.style.display="none";
  const rowCtrl=SHOW_CONTAINER_5, rowLegend=SHOW_CONTAINER_2, rowFooter=SHOW_CONTAINER_3||SHOW_CONTAINER_4;
  const rows=[]; let r=1;
  rows.push((20-(rowCtrl?1:0)-(rowLegend?1:0)-(rowFooter?1:0))+"fr");   // chart absorbs freed rows
  if (rowCtrl)   { rows.push("1fr"); r++; if (c5) c5.style.gridRow=r; }
  if (rowLegend) { rows.push("1fr"); r++; if (c2) c2.style.gridRow=r; }
  if (rowFooter) { rows.push("1fr"); r++;
    if (SHOW_CONTAINER_3 && c3) c3.style.gridRow=r;
    if (SHOW_CONTAINER_4 && c4) c4.style.gridRow=r;
    if (SHOW_CONTAINER_3 && !SHOW_CONTAINER_4 && c3) c3.style.gridColumn="1 / span 2";  // lone item spans both cols
    if (!SHOW_CONTAINER_3 && SHOW_CONTAINER_4 && c4) c4.style.gridColumn="1 / span 2";
  }
  grid.style.gridTemplateRows = rows.join(" ");
  c(1).style.gridRow = "1";
})();

/* ---- helpers ---- */
function fmtCompact(v){
  if (v == null) return "n/a";
  for (const [div, suf] of [[1e9,"B"],[1e6,"M"],[1e3,"K"]])
    if (v >= div) { const x = v/div; return (x >= 100 ? Math.round(x) : +x.toFixed(1)) + suf; }
  return String(Math.round(v));
}
const groupTotal = g => d3.sum(g.members, m => m.dl30);

/* Text measuring. Done with a hidden <text> inside the live SVG rather than a canvas,
   so widths come from the font actually used to draw (a canvas ctx.font built from the
   house stack can silently fail to parse and fall back to 10px sans-serif). */
function makeMeasurer(svg){
  const probe = svg.append("text").attr("visibility", "hidden").attr("x", 0).attr("y", 0);
  const node = probe.node(), cache = new Map();
  return function textW(s, px, weight){
    const key = weight + "|" + px + "|" + s;
    let w = cache.get(key);
    if (w === undefined){
      node.setAttribute("font-size", px);
      node.setAttribute("font-weight", weight);
      node.textContent = s;
      w = node.getComputedTextLength();
      cache.set(key, w);
    }
    return w;
  };
}
/* Greedy pack: words concatenate with no separator, so lines rejoin into the handle. */
function wrapWords(words, px, weight, maxW, textW){
  const lines = [];
  let cur = "";
  for (const w of words){
    const next = cur + w;
    if (cur && textW(next, px, weight) > maxW) { lines.push(cur); cur = w; }
    else cur = next;
  }
  if (cur) lines.push(cur);
  return lines;
}
/* Pick white or ink for text on a filled disc, whichever has more contrast (WCAG). */
function onColor(bg){
  const c = d3.rgb(bg), lin = v => { v /= 255; return v <= 0.03928 ? v/12.92 : Math.pow((v+0.055)/1.055, 2.4); };
  const Lb = 0.2126*lin(c.r) + 0.7152*lin(c.g) + 0.0722*lin(c.b);
  const cw = 1.05 / (Lb + 0.05);                      // against #fff
  const ci = (Lb + 0.05) / (0.0114 + 0.05);           // against INK (#0f172a)
  return cw >= ci ? "#fff" : INK;
}

/* ---- legend: one item per type, with that type's 30-day download total ---- */
(function buildLegend(){
  const el = document.getElementById("legend");
  el.innerHTML = "";
  for (const g of DATA){
    const item = document.createElement("span");
    const chip = document.createElement("span");
    chip.className = "chip"; chip.style.background = COLORS[g.type];
    const val = document.createElement("span");
    val.className = "val"; val.textContent = " " + fmtCompact(groupTotal(g));
    item.append(chip, document.createTextNode(g.label), val);
    el.appendChild(item);
  }
  const note = document.createElement("span");
  note.className = "note";
  note.textContent = "seven most-downloaded accounts per type (Hub sector badge) · "
    + "downloads, last 30 days · circle size carries no data";
  el.appendChild(note);
})();

/* ---- tooltip ---- */
const tooltipEl = document.getElementById("tooltip");
function showTooltip(evt, m, color){
  if (!SHOW_TOOLTIP) return;
  tooltipEl.style.borderTopColor = color;
  tooltipEl.innerHTML =
    '<div class="tt-cc" style="color:' + color + '">' + [m.cc, m.country].filter(Boolean).join(" · ") + '</div>' +
    '<div class="tt-title">' + m.name + '</div>' +
    '<div class="tt-handle">@' + m.handle + '</div>' +
    '<div class="tt-blurb">' + m.blurb + '</div>' +
    '<dl class="tt-stats">' +
      '<div><dt>Datasets</dt><dd>' + m.datasets.toLocaleString("en-GB") + '</dd></div>' +
      '<div><dt>Models</dt><dd>' + m.models.toLocaleString("en-GB") + '</dd></div>' +
      '<div><dt>DL, 30 days</dt><dd>' + fmtCompact(m.dl30) + '</dd></div>' +
      '<div><dt>Followers</dt><dd>' + fmtCompact(m.followers) + '</dd></div>' +
    '</dl>' +
    '<div class="tt-known">Known for: ' + m.known + '</div>';
  tooltipEl.style.display = "block";
  moveTooltip(evt);
}
function moveTooltip(evt){
  if (tooltipEl.style.display !== "block") return;
  const w = tooltipEl.offsetWidth, h = tooltipEl.offsetHeight;
  tooltipEl.style.left = Math.max(8, Math.min(evt.clientX + 16, innerWidth - w - 8)) + "px";
  tooltipEl.style.top  = Math.max(8, Math.min(evt.clientY + 16, innerHeight - h - 8)) + "px";
}
function hideTooltip(){ tooltipEl.style.display = "none"; }

/* ---- render ---- */
const vizContainer = document.getElementById("viz");
let firstRender = true;

function render(){
  d3.select("#viz").selectAll("*").remove();
  hideTooltip();
  const rect = vizContainer.getBoundingClientRect();
  const W = Math.max(1, rect.width), H = Math.max(1, rect.height);
  // In a short embed all chrome is hidden and the frame is used for snapshots / og:image,
  // so draw the final state straight away instead of animating into it.
  const compact = window.matchMedia("(max-height:" + BREAKPOINT_COMPACT + "px)").matches;
  const animate = firstRender && !compact
    && !window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const svg = d3.select("#viz").append("svg")
    .attr("width", W).attr("height", H).attr("viewBox", [0, 0, W, H])
    .attr("role", "img")
    .attr("aria-label", "Seven most-downloaded Hugging Face accounts for each of four organisation types");
  const textW = makeMeasurer(svg);

  // Node angles, first at 12 o'clock; side of the label follows the horizontal direction.
  const geom = d3.range(N_NODES).map(i => {
    const a = (-90 + i * (360 / N_NODES)) * Math.PI / 180;
    return { a, ux: Math.cos(a) * R_RING, uy: Math.sin(a) * R_RING,
             side: i === 0 ? "t" : (Math.cos(a) > 0 ? "r" : "l") };
  });

  /* Measure the content box, in units of L. Label font size (FS_LABEL·L) and wrap width
     (MAX_LABEL_W·L) both scale with L, so how a label wraps -- and how wide it ends up
     relative to L -- is the same at every size. That makes one measurement pass valid
     for all L, so the box can be measured rather than guessed. */
  const REF = 100;                                    // reference L for the measuring pass
  const wrapAt = m => wrapWords(m.words, REF * FS_LABEL, 600, REF * MAX_LABEL_W, textW);
  let extTop = 0, halfW = 0;
  for (const g of DATA){
    g.members.forEach((m, i) => {
      const { ux, side } = geom[i];
      const lines = wrapAt(m);
      const wide = d3.max(lines, ln => textW(ln, REF * FS_LABEL, 600)) / REF;
      if (side === "t")
        extTop = Math.max(extTop, R_RING + LABEL_GAP + (lines.length - 1) * FS_LABEL * LINE_H + FS_LABEL / 2);
      halfW = Math.max(halfW, side === "t" ? Math.max(0.5, wide / 2) : Math.abs(ux) + LABEL_GAP + wide);
    });
  }
  extTop = Math.max(extTop, R_RING + 0.5 * 1.07);     // a lone short label must still clear the disc
  const BOX_W = 2 * halfW, BOX_H = extTop + EXT_BOTTOM;
  const HUB_DY = (extTop - EXT_BOTTOM) / 2;           // hub offset below the cell centre

  // Pick the cluster arrangement that makes the rosettes biggest.
  let best = null;
  for (const cols of COL_CHOICES){
    const rows = Math.ceil(DATA.length / cols);
    const L = Math.min((W / cols) / BOX_W, (H / rows) / BOX_H);
    if (!best || L > best.L) best = { cols, rows, L };
  }
  const { cols, rows, L } = best;
  const cellW = W / cols, cellH = H / rows;

  // One clip path for every logo disc (all nodes share a radius; the node group's
  // transform scales the clip along with the image on hover).
  svg.append("defs").append("clipPath").attr("id", "logoClip")
    .append("circle").attr("r", L / 2);

  const fsLabel = Math.max(6, L * FS_LABEL);
  const lineH = fsLabel * LINE_H;
  const maxLabelW = L * MAX_LABEL_W;
  // One hub font size for all four discs (shrunk to fit the longest label), so the
  // type names stay visually consistent across the clusters.
  let fsHub = Math.max(6, L * FS_HUB);
  while (fsHub > 5 && d3.max(DATA, g => textW(g.label, fsHub, 700)) > R_HUB * L * 1.72) fsHub -= 0.5;

  DATA.forEach((g, gi) => {
    const color = COLORS[g.type];
    const cx = (gi % cols) * cellW + cellW / 2;
    const cy = Math.floor(gi / cols) * cellH + cellH / 2 + HUB_DY * L;
    const cluster = svg.append("g").attr("transform", "translate(" + cx + "," + cy + ")");

    // hub disc: the organisation type
    const hub = cluster.append("g").attr("class", "hub");
    hub.append("circle").attr("r", R_HUB * L).attr("fill", color);
    hub.append("text")
      .attr("text-anchor", "middle").attr("dominant-baseline", "central")
      .attr("fill", onColor(color)).attr("font-size", fsHub).attr("font-weight", 700)
      .attr("letter-spacing", "-0.015em")
      .text(g.label);

    g.members.forEach((m, i) => {
      const { ux, uy, side } = geom[i];
      const nx = ux * L, ny = uy * L;

      // ---- label (drawn first so the disc always sits on top) ----
      const lines = wrapWords(m.words, fsLabel, 600, maxLabelW, textW);
      const label = cluster.append("text")
        .attr("class", "label")
        .attr("font-size", fsLabel).attr("font-weight", 600).attr("fill", MUTED)
        .attr("text-anchor", side === "t" ? "middle" : (side === "r" ? "start" : "end"))
        .style("cursor", "pointer");
      lines.forEach((ln, k) => {
        const y = side === "t"
          ? ny - LABEL_GAP * L - (lines.length - 1 - k) * lineH
          : ny + (k - (lines.length - 1) / 2) * lineH;
        label.append("tspan")
          .attr("x", side === "t" ? nx : nx + (side === "r" ? LABEL_GAP * L : -LABEL_GAP * L))
          .attr("y", y).attr("dominant-baseline", "central")
          .text(ln);
      });

      // ---- logo disc ----
      const node = cluster.append("g")
        .attr("class", "node")
        .attr("transform", "translate(" + nx + "," + ny + ")")
        .attr("role", "button").attr("tabindex", 0)
        .attr("aria-label", m.handle + ": " + m.name);
      node.append("title").text(m.handle + ": " + m.name);
      node.append("circle").attr("r", L / 2).attr("fill", "#fff");
      node.append("image")
        .attr("href", m.avatar)
        .attr("x", -L / 2).attr("y", -L / 2).attr("width", L).attr("height", L)
        .attr("preserveAspectRatio", "xMidYMid slice")
        .attr("clip-path", "url(#logoClip)");
      const ring = node.append("circle")
        .attr("r", L / 2 - 0.5).attr("fill", "none")
        .attr("stroke", "rgba(31,31,33,0.15)").attr("stroke-width", 1);

      // hover / focus: lift the disc, ring it in the type colour, darken the label
      const on = evt => {
        node.raise().attr("transform", "translate(" + nx + "," + ny + ") scale(1.07)");
        ring.attr("stroke", color).attr("stroke-width", 2.5);
        label.attr("fill", INK);
        showTooltip(evt, m, color);
      };
      const off = () => {
        node.attr("transform", "translate(" + nx + "," + ny + ")");
        ring.attr("stroke", "rgba(31,31,33,0.15)").attr("stroke-width", 1);
        label.attr("fill", MUTED);
        hideTooltip();
      };
      const open = () => window.open(m.url, "_blank", "noopener");
      node.on("mouseenter", on).on("mousemove", moveTooltip).on("mouseleave", off)
          .on("focus", evt => on(evt)).on("blur", off).on("click", open)
          .on("keydown", evt => { if (evt.key === "Enter" || evt.key === " ") { evt.preventDefault(); open(); } });
      label.on("mouseenter", on).on("mousemove", moveTooltip).on("mouseleave", off).on("click", open);

      if (animate){
        node.attr("opacity", 0)
          .attr("transform", "translate(" + nx + "," + ny + ") scale(0.4)")
          .transition().duration(850).delay(120 + gi * 60 + i * 45).ease(d3.easeCubicOut)
          .attr("opacity", 1).attr("transform", "translate(" + nx + "," + ny + ")");
        label.attr("opacity", 0)
          .transition().duration(500).delay(320 + gi * 60 + i * 45).attr("opacity", 1);
      }
    });

    if (animate){
      hub.attr("opacity", 0).attr("transform", "scale(0.5)")
        .transition().duration(850).delay(gi * 60).ease(d3.easeCubicOut)
        .attr("opacity", 1).attr("transform", "scale(1)");
    }
  });

  firstRender = false;
}

try { render(); }
catch (err){
  console.error(err);
  const box = document.createElement("div");
  box.style.cssText = "position:fixed;left:8px;top:8px;z-index:99;background:#fee2e2;color:#991b1b;" +
    "border:1px solid #991b1b;border-radius:4px;padding:6px 8px;font-size:12px;font-weight:600";
  box.textContent = "Could not draw the chart: " + err.message;
  document.body.appendChild(box);
}

let resizeTimer = null;
const requeue = () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(render, 220); };
window.addEventListener("resize", requeue);
if (typeof ResizeObserver !== "undefined") new ResizeObserver(requeue).observe(vizContainer);
</script></body></html>
"""


def build(fetch: bool) -> None:
    cached = {} if fetch else existing_avatars()
    if not fetch and not cached:
        log.warning("no cached avatars found in %s; fetching after all", OUT_HTML)
        fetch = True

    data = []
    for org_type, label, members in GROUPS:
        rows = []
        for m in sorted(members, key=lambda d: -d["dl30"]):
            avatar = cached.get(m["handle"])
            if not avatar:
                log.info("avatar %s", m["handle"])
                avatar = fetch_avatar(m["handle"])
            rows.append({
                "handle": m["handle"], "words": m["words"], "name": m["name"],
                "cc": m["cc"], "country": m["country"], "blurb": m["blurb"], "known": m["known"],
                "datasets": m["datasets"], "models": m["models"],
                "followers": m["followers"], "dl30": m["dl30"],
                "url": f"https://huggingface.co/{m['handle']}", "avatar": avatar,
            })
        data.append({"type": org_type, "label": label, "members": rows})

    # clusters read in order of the type's 30-day download total, biggest first
    data.sort(key=lambda g: -sum(m["dl30"] for m in g["members"]))

    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    html = TEMPLATE.replace("__DATA__", payload).replace("__SNAPSHOT__", SNAPSHOT)
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(html, encoding="utf-8")

    log.info("wrote %s (%.0f KB)", OUT_HTML, OUT_HTML.stat().st_size / 1024)
    for g in data:
        log.info("  %-12s %s accounts, %d downloads/30d",
                 g["type"], len(g["members"]), sum(m["dl30"] for m in g["members"]))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-avatars", action="store_true",
                    help="reuse the avatars already embedded in the built viz")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    build(fetch=not args.no_avatars)
    return 0


if __name__ == "__main__":
    sys.exit(main())
