"""Giao diện: CSS + các hàm dựng HTML (không phụ thuộc Streamlit)."""
import base64
import html
import io
import math

FONT_IMPORT = ("<style>@import url('https://fonts.googleapis.com/css2?"
               "family=Be+Vietnam+Pro:wght@300;400;500;600;700;800&display=swap');</style>")

CSS = """
<style>
:root{
  --ink:#14213D; --ink-2:#4A5872; --ink-3:#8492A8;
  --paper:#F2F4F7; --panel:#FFFFFF; --line:#E1E6EE;
  --real:#0B8F7B; --real-tint:#E4F4F0;
  --ai:#6D28D9;   --ai-tint:#EFE8FB;
  --amber:#E0A21B;
}
.stApp{background:var(--paper);}
.stApp, .stApp p, .stApp label, .stApp button, .stApp input, .stApp [data-baseweb], .stApp small{
  font-family:'Be Vietnam Pro',system-ui,-apple-system,'Segoe UI',sans-serif;}
[data-testid="stHeader"]{background:transparent;}
[data-testid="stSidebar"], [data-testid="collapsedControl"]{display:none;}
#MainMenu, footer{visibility:hidden;}
.block-container{max-width:1160px; padding-top:1.4rem; padding-bottom:4rem;}

/* ── masthead ── */
.mast{display:flex; align-items:center; gap:10px; margin-bottom:26px;}
.mast span{font-weight:700; font-size:1.05rem; letter-spacing:-.01em; color:var(--ink);}
.headline{font-weight:800; font-size:clamp(2rem,4.2vw,3.1rem); line-height:1.08;
  letter-spacing:-.03em; color:var(--ink); margin:0 0 12px 0; max-width:20em; text-wrap:balance;}
.lede{color:var(--ink-2); font-size:1.05rem; line-height:1.6; max-width:38em; margin:0 0 30px 0;}

/* ── uploader ── */
[data-testid="stFileUploaderDropzone"]{
  background:var(--panel); border:1.5px dashed #9AA6B8; border-radius:16px; padding:30px 24px;}
[data-testid="stFileUploaderDropzone"]:hover{border-color:var(--ink); background:#FBFCFD;}
[data-testid="stFileUploader"] small{color:var(--ink-3);}

/* ── image frame with crop marks ── */
.frame{position:relative; margin:26px 8px 10px 8px;}
.frame .inner{position:relative; overflow:hidden; border-radius:4px; background:#DDE2EA; line-height:0;
  box-shadow:0 1px 2px rgba(20,33,61,.08), 0 14px 34px -14px rgba(20,33,61,.35);}
.frame img{width:100%; display:block; max-height:520px; object-fit:contain; background:#DDE2EA;}
.cm{position:absolute; width:20px; height:20px; border:0 solid var(--ink);}
.cm.tl{top:-9px; left:-9px; border-top-width:2px; border-left-width:2px;}
.cm.tr{top:-9px; right:-9px; border-top-width:2px; border-right-width:2px;}
.cm.bl{bottom:-9px; left:-9px; border-bottom-width:2px; border-left-width:2px;}
.cm.br{bottom:-9px; right:-9px; border-bottom-width:2px; border-right-width:2px;}
.sl{position:absolute; left:0; right:0; top:0; height:3px; opacity:0; background:#2EE6C5;
  box-shadow:0 0 22px 8px rgba(46,230,197,.55);}
.scan .sl{animation:scan 1.5s cubic-bezier(.5,0,.3,1) 1 both;}
@keyframes scan{0%{top:0; opacity:1} 90%{opacity:1} 100%{top:100%; opacity:0}}
.fcap{color:var(--ink-3); font-size:.82rem; margin:8px 10px 0 10px; display:flex; justify-content:space-between;}

/* ── panels ── */
.panel{background:var(--panel); border:1px solid var(--line); border-radius:16px; padding:6px 22px;
  margin-bottom:16px;}
.panel-title{font-weight:700; font-size:1rem; color:var(--ink); margin:2px 0 10px 2px;}

/* ── verdict ── */
.verdict{display:flex; align-items:center; gap:22px; flex-wrap:wrap; border-radius:16px;
  padding:20px 24px; margin-bottom:16px; border:1px solid transparent;}
.verdict.real{background:var(--real-tint); border-color:rgba(11,143,123,.3);}
.verdict.ai{background:var(--ai-tint); border-color:rgba(109,40,217,.28);}
.verdict .dial{flex:0 0 232px; max-width:100%;}
.verdict .dial svg{width:100%; height:auto; display:block;}
.verdict .vtext{flex:1 1 200px; min-width:0;}
.verdict h2{margin:0 0 6px 0; padding:0; font-size:1.55rem; font-weight:800; letter-spacing:-.02em; line-height:1.2;}
.verdict.real h2{color:#07695B;} .verdict.ai h2{color:#5219AE;}
.verdict p{margin:0; color:var(--ink-2); font-size:.95rem; line-height:1.55;}
.verdict p b{color:var(--ink);}
.needle{animation:sweep 1.2s cubic-bezier(.2,.9,.25,1.05) both;}
@keyframes sweep{from{transform:rotate(-180deg)}}

/* ── model rows ── */
.mhead, .mrow{display:grid; grid-template-columns:minmax(190px,1.5fr) minmax(90px,1fr) minmax(92px,.75fr);
  gap:16px; align-items:center;}
.mhead{padding:8px 0 10px 0; color:var(--ink-3); font-size:.8rem; border-bottom:1px solid var(--line);}
.mhead .axis{display:flex; justify-content:space-between;}
.mhead .axis i{font-style:normal;}
.mhead .axis i:first-child{color:var(--real);} .mhead .axis i:last-child{color:var(--ai);}
.mhead .r{text-align:right;}
.mrow{padding:14px 0; border-bottom:1px solid var(--line);}
.mrow:last-child{border-bottom:0;}
.mname{display:flex; align-items:center; gap:12px; min-width:0;}
.mono{flex:0 0 auto; width:40px; height:40px; border-radius:10px; background:var(--ink); color:#fff;
  display:flex; align-items:center; justify-content:center; font-weight:700; font-size:.78rem; letter-spacing:.02em;}
.mrow.off .mono{background:#C7CEDA;}
.mname b{display:block; font-size:.95rem; color:var(--ink); line-height:1.25;}
.mname small{display:block; color:var(--ink-3); font-size:.78rem; margin-top:2px;}
.mtrack{position:relative; height:10px; border-radius:6px; background:#E8ECF2;}
.mtrack::after{content:""; position:absolute; left:50%; top:-4px; bottom:-4px; width:2px; margin-left:-1px; background:var(--ink-3); opacity:.6;}
.mfill{position:absolute; top:0; bottom:0; border-radius:6px;}
.mfill.real{background:var(--real);} .mfill.ai{background:var(--ai);}
.mres{text-align:right; line-height:1.25;}
.mres b{display:block; font-size:.98rem; font-weight:700;}
.mres small{color:var(--ink-3); font-size:.78rem;}
.mres.real b{color:var(--real);} .mres.ai b{color:var(--ai);}
.mstat{text-align:right; font-size:.85rem; color:var(--ink-2); display:flex; justify-content:flex-end; align-items:center; gap:8px;}
.dot{width:9px; height:9px; border-radius:50%; background:#22C55E; flex:0 0 auto;}
.dot.off{background:#C7CEDA;} .dot.err{background:#F59E0B;}
.mnote{grid-column:1 / -1; color:var(--ink-3); font-size:.8rem; margin-top:-6px;}

/* ── controls ── */
.stButton > button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"]{
  background:var(--ink); border:0; border-radius:12px; font-weight:700; padding:.72rem 1rem;
  letter-spacing:.005em; box-shadow:0 8px 20px -8px rgba(20,33,61,.55);}
.stButton > button[kind="primary"]:hover{background:#0B1530; transform:translateY(-1px);}
[data-testid="stSegmentedControl"] button{font-weight:600;}
[data-testid="stSelectbox"] div[data-baseweb="select"] > div{border-radius:10px;}
[data-testid="stAlert"]{border-radius:12px;}

@media (prefers-reduced-motion:reduce){
  .needle, .scan .sl{animation:none;} .sl{display:none;}
}
@media (max-width:640px){
  .mhead, .mrow{grid-template-columns:1fr; gap:8px;}
  .mhead{display:none;} .mres,.mstat{text-align:left; justify-content:flex-start;}
}
</style>
"""

MARK = ('<svg width="30" height="30" viewBox="0 0 32 32" aria-hidden="true">'
        '<circle cx="14" cy="14" r="10" fill="none" stroke="#14213D" stroke-width="3"/>'
        '<path d="M14 4 A10 10 0 0 1 14 24 Z" fill="#14213D"/>'
        '<path d="M21.5 21.5 L29 29" stroke="#14213D" stroke-width="3.6" stroke-linecap="round"/></svg>')


def masthead():
    return f'<div class="mast">{MARK}<span>Soi ảnh</span></div>'


def headline():
    return ('<h1 class="headline">Bức ảnh này là thật hay do AI tạo ra?</h1>'
            '<p class="lede">Tải ảnh lên, chọn một mô hình hoặc chạy cả bốn cùng lúc '
            'để xem các mô hình có đồng ý với nhau không.</p>')


def image_data_uri(img, max_side=1100):
    im = img.copy()
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=86)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def image_frame(img, name, scanning=False):
    uri = image_data_uri(img)
    return (f'<div class="frame{" scan" if scanning else ""}">'
            '<span class="cm tl"></span><span class="cm tr"></span>'
            '<span class="cm bl"></span><span class="cm br"></span>'
            f'<div class="inner"><img src="{uri}" alt="Ảnh đã tải lên"><div class="sl"></div></div></div>'
            f'<div class="fcap"><span>{html.escape(name)}</span><span>{img.width}×{img.height} px</span></div>')


def dial_svg(p_ai):
    p = min(max(p_ai, 0.0), 1.0)
    rot = -180 * (1 - p)
    ticks = ""
    for i in range(5):
        a = math.pi * (1 - i / 4)
        x1, y1 = 150 + 104 * math.cos(a), 150 - 104 * math.sin(a)
        x2, y2 = 150 + 112 * math.cos(a), 150 - 112 * math.sin(a)
        ticks += (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                  'stroke="#14213D" stroke-opacity=".35" stroke-width="2" stroke-linecap="round"/>')
    return (
        '<svg viewBox="0 0 300 182" role="img" aria-label="Đồng hồ xác suất AI">'
        '<defs><linearGradient id="g" gradientUnits="userSpaceOnUse" x1="24" y1="0" x2="276" y2="0">'
        '<stop offset="0" stop-color="#0B8F7B"/><stop offset=".5" stop-color="#E0A21B"/>'
        '<stop offset="1" stop-color="#6D28D9"/></linearGradient></defs>'
        '<path d="M24 150 A126 126 0 0 1 276 150" fill="none" stroke="url(#g)" stroke-width="15" stroke-linecap="round"/>'
        f'{ticks}'
        f'<g class="needle" style="transform:rotate({rot:.1f}deg);transform-origin:150px 150px">'
        '<path d="M150 145 L246 150 L150 155 Z" fill="#14213D"/></g>'
        '<circle cx="150" cy="150" r="11" fill="#14213D"/><circle cx="150" cy="150" r="4" fill="#fff"/>'
        '<text x="24" y="176" text-anchor="middle" font-size="13" font-weight="600" fill="#0B8F7B" '
        'font-family="Be Vietnam Pro,system-ui,sans-serif">Thật</text>'
        '<text x="276" y="176" text-anchor="middle" font-size="13" font-weight="600" fill="#6D28D9" '
        'font-family="Be Vietnam Pro,system-ui,sans-serif">AI</text></svg>')


def verdict(is_ai, p_ai, detail):
    cls = "ai" if is_ai else "real"
    title = "Nhiều khả năng là ảnh do AI tạo ra" if is_ai else "Nhiều khả năng là ảnh thật"
    return (f'<div class="verdict {cls}"><div class="dial">{dial_svg(p_ai)}</div>'
            f'<div class="vtext"><h2>{title}</h2><p>{detail}</p></div></div>')


def _mono(abbr):
    return f'<div class="mono">{html.escape(abbr)}</div>'


def rows_panel(items, title=None, with_head=True):
    """items: list of dict(name, abbr, desc, state, res). state: result|ready|missing|error."""
    out = ['<div class="panel">']
    if title:
        out.append(f'<div class="panel-title" style="padding-top:12px">{html.escape(title)}</div>')
    if with_head:
        out.append('<div class="mhead"><span>Mô hình</span>'
                   '<span class="axis"><i>Thật</i><i>AI</i></span><span class="r">Kết luận</span></div>')
    for it in items:
        name, abbr, desc, state, res = it["name"], it["abbr"], it["desc"], it["state"], it.get("res")
        if state == "result":
            p = min(max(res["p_ai"], 0.0), 1.0)
            is_ai = res["label"] == 1
            cls = "ai" if is_ai else "real"
            conf = p if is_ai else 1 - p
            left, width = (50, (p - .5) * 100) if p >= .5 else (p * 100, (.5 - p) * 100)
            out.append(
                f'<div class="mrow">'
                f'<div class="mname">{_mono(abbr)}<div><b>{html.escape(name)}</b>'
                f'<small>{html.escape(desc)} · {res["ms"]:.0f} ms</small></div></div>'
                f'<div class="mtrack"><i class="mfill {cls}" style="left:{left:.1f}%;width:{max(width, 1.5):.1f}%"></i></div>'
                f'<div class="mres {cls}"><b>{"AI tạo ra" if is_ai else "Ảnh thật"}</b>'
                f'<small>tin cậy {conf:.0%}</small></div></div>')
        else:
            label, dcls = {"ready": ("Sẵn sàng", ""), "missing": ("Chưa có file", " off"),
                           "error": ("Không chạy được", " err")}[state]
            note = (f'<div class="mnote">{html.escape(str(res.get("error", "")))}</div>'
                    if state == "error" and res else "")
            out.append(
                f'<div class="mrow{" off" if state != "ready" else ""}">'
                f'<div class="mname">{_mono(abbr)}<div><b>{html.escape(name)}</b>'
                f'<small>{html.escape(desc)}</small></div></div>'
                f'<div class="mtrack"></div>'
                f'<div class="mstat"><span class="dot{dcls}"></span>{label}</div>{note}</div>')
    out.append('</div>')
    return "".join(out)
