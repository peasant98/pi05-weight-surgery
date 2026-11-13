"""Side-by-side comparison videos (original vs edited) for social posts / project pages. Needs only PIL + imageio.

    from wsp_vla.viz.compare import Theme, Pair, title_card, spotlight, grid, results_card, write_video
    pair = Pair.from_files("orig.mp4", "edit.mp4", ok_orig=False, ok_edit=True, label="LIBERO-Object", caption="pick up ...")
    write_video("out.mp4", [title_card("..."), spotlight(pair, "background swapped"), results_card(cols, rows)])
Fonts: WSP_FONT / WSP_FONT_BOLD = path[:index] (e.g. HelveticaNeue.ttc:7) or pass Theme(font=..., font_bold=...).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

DEJAVU = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"               # fallback for the check / cross glyphs


def _spec(v, default):
    if not v:
        return default
    path, _, idx = v.partition(":")
    return path, int(idx or 0)


@dataclass
class Theme:
    bg: tuple = (8, 14, 30)                       # deep navy
    accent: tuple = (64, 156, 255)                # edited / success (saturated blue)
    bad: tuple = (255, 120, 120)                  # original failure
    text: tuple = (206, 228, 255)                 # light blue text
    muted: tuple = (138, 168, 214)                # muted blue-grey
    highlight: tuple = (18, 38, 74)
    font: tuple = field(default_factory=lambda: _spec(os.environ.get("WSP_FONT"), ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 0)))
    font_bold: tuple = field(default_factory=lambda: _spec(os.environ.get("WSP_FONT_BOLD") or os.environ.get("WSP_FONT"), (DEJAVU, 0)))
    W: int = 1920
    H: int = 1080
    fps: int = 30

    def f(self, size, bold=False):
        p, i = self.font_bold if bold else self.font
        return ImageFont.truetype(p, size, index=i)


@dataclass
class Pair:
    orig: List[np.ndarray]
    edit: List[np.ndarray]
    ok_orig: bool
    ok_edit: bool
    label: str = ""
    caption: str = ""
    inset: Optional[Image.Image] = None           # e.g. the normal camera view for camera-shift demos

    @staticmethod
    def from_files(orig, edit, ok_orig, ok_edit, label="", caption="", inset=None):
        rd = lambda p: imageio.mimread(p, memtest=False)
        return Pair(rd(orig), rd(edit), ok_orig, ok_edit, label, caption, Image.open(inset) if inset else None)


def pill(d, xy, text, f, fg, anchor="la", pad=10, bg=(0, 0, 0)):
    """Text on a rounded dark pill; a leading check/cross is drawn with a glyph-complete fallback font."""
    if text[:1] in ("✓", "✗"):
        sf = ImageFont.truetype(DEJAVU, f.size); sym, rest = text[0], text[1:]
        ws, wr = d.textlength(sym, font=sf), (d.textlength(rest, font=f) if rest else 0)
        tb = d.textbbox((0, 0), "Hg", font=f, anchor="lm")
        cx, cy = (xy if anchor == "mm" else (xy[0] + (ws + wr) / 2, xy[1] + (tb[3] - tb[1]) / 2))
        x0 = cx - (ws + wr) / 2
        d.rounded_rectangle((x0 - pad, cy + tb[1] - pad * 0.6, x0 + ws + wr + pad, cy + tb[3] + pad * 0.6), 10, fill=bg)
        d.text((x0, cy), sym, font=sf, fill=fg, anchor="lm")
        if rest:
            d.text((x0 + ws, cy), rest, font=f, fill=fg, anchor="lm")
        return
    b = d.textbbox(xy, text, font=f, anchor=anchor)
    d.rounded_rectangle((b[0] - pad, b[1] - pad * 0.6, b[2] + pad, b[3] + pad * 0.6), 10, fill=bg)
    d.text(xy, text, font=f, fill=fg, anchor=anchor)


def _canvas(t):
    im = Image.new("RGB", (t.W, t.H), t.bg)
    return im, ImageDraw.Draw(im)


def _at(v, i):
    return Image.fromarray(v[min(i, len(v) - 1)])


def title_card(title, subtitle="", note="", seconds=2.2, t=Theme()):
    im, d = _canvas(t)
    d.text((t.W / 2, 440), title, font=t.f(72, True), fill=t.text, anchor="mm")
    if subtitle:
        d.text((t.W / 2, 540), subtitle, font=t.f(40), fill=t.muted, anchor="mm")
    if note:
        d.text((t.W / 2, 610), note, font=t.f(34), fill=t.muted, anchor="mm")
    return [np.asarray(im)] * int(seconds * t.fps)


def spotlight(p: Pair, title, seconds=3.2, hold=0.8, names=("original", "+ weight edit"), t=Theme()):
    n = max(len(p.orig), len(p.edit)); T = int(seconds * t.fps); S = 860; out = []
    for k in range(T + int(hold * t.fps)):
        i = min(int(k * n / T), n - 1)
        im, d = _canvas(t)
        for j, (v, name, col, ok) in enumerate(((p.orig, names[0], t.bad, p.ok_orig), (p.edit, names[1], t.accent, p.ok_edit))):
            x, y = 80 + j * (S + 40), 120
            im.paste(_at(v, i).resize((S, S), Image.LANCZOS), (x, y))
            pill(d, (x + 20, y + 20), name, t.f(40, True), col)
            if p.inset is not None:
                im.paste(p.inset.resize((230, 230), Image.LANCZOS), (x + S - 245, y + 15))
                d.rectangle((x + S - 245, y + 15, x + S - 15, y + 245), outline=(255, 255, 255), width=3)
                pill(d, (x + S - 130, y + 268), "normal view", t.f(24), t.text, anchor="mm", pad=6)
            if k >= T or i >= len(v) - 1:
                pill(d, (x + S / 2, y + S / 2), "✓ success" if ok else "✗ failed", t.f(80, True), t.accent if ok else t.bad, anchor="mm", pad=24)
        d.text((t.W / 2, 60), " · ".join(x for x in (title, p.label) if x), font=t.f(40, True), fill=t.text, anchor="mm")
        if p.caption:
            d.text((t.W / 2, t.H - 50), "“" + p.caption + "”", font=t.f(34), fill=t.muted, anchor="mm")
        out.append(np.asarray(im))
    return out


def grid(pairs: List[Pair], title, subtitle="left = original,  right = + weight edit", seconds=4.5, hold=1.2, tally=True, t=Theme()):
    cols, S = 3, 300; pw = 2 * S + 10; x0 = (t.W - cols * pw - 2 * 30) // 2; y0 = 150; T = int(seconds * t.fps); out = []
    for k in range(T + int(hold * t.fps)):
        im, d = _canvas(t)
        d.text((t.W / 2, 55), title, font=t.f(40, True), fill=t.text, anchor="mm")
        d.text((t.W / 2, 105), subtitle, font=t.f(30), fill=t.muted, anchor="mm")
        for q, p in enumerate(pairs[:6]):
            r, c = divmod(q, cols); x, y = x0 + c * (pw + 30), y0 + r * (S + 110)
            for j, (v, ok) in enumerate(((p.orig, p.ok_orig), (p.edit, p.ok_edit))):
                i = min(int(k * len(v) / T), len(v) - 1)
                im.paste(_at(v, i).resize((S, S), Image.LANCZOS), (x + j * (S + 10), y))
                if i >= len(v) - 1:
                    pill(d, (x + j * (S + 10) + S / 2, y + S / 2), "✓" if ok else "✗", t.f(70, True), t.accent if ok else t.bad, anchor="mm", pad=14)
            d.text((x + pw / 2, y + S + 28), p.label, font=t.f(26, True), fill=t.muted, anchor="mm")
        if tally and k >= T:
            no, ne = sum(p.ok_orig for p in pairs[:6]), sum(p.ok_edit for p in pairs[:6])
            pill(d, (t.W / 2, t.H - 45), "this sample:   original %d/%d   ·   edited %d/%d" % (no, len(pairs[:6]), ne, len(pairs[:6])),
                 t.f(36, True), t.text, anchor="mm", pad=14)
        out.append(np.asarray(im))
    return out


def results_card(title, subtitle, columns, rows, footnote="", seconds=4.5, t=Theme()):
    """columns: list of header strings; rows: list of (name, [values], highlight_bool)."""
    im, d = _canvas(t)
    d.text((t.W / 2, 110), title, font=t.f(64, True), fill=t.text, anchor="mm")
    d.text((t.W / 2, 185), subtitle, font=t.f(34), fill=t.muted, anchor="mm")
    xs = np.linspace(640, 1860, len(columns) + 1)
    for i, c in enumerate(columns):
        d.multiline_text(((xs[i] + xs[i + 1]) / 2, 330), c, font=t.f(34, True), fill=t.muted, anchor="mm", align="center")
    d.line((140, 400, 1860, 400), fill=(60, 84, 128), width=3)
    for r, (name, vals, hi) in enumerate(rows):
        y = 490 + r * 170
        if hi:
            d.rounded_rectangle((130, y - 75, 1870, y + 75), 18, fill=t.highlight)
        d.text((160, y), name, font=t.f(46, True), fill=t.accent if hi else t.text, anchor="lm")
        for i, v in enumerate(vals):
            d.text(((xs[i] + xs[i + 1]) / 2, y), v, font=t.f(66, True), fill=t.text, anchor="mm")
    if footnote:
        d.text((t.W / 2, 930), footnote, font=t.f(30), fill=t.muted, anchor="mm")
    return [np.asarray(im)] * int(seconds * t.fps)


def write_video(path, sections, t=Theme(), crf=20):
    w = imageio.get_writer(path, fps=t.fps, codec="libx264", quality=None, pixelformat="yuv420p", macro_block_size=8,
                           output_params=["-crf", str(crf), "-preset", "slow"])
    for sec in sections:
        for fr in sec:
            w.append_data(fr)
    w.close()
    return path
