"""Build a comparison video from a JSON manifest (any env with PIL + imageio).

manifest = {"sections": [
  {"type": "title", "title": "...", "subtitle": "...", "note": "..."},
  {"type": "spotlight", "title": "background swapped", "pairs": [PAIR, ...]},
  {"type": "grid", "title": "...", "pairs": [PAIR, ...]},
  {"type": "card", "title": "...", "subtitle": "...", "columns": [...], "rows": [[name, [vals], highlight], ...], "footnote": "..."}]}
PAIR = {"orig": "a.mp4", "edit": "b.mp4", "ok_orig": false, "ok_edit": true, "label": "LIBERO-Object", "caption": "...", "inset": "view.png"}
usage: python scripts/make_demo_video.py manifest.json out.mp4
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wsp_vla.viz.compare import Pair, Theme, grid, results_card, spotlight, title_card, write_video  # noqa: E402


def build(manifest, t=Theme()):
    out = []
    for s in manifest["sections"]:
        pairs = [Pair.from_files(**p) for p in s.get("pairs", [])]
        if s["type"] == "title":
            out.append(title_card(s["title"], s.get("subtitle", ""), s.get("note", ""), s.get("seconds", 2.2), t=t))
        elif s["type"] == "spotlight":
            out += [spotlight(p, s["title"], t=t) for p in pairs]
        elif s["type"] == "grid":
            out.append(grid(pairs, s["title"], t=t))
        elif s["type"] == "card":
            out.append(results_card(s["title"], s["subtitle"], s["columns"], [tuple(r) for r in s["rows"]], s.get("footnote", ""), t=t))
    return out


if __name__ == "__main__":
    m = json.loads(Path(sys.argv[1]).read_text())
    print("wrote", write_video(sys.argv[2], build(m)))
