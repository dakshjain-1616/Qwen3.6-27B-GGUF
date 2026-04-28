"""Render the SVG infographics shipped with the hf_export/ bundle.

Pure stdlib -- no matplotlib, no extra deps. Produces deterministic,
GitHub- and HuggingFace-renderable bar charts under ``charts/`` next to
the model card.
"""

from __future__ import annotations

from pathlib import Path

# Real numbers from the V100 production run + re-bench.
QUANTS = ["Q2_K", "Q4_K_M", "Q5_K_S", "Q8_0"]
SIZES_GB = [10.0, 15.4, 17.4, 26.6]
PPL = [6.8364, 5.9013, 5.7555, 5.7384]
PP512 = [None, 360.89, 402.79, 133.71]   # Q2_K not measured (old pipeline output didn't save pp512)
TG128 = [37.03, 4.88, 4.98, 1.98]

W, H = 720, 420
PAD_L, PAD_R, PAD_T, PAD_B = 80, 30, 60, 70
PLOT_W = W - PAD_L - PAD_R
PLOT_H = H - PAD_T - PAD_B

NEUTRAL = "#4a90e2"
HIGHLIGHT = "#f5a623"   # Q4_K_M (recommended)
TEXT = "#1f2933"
GRID = "#e4e7eb"
AXIS = "#9aa5b1"


def _bar_chart(
    title: str,
    subtitle: str,
    values: list[float | None],
    y_min: float,
    y_max: float,
    y_label: str,
    value_fmt: str = "{:.2f}",
    lower_is_better: bool = False,
) -> str:
    """Render a single labeled bar chart as a self-contained SVG."""
    # Pick best quant by min/max depending on direction; only over non-None values.
    non_null = [(i, v) for i, v in enumerate(values) if v is not None]
    if non_null:
        best_idx = (min if lower_is_better else max)(non_null, key=lambda t: t[1])[0]
    else:
        best_idx = -1

    n = len(values)
    bar_w = PLOT_W / n * 0.55
    gap = (PLOT_W / n) - bar_w

    # Y grid (4 ticks)
    y_ticks = [y_min + (y_max - y_min) * i / 4 for i in range(5)]

    def y_pix(v: float) -> float:
        return PAD_T + PLOT_H * (1.0 - (v - y_min) / (y_max - y_min))

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
        f'font-family="-apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, Arial, sans-serif">'
    )
    parts.append(f'<rect width="{W}" height="{H}" fill="white"/>')
    # Title
    parts.append(
        f'<text x="{W/2}" y="28" text-anchor="middle" font-size="18" font-weight="600" fill="{TEXT}">{title}</text>'
    )
    parts.append(
        f'<text x="{W/2}" y="48" text-anchor="middle" font-size="12" fill="{AXIS}">{subtitle}</text>'
    )
    # Y axis label (rotated)
    parts.append(
        f'<text x="22" y="{PAD_T + PLOT_H/2}" text-anchor="middle" font-size="12" fill="{AXIS}" '
        f'transform="rotate(-90 22 {PAD_T + PLOT_H/2})">{y_label}</text>'
    )
    # Grid + Y ticks
    for t in y_ticks:
        y = y_pix(t)
        parts.append(f'<line x1="{PAD_L}" y1="{y}" x2="{W-PAD_R}" y2="{y}" stroke="{GRID}" stroke-width="1"/>')
        parts.append(
            f'<text x="{PAD_L-8}" y="{y+4}" text-anchor="end" font-size="11" fill="{AXIS}">{t:.1f}</text>'
        )
    # X axis baseline
    base_y = y_pix(y_min)
    parts.append(f'<line x1="{PAD_L}" y1="{base_y}" x2="{W-PAD_R}" y2="{base_y}" stroke="{AXIS}" stroke-width="1.5"/>')

    # Bars
    for i, (label, v) in enumerate(zip(QUANTS, values, strict=True)):
        x = PAD_L + i * (bar_w + gap) + gap / 2
        if v is None:
            # Render an empty slot with "--"
            parts.append(
                f'<text x="{x + bar_w/2}" y="{base_y - 8}" text-anchor="middle" font-size="13" fill="{AXIS}">--</text>'
            )
        else:
            top_y = y_pix(v)
            color = HIGHLIGHT if i == best_idx else NEUTRAL
            parts.append(
                f'<rect x="{x}" y="{top_y}" width="{bar_w}" height="{base_y - top_y}" fill="{color}" rx="3"/>'
            )
            parts.append(
                f'<text x="{x + bar_w/2}" y="{top_y - 6}" text-anchor="middle" font-size="12" font-weight="600" fill="{TEXT}">{value_fmt.format(v)}</text>'
            )
        # X label
        parts.append(
            f'<text x="{x + bar_w/2}" y="{base_y + 22}" text-anchor="middle" font-size="13" fill="{TEXT}">{label}</text>'
        )

    # Legend hint about highlight
    if best_idx >= 0:
        legend_y = H - 18
        parts.append(
            f'<rect x="{PAD_L}" y="{legend_y - 10}" width="14" height="12" fill="{HIGHLIGHT}" rx="2"/>'
        )
        direction = "lowest" if lower_is_better else "highest"
        parts.append(
            f'<text x="{PAD_L + 22}" y="{legend_y}" font-size="12" fill="{AXIS}">{QUANTS[best_idx]} = {direction} value of the four</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts)


def _pipeline_diagram() -> str:
    """A clean pipeline flowchart, hand-laid-out (not generated)."""
    parts: list[str] = []
    parts.append(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 920 380" '
        'font-family="-apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, Arial, sans-serif">'
    )
    parts.append('<rect width="920" height="380" fill="white"/>')
    parts.append(
        '<text x="460" y="28" text-anchor="middle" font-size="18" font-weight="600" fill="#1f2933">'
        "qwen36gguf pipeline -- what built these GGUFs</text>"
    )

    def box(x: int, y: int, w: int, h: int, label: str, sub: str = "", fill: str = "#e3effd", stroke: str = "#4a90e2") -> str:
        out = (
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
            f'<text x="{x + w/2}" y="{y + h/2 + (3 if not sub else -4)}" text-anchor="middle" font-size="13" font-weight="600" fill="#1f2933">{label}</text>'
        )
        if sub:
            out += f'<text x="{x + w/2}" y="{y + h/2 + 14}" text-anchor="middle" font-size="11" fill="#52606d">{sub}</text>'
        return out

    def arrow(x1: int, y1: int, x2: int, y2: int) -> str:
        return (
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#9aa5b1" stroke-width="1.5" marker-end="url(#arr)"/>'
        )

    parts.append(
        '<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">'
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#9aa5b1"/></marker></defs>'
    )

    # Row 1: download -> convert -> quantize
    parts.append(box(40, 70, 180, 70, "Qwen/Qwen3.6-27B", "snapshot_download (HF)"))
    parts.append(arrow(220, 105, 270, 105))
    parts.append(box(270, 70, 180, 70, "convert_hf_to_gguf.py", "-> f16.gguf  (50 GB)"))
    parts.append(arrow(450, 105, 500, 105))
    parts.append(box(500, 70, 180, 70, "llama-quantize", "Q2 / Q4_K_M / Q5_K_S / Q8_0", fill="#fff4e0", stroke="#f5a623"))

    # Branch out: 4 GGUFs
    cy = 230
    quant_x = [40, 250, 460, 670]
    sizes = ["Q2_K - 10 GB", "Q4_K_M - 15.4 GB", "Q5_K_S - 17.4 GB", "Q8_0 - 26.6 GB"]
    for x, label in zip(quant_x, sizes, strict=True):
        parts.append(box(x, cy, 170, 50, label, fill="#fff4e0", stroke="#f5a623"))
        # arrow from quantize box down to each
        parts.append(arrow(590, 140, x + 85, cy))

    # Row 3: bench -> assemble -> push
    parts.append(box(40, 310, 230, 50, "llama-perplexity + llama-bench", "WikiText-2 PPL - pp512 - tg128"))
    parts.append(arrow(270, 335, 320, 335))
    parts.append(box(320, 310, 230, 50, "hf_export/", "README + .gitattributes + GGUFs"))
    parts.append(arrow(550, 335, 600, 335))
    parts.append(box(600, 310, 280, 50, "HfApi.upload_folder", "-> huggingface.co/daksh-neo/...", fill="#e6f6ec", stroke="#3aa55a"))

    # Connector from quants to bench
    parts.append(arrow(125, 280, 125, 310))
    parts.append(arrow(335, 280, 200, 310))
    parts.append(arrow(545, 280, 200, 310))
    parts.append(arrow(755, 280, 200, 310))

    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    # Write to both the tracked GitHub-facing location and the HF bundle.
    targets = [
        Path("assets/charts"),
        Path("out/qwen36-27b/hf_export/charts"),
    ]
    for t in targets:
        t.mkdir(parents=True, exist_ok=True)
    # The first target is the canonical one; the loop at the bottom mirrors to the rest.
    out = targets[0]

    (out / "perplexity.svg").write_text(
        _bar_chart(
            title="Qwen3.6-27B GGUF -- WikiText-2 perplexity",
            subtitle="Lower is better - ctx 512, --parallel 1, V100",
            values=PPL,
            y_min=5.5,
            y_max=7.0,
            y_label="Perplexity",
            value_fmt="{:.3f}",
            lower_is_better=True,
        )
    )

    (out / "filesize.svg").write_text(
        _bar_chart(
            title="Qwen3.6-27B GGUF -- file size on disk",
            subtitle="Smaller is cheaper to host &amp; download",
            values=SIZES_GB,
            y_min=0.0,
            y_max=30.0,
            y_label="Size (GB)",
            value_fmt="{:.1f}",
            lower_is_better=True,
        )
    )

    (out / "throughput_pp.svg").write_text(
        _bar_chart(
            title="Qwen3.6-27B GGUF -- prompt-processing throughput",
            subtitle="llama-bench pp512 - V100 - partial offload sized to fit 16 GB VRAM",
            values=PP512,
            y_min=0.0,
            y_max=500.0,
            y_label="tok/s",
            value_fmt="{:.1f}",
            lower_is_better=False,
        )
    )

    (out / "throughput_tg.svg").write_text(
        _bar_chart(
            title="Qwen3.6-27B GGUF -- token-generation throughput",
            subtitle="llama-bench tg128 - V100 - partial offload (Q2_K is the only one that fully fits)",
            values=TG128,
            y_min=0.0,
            y_max=40.0,
            y_label="tok/s",
            value_fmt="{:.2f}",
            lower_is_better=False,
        )
    )

    (out / "pipeline.svg").write_text(_pipeline_diagram())

    # Mirror the canonical SVGs to every other target (e.g. the hf_export bundle).
    for t in targets[1:]:
        for svg in out.glob("*.svg"):
            (t / svg.name).write_text(svg.read_text())

    # Also render PNG copies — HuggingFace's markdown sanitizer drops <img> refs
    # to *.svg, so the model card needs PNG. We keep both formats so the GitHub
    # README can use crisp SVG and HF can use the PNG fallback.
    try:
        import cairosvg  # type: ignore[import-not-found]
    except ImportError:
        print("cairosvg not installed; skipping PNG render. `uv pip install cairosvg` to enable.")
        return
    for t in targets:
        for svg in t.glob("*.svg"):
            cairosvg.svg2png(url=str(svg), write_to=str(svg.with_suffix(".png")), output_width=1440)

    print(f"wrote {len(list(out.glob('*.svg')))} SVG + PNG pairs to {' & '.join(str(t) for t in targets)}/")


if __name__ == "__main__":
    main()
