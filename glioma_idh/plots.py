"""Small standalone SVG figures without an additional graphics dependency."""

from pathlib import Path
from xml.sax.saxutils import escape

COLORS = ["#285579", "#9a6133", "#547448", "#80548c", "#888888"]


def save_svg_charts(output: Path, curves: list[dict], wildtype: int, mutant: int) -> None:
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="700" height="500" viewBox="0 0 700 500">',
             '<rect width="700" height="500" fill="white"/>',
             '<g font-family="Arial, sans-serif" font-size="13" fill="#222">',
             '<text x="70" y="25" font-size="17">Development out-of-fold ROC curves</text>']
    for tick in range(6):
        value = tick / 5
        x, y = 70 + value * 580, 420 - value * 360
        parts += [f'<line x1="{x}" y1="60" x2="{x}" y2="420" stroke="#ececec"/>',
                  f'<line x1="70" y1="{y}" x2="650" y2="{y}" stroke="#ececec"/>',
                  f'<text x="{x}" y="443" text-anchor="middle">{value:.1f}</text>',
                  f'<text x="59" y="{y + 4}" text-anchor="end">{value:.1f}</text>']
    parts += ['<path d="M70 60 V420 H650" fill="none" stroke="#333"/>',
              '<path d="M70 420 L650 60" fill="none" stroke="#999" stroke-dasharray="6 5"/>',
              '<text x="360" y="480" text-anchor="middle">False positive rate</text>',
              '<text x="19" y="240" transform="rotate(-90 19 240)" text-anchor="middle">True positive rate</text>']
    for index, curve in enumerate(curves):
        color = COLORS[index % len(COLORS)]
        points = " ".join(f"{70 + x * 580:.2f},{420 - y * 360:.2f}" for x, y in zip(curve["fpr"], curve["tpr"]))
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
    parts.append('<rect x="350" y="305" width="295" height="109" fill="white" opacity="0.94"/>')
    for index, curve in enumerate(curves):
        y = 325 + index * 21
        color = COLORS[index % len(COLORS)]
        label = escape(f"{curve['name'].replace('_', ' ')} ({curve['auc']:.3f})")
        parts += [f'<line x1="360" y1="{y - 4}" x2="382" y2="{y - 4}" stroke="{color}" stroke-width="2"/>',
                  f'<text x="390" y="{y}">{label}</text>']
    parts.append('</g></svg>')
    (output / "clinical_development_roc.svg").write_text("\n".join(parts), encoding="utf-8")
    max_count = max(wildtype, mutant, 1)
    bars = ['<svg xmlns="http://www.w3.org/2000/svg" width="650" height="390" viewBox="0 0 650 390">',
            '<rect width="650" height="390" fill="white"/>',
            '<g font-family="Arial, sans-serif" fill="#222" font-size="15">',
            '<text x="50" y="32" font-size="19">Development class distribution</text>',
            '<path d="M60 70 V320 H600" fill="none" stroke="#333"/>']
    for x, count, label, color in [(150, wildtype, "IDH-wildtype", "#536878"), (370, mutant, "IDH-mutant", "#96734e")]:
        height = count / max_count * 220
        bars += [f'<rect x="{x}" y="{320-height}" width="130" height="{height}" fill="{color}"/>',
                 f'<text x="{x+65}" y="{305-height}" text-anchor="middle">{count}</text>',
                 f'<text x="{x+65}" y="350" text-anchor="middle">{label}</text>']
    bars += ['<text x="25" y="195" transform="rotate(-90 25 195)" text-anchor="middle">Patients</text>', '</g></svg>']
    (output / "development_cohort.svg").write_text("\n".join(bars), encoding="utf-8")
