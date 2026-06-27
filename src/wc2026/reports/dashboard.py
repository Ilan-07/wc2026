"""Static HTML dashboard for the WC2026 forecast (plan P6 / Tier C output).

Renders, with no server and no JS dependencies:
  * the blended forecast (model x market) with stage probabilities markets don't quote,
  * the Crowd-vs-Model divergence table (where market and model disagree),
  * a glanceable news Pulse feed per team — explicitly DISPLAY ONLY, not a forecast input.

Everything is built from plain Python data structures so it can be regenerated on demand.
"""

from __future__ import annotations

import html
from dataclasses import dataclass


@dataclass
class ForecastRow:
    team: str
    model: float
    market: float
    blended: float
    reach_sf: float
    reach_final: float
    model_sd: float = 0.0  # ±1 s.d. of model champion% across the bootstrap ensemble

    @property
    def divergence(self) -> float:
        return self.market - self.model


_CSS = """
:root{--bg:#0d1117;--card:#161b22;--line:#21262d;--fg:#e6edf3;--mut:#8b949e;
--pos:#2ea043;--neg:#f85149;--accent:#58a6ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1100px;margin:0 auto;padding:32px 20px 80px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:17px;margin:34px 0 12px;color:var(--fg)}
.sub{color:var(--mut);margin:0 0 8px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:4px 0;overflow:hidden}
table{width:100%;border-collapse:collapse}
th,td{padding:10px 14px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}
th:first-child,td:first-child{text-align:left}th{color:var(--mut);font-weight:600;font-size:12px;
text-transform:uppercase;letter-spacing:.04em}tr:last-child td{border-bottom:none}
.bar{height:7px;border-radius:4px;background:var(--accent);display:inline-block;vertical-align:middle}
.rank{color:var(--mut);width:28px}
.pos{color:var(--pos)}.neg{color:var(--neg)}.mut{color:var(--mut)}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:760px){.grid{grid-template-columns:1fr}}
.feed{padding:14px 16px}.feed h3{margin:0 0 2px;font-size:15px;display:flex;justify-content:space-between;align-items:center}
.pulse{font-size:12px;padding:2px 8px;border-radius:999px;border:1px solid var(--line)}
.feed ul{margin:8px 0 0;padding-left:0;list-style:none}
.feed li{padding:6px 0;border-top:1px solid var(--line);font-size:13px;color:var(--mut)}
.feed li a{color:var(--fg);text-decoration:none}.feed li a:hover{color:var(--accent)}
.note{color:var(--mut);font-size:12.5px;margin-top:6px}
.badge{display:inline-block;font-size:11px;color:var(--mut);border:1px solid var(--line);
border-radius:6px;padding:2px 7px;margin-left:6px}
.hero{background:linear-gradient(135deg,#161b22,#1b2330);border:1px solid var(--accent);
border-radius:14px;padding:22px 24px;margin:18px 0 8px}
.hero .lab{color:var(--mut);font-size:12px;text-transform:uppercase;letter-spacing:.08em}
.hero .pick{font-size:34px;font-weight:700;margin:4px 0 2px}
.hero .pk{font-size:34px;color:var(--accent)}
.hero .meta{color:var(--mut);font-size:13px;margin-top:6px}
.live{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--pos);
margin-right:6px;vertical-align:middle}
"""


def _pct(x: float) -> str:
    return f"{x:.1%}"


def _div_cell(d: float) -> str:
    cls = "pos" if d < -0.005 else "neg" if d > 0.005 else "mut"
    # market > model (d>0) = crowd higher (potential overhype) -> neg colour
    sign = "+" if d > 0 else ""
    return f'<span class="{cls}">{sign}{d*100:.1f} pp</span>'


def _forecast_table(rows: list[ForecastRow]) -> str:
    mx = max((r.blended for r in rows), default=1.0) or 1.0
    out = [
        "<table><thead><tr><th>#</th><th>Team</th><th>Model (±unc.)</th><th>Market</th>"
        "<th>Blended</th><th>Reach SF</th><th>Reach Final</th><th>Crowd vs Model</th>"
        "</tr></thead><tbody>"
    ]
    for i, r in enumerate(sorted(rows, key=lambda x: x.blended, reverse=True), 1):
        w = int(120 * r.blended / mx)
        band = f" <span class='mut'>±{r.model_sd*100:.1f}</span>" if r.model_sd else ""
        out.append(
            f"<tr><td class='rank'>{i}</td><td>{html.escape(r.team)}</td>"
            f"<td>{_pct(r.model)}{band}</td><td>{_pct(r.market)}</td>"
            f"<td><span class='bar' style='width:{w}px'></span> {_pct(r.blended)}</td>"
            f"<td class='mut'>{_pct(r.reach_sf)}</td><td class='mut'>{_pct(r.reach_final)}</td>"
            f"<td>{_div_cell(r.divergence)}</td></tr>"
        )
    out.append("</tbody></table>")
    return "".join(out)


def _divergence_lists(rows: list[ForecastRow], n: int = 5) -> str:
    by_div = sorted(rows, key=lambda r: r.divergence, reverse=True)
    over = by_div[:n]
    under = list(reversed(by_div[-n:]))

    def block(title, items, desc):
        li = "".join(
            f"<li><a>{html.escape(r.team)}</a> &middot; market {_pct(r.market)} vs model "
            f"{_pct(r.model)} &nbsp; {_div_cell(r.divergence)}</li>"
            for r in items
        )
        return f"<div class='card feed'><h3>{title}</h3><p class='note'>{desc}</p><ul>{li}</ul></div>"

    return (
        "<div class='grid'>"
        + block("Market higher than model", over, "Possible overhype — the crowd rates these above the fundamentals.")
        + block("Model higher than market", under, "Possible value — the model likes these more than the crowd does.")
        + "</div>"
    )


def _news_feed(news: dict) -> str:
    if not news:
        return "<p class='note'>News feed unavailable (offline).</p>"
    cards = []
    for team, tp in news.items():
        color = "var(--pos)" if tp.mood == "positive" else "var(--neg)" if tp.mood == "negative" else "var(--mut)"
        items = "".join(
            f"<li><a href='{html.escape(it.link)}' target='_blank'>{html.escape(it.title)}</a>"
            f"<br><span class='mut'>{html.escape(it.source)} &middot; {html.escape(it.date)}</span></li>"
            for it in tp.items
        ) or "<li class='mut'>No recent headlines.</li>"
        cards.append(
            f"<div class='card feed'><h3>{html.escape(team)}"
            f"<span class='pulse' style='color:{color}'>Pulse {tp.pulse:.0f} &middot; {tp.mood}</span></h3>"
            f"<ul>{items}</ul></div>"
        )
    return "<div class='grid'>" + "".join(cards) + "</div>"


def _hero(headline, status_note: str) -> str:
    """Prominent 'predicted winner' banner with live conditioning status."""
    if headline is None:
        return ""
    pick, prob, sd = headline
    return (
        f"<div class='hero'><div class='lab'><span class='live'></span>Predicted winner — "
        f"live, updates each matchday</div>"
        f"<div class='pick'><span class='pk'>{html.escape(pick)}</span></div>"
        f"<div class='meta'>{prob:.1%} &plusmn; {sd:.1%} to lift the trophy &middot; "
        f"{html.escape(status_note)}</div></div>"
    )


def build_dashboard(
    rows: list[ForecastRow],
    news: dict,
    n_sims: int,
    model_weight: float,
    generated: str,
    odds_note: str,
    headline: tuple[str, float, float] | None = None,
    status_note: str = "",
) -> str:
    """Assemble the full HTML document string."""
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>WC2026 Forecast</title><style>{_CSS}</style></head><body><div class="wrap">
<h1>FIFA World Cup 2026 — Forecast & Intelligence</h1>
<p class="sub">Structural simulator blended with the betting market &middot; generated {html.escape(generated)}
<span class="badge">{n_sims:,} simulations</span>
<span class="badge">blend: {model_weight:.0%} model / {1-model_weight:.0%} market</span></p>
{_hero(headline, status_note)}
<h2>Title forecast</h2>
<div class="card">{_forecast_table(rows)}</div>
<p class="note">Model = Dixon-Coles + Monte Carlo over the real draw. Market = de-vigged outright odds
({html.escape(odds_note)}). Blended = logarithmic opinion pool. <b>Reach SF / Final</b> are stage
probabilities the market does not quote — the simulator's added value. The <b>±unc.</b> on the model
column is ±1 s.d. of the champion probability across a bootstrap ensemble — the <i>real</i> parameter
uncertainty, which is far wider than naive sampling error and the honest measure of how unsure the
model is.</p>

<h2>Crowd vs Model</h2>
{_divergence_lists(rows)}

<h2>News Pulse <span class="badge">display only — not used in the forecast</span></h2>
<p class="note">Live headlines and a heuristic mood score, so you can eyeball the narrative around a
team. Deliberately excluded from the probabilistic forecast (sentiment is noisy and the market
already prices it in).</p>
{_news_feed(news)}

<p class="note" style="margin-top:40px">Forecast is fundamentals + market only. Venues treated as
neutral except hosts (US/Canada/Mexico carry home advantage); knockout third-place allocation is
performance-seeded. Odds are an editable snapshot, not live. This is an analytics tool, not betting advice.</p>
</div></body></html>"""
