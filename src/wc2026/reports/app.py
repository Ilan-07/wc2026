"""Interactive WC2026 dashboard (single self-contained HTML; Python injects data, JS renders).

Aesthetic: an "intelligence terminal × editorial almanac" — deep ink, a warm trophy-gold signal
accent with a cool data-teal, Fraunces serif for editorial moments over a grotesk body and a mono
for figures, grain + glow, staggered reveals. Interactions: click a team to expand its reasoning
drawer; an R32→Final projected bracket whose nodes open the relevant team's "why".

``build_app(data, generated)`` takes the full forecast payload (see predict.py) and returns the HTML.
"""

from __future__ import annotations

import json

_TEMPLATE = r"""<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="__RELOAD__">
<title>WC2026 · Forecast Terminal</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,900&family=Hanken+Grotesk:wght@400;500;600;700&family=Space+Mono:wght@400;700&display=swap" rel="stylesheet">
<style>
:root{
 --ink:#0a0c10; --panel:#11141b; --panel2:#161a22; --line:#222835; --line2:#2c3340;
 --fg:#eceae3; --mut:#878d9a; --gold:#f3b340; --gold-d:#c98a1f; --teal:#3fd6c0;
 --red:#ff6b6b; --green:#5fd98c;
 --display:'Fraunces',serif; --body:'Hanken Grotesk',system-ui,sans-serif; --mono:'Space Mono',monospace;
}
*{box-sizing:border-box}html{scroll-behavior:smooth}
body{margin:0;background:var(--ink);color:var(--fg);font-family:var(--body);
 font-size:15px;line-height:1.55;-webkit-font-smoothing:antialiased;overflow-x:hidden}
/* atmosphere */
body::before{content:"";position:fixed;inset:0;z-index:0;pointer-events:none;
 background:
  radial-gradient(900px 600px at 78% -8%, rgba(243,179,64,.16), transparent 60%),
  radial-gradient(800px 700px at 12% 8%, rgba(63,214,192,.10), transparent 60%);}
body::after{content:"";position:fixed;inset:0;z-index:0;pointer-events:none;opacity:.035;
 background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='140' height='140'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.9' numOctaves='2'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");}
.wrap{position:relative;z-index:1;max-width:1180px;margin:0 auto;padding:0 22px}
a{color:inherit;text-decoration:none}
.mono{font-family:var(--mono)}
.kicker{font-family:var(--mono);font-size:11px;letter-spacing:.28em;text-transform:uppercase;color:var(--gold)}
/* top bar */
.bar{position:sticky;top:0;z-index:20;display:flex;align-items:center;justify-content:space-between;
 padding:14px 22px;background:rgba(10,12,16,.72);backdrop-filter:blur(12px);border-bottom:1px solid var(--line)}
.bar .logo{font-family:var(--display);font-weight:900;font-size:18px;letter-spacing:-.01em}
.bar .logo b{color:var(--gold)}
.nav{display:flex;gap:4px}
.nav a{font-family:var(--mono);font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:var(--mut);
 padding:7px 11px;border-radius:8px;transition:.18s}
.nav a:hover{color:var(--fg);background:var(--panel2)}
/* hero */
.hero{padding:64px 0 30px;border-bottom:1px solid var(--line);position:relative}
.hero .lab{display:flex;align-items:center;gap:9px;margin-bottom:14px}
.dot{width:8px;height:8px;border-radius:50%;background:var(--green);box-shadow:0 0 0 0 rgba(95,217,140,.7);
 animation:pulse 2.4s infinite}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(95,217,140,.6)}70%{box-shadow:0 0 0 9px rgba(95,217,140,0)}100%{box-shadow:0 0 0 0 rgba(95,217,140,0)}}
.hero h1{font-family:var(--display);font-weight:900;letter-spacing:-.03em;line-height:.92;
 font-size:clamp(54px,11vw,132px);margin:0;background:linear-gradient(180deg,#fff,#cdae74);
 -webkit-background-clip:text;background-clip:text;color:transparent}
.hero .sub{display:flex;flex-wrap:wrap;align-items:baseline;gap:8px 22px;margin-top:18px}
.hero .pct{font-family:var(--mono);font-size:30px;color:var(--gold)}
.hero .pct small{font-size:15px;color:var(--mut)}
.hero .status{color:var(--mut);font-size:13.5px}
.badges{display:flex;flex-wrap:wrap;gap:8px;margin-top:22px}
.badge{font-family:var(--mono);font-size:11px;letter-spacing:.04em;color:var(--mut);
 border:1px solid var(--line2);border-radius:999px;padding:5px 12px;background:var(--panel)}
.badge b{color:var(--teal);font-weight:400}
/* sections */
section{padding:48px 0;border-bottom:1px solid var(--line)}
.h2{font-family:var(--display);font-weight:600;font-size:26px;letter-spacing:-.01em;margin:0 0 4px;display:flex;align-items:baseline;gap:12px}
.h2 .n{font-family:var(--mono);font-size:12px;color:var(--gold);letter-spacing:.1em}
.note{color:var(--mut);font-size:12.5px;margin:0 0 20px;max-width:62ch}
/* forecast table */
.tools{display:flex;justify-content:space-between;align-items:center;gap:14px;margin-bottom:12px;flex-wrap:wrap}
.search{font-family:var(--mono);font-size:12.5px;background:var(--panel);border:1px solid var(--line2);
 color:var(--fg);border-radius:9px;padding:8px 12px;width:210px;outline:none}
.search:focus{border-color:var(--teal)}
.legend{font-family:var(--mono);font-size:11px;color:var(--mut);display:flex;gap:14px}
.tbl{border:1px solid var(--line);border-radius:14px;overflow:hidden;background:var(--panel)}
.row{display:grid;grid-template-columns:34px minmax(0,1fr) 56px 56px 128px 84px 26px;align-items:center;
 gap:12px;padding:13px 16px;border-top:1px solid var(--line);cursor:pointer;transition:background .15s}
.row:first-child{border-top:none}
.row:hover{background:var(--panel2)}
.row.open{background:var(--panel2)}
.row>*{min-width:0}
.rk{font-family:var(--mono);font-size:12px;color:var(--mut)}
.tm{font-weight:600;font-size:15px;display:flex;align-items:center;gap:9px;overflow:hidden;
 white-space:nowrap;text-overflow:ellipsis}
.swatch{width:9px;height:18px;border-radius:3px;background:linear-gradient(var(--g1),var(--g2));flex:0 0 auto}
.num{font-family:var(--mono);font-size:13px;text-align:right;color:var(--mut);white-space:nowrap}
.blend{font-family:var(--mono);font-size:13px;display:flex;align-items:center;gap:9px;
 white-space:nowrap;overflow:hidden}
.bar2{height:7px;border-radius:4px;background:linear-gradient(90deg,var(--gold),var(--gold-d));
 flex:0 0 auto;max-width:60px}
.chip{font-family:var(--mono);font-size:11.5px;text-align:right;white-space:nowrap}
.chip.up{color:var(--red)} .chip.dn{color:var(--green)} .chip.eq{color:var(--mut)}
.chev{justify-self:center;color:var(--mut);transition:transform .2s}
.row.open .chev{transform:rotate(90deg);color:var(--gold)}
/* drawer */
.drawer{display:none;border-top:1px solid var(--line);background:#0d1017;padding:0 16px}
.drawer.open{display:block;animation:fade .35s ease}
@keyframes fade{from{opacity:0;transform:translateY(-6px)}to{opacity:1;transform:none}}
.drawer .inner{display:grid;grid-template-columns:300px 1fr;gap:26px;padding:20px 4px 26px}
@media(max-width:760px){.drawer .inner{grid-template-columns:1fr}.row{grid-template-columns:30px minmax(0,1fr) auto 26px}.hide-sm{display:none}}
.facts dt{font-family:var(--mono);font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;color:var(--mut);margin-top:12px}
.facts dd{margin:2px 0 0;font-size:14px}
.facts dd.warn{color:var(--gold)}
.pipe{position:relative;padding-left:22px}
.pipe::before{content:"";position:absolute;left:6px;top:6px;bottom:6px;width:2px;background:var(--line2)}
.claim{position:relative;margin-bottom:16px}
.claim::before{content:"";position:absolute;left:-19px;top:4px;width:11px;height:11px;border-radius:50%;
 background:var(--ink);border:2px solid var(--teal)}
.claim.judge::before{border-color:var(--gold)}
.claim .role{font-family:var(--mono);font-size:10.5px;letter-spacing:.14em;text-transform:uppercase;color:var(--teal)}
.claim.judge .role{color:var(--gold)}
.claim .txt{font-size:14px;margin:3px 0}
.claim .ev{font-family:var(--mono);font-size:10.5px;color:var(--mut)}
/* group badge */
.gb{font-family:var(--mono);font-size:9px;font-weight:700;width:15px;height:15px;flex:0 0 auto;
 display:inline-flex;align-items:center;justify-content:center;border-radius:4px;background:var(--line2);color:var(--fg)}
/* groups */
.groups{display:grid;grid-template-columns:repeat(auto-fill,minmax(218px,1fr));gap:13px}
.gcard{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:12px 15px}
.gcard .gh{font-family:var(--display);font-weight:600;font-size:14px;margin-bottom:9px;display:flex;align-items:center;gap:9px}
.gcard .gh .gb{width:22px;height:22px;font-size:11.5px;background:var(--gold);color:#1a1205}
.gcard .gh small{font-family:var(--mono);font-size:9.5px;color:var(--mut);letter-spacing:.1em;text-transform:uppercase}
.gt{display:flex;justify-content:space-between;align-items:center;padding:6px 0;border-top:1px solid var(--line);font-size:13px;cursor:pointer}
.gt:first-of-type{border-top:none}
.gt.out{color:var(--mut)} .gt.adv .tn{font-weight:600}
.gt .tn{display:flex;align-items:center;gap:7px}
.gt .q{font-family:var(--mono);font-size:10.5px;color:var(--teal)}
.gt.out .q{color:var(--mut)}
.gt .qbar{height:5px;border-radius:3px;background:var(--teal);opacity:.5}
/* bracket — matches flex-grow so each round centres between its feeder pair */
.bracket{display:flex;gap:10px;overflow-x:auto;padding:6px 2px 24px;align-items:stretch;min-height:760px}
.bcol{display:flex;flex-direction:column;min-width:176px;flex:0 0 auto}
.bcol>.ch{font-family:var(--mono);font-size:9.5px;letter-spacing:.16em;text-transform:uppercase;
 color:var(--mut);text-align:center;height:16px;margin-bottom:6px}
.matches{display:flex;flex-direction:column;justify-content:space-around;flex:1}
.bmatch{flex:1 0 auto;display:flex;flex-direction:column;justify-content:center;gap:5px;position:relative;padding-right:11px}
.bmatch::after{content:"";position:absolute;right:0;top:28%;bottom:28%;width:9px;
 border:1px solid var(--line);border-left:0;border-radius:0 6px 6px 0}
.bcol:nth-last-child(-n+2) .bmatch::after{display:none}
.bnode{position:relative;background:var(--panel);border:1px solid var(--line2);border-radius:7px;
 padding:6px 9px;display:flex;justify-content:space-between;align-items:center;gap:7px;cursor:pointer;
 font-size:12px;transition:.14s}
.bnode:hover{border-color:var(--teal);background:var(--panel2)}
.bnode .nm{display:flex;align-items:center;gap:6px;overflow:hidden;white-space:nowrap;text-overflow:ellipsis}
.bnode .bp{font-family:var(--mono);font-size:10px;color:var(--gold);flex:0 0 auto}
.bnode.win{border-color:var(--gold);background:linear-gradient(90deg,rgba(243,179,64,.12),var(--panel))}
.bnode.lose{opacity:.42}
.champcol{justify-content:center;min-width:160px}
.champ{font-family:var(--display);font-weight:900;font-size:21px;text-align:center;color:var(--gold);
 border:1px dashed var(--gold-d);border-radius:11px;padding:16px 12px;line-height:1.1}
.champ .bp{display:block;font-size:12px;margin-top:4px}
/* split cards */
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:760px){.grid2{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:13px;padding:16px 18px}
.card h3{margin:0 0 3px;font-size:14.5px}
.card .cn{color:var(--mut);font-size:12px;margin:0 0 10px}
.card ul{list-style:none;margin:0;padding:0}
.card li{display:flex;justify-content:space-between;gap:10px;padding:7px 0;border-top:1px solid var(--line);font-size:13.5px}
.card li:first-child{border-top:none}
.card li .v{font-family:var(--mono);font-size:11.5px}
/* latest stream */
.latest{border:1px solid var(--line);border-radius:13px;background:var(--panel);margin-bottom:20px;
 max-height:360px;overflow-y:auto}
.latest::-webkit-scrollbar{width:8px}.latest::-webkit-scrollbar-thumb{background:var(--line2);border-radius:8px}
.litem{display:flex;gap:11px;align-items:flex-start;padding:11px 16px;border-top:1px solid var(--line)}
.litem:first-child{border-top:none}
.litem:hover{background:var(--panel2)}
.litem .mood{width:7px;height:7px;border-radius:50%;margin-top:6px;flex:0 0 auto}
.litem .lt{flex:1;min-width:0}
.litem .lt a{font-size:13.5px;line-height:1.4}.litem .lt a:hover{color:var(--teal)}
.litem .meta{color:var(--mut);font-size:11px;font-family:var(--mono);margin-top:3px}
.litem .tt{color:var(--gold)}
.empty{color:var(--mut);font-size:13px;padding:18px 16px;text-align:center}
/* news */
.feed{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:760px){.feed{grid-template-columns:1fr}}
.fcard{background:var(--panel);border:1px solid var(--line);border-radius:13px;padding:14px 16px}
.fcard .fh{display:flex;justify-content:space-between;align-items:center;margin-bottom:4px}
.fcard .fh b{font-size:14.5px}
.ring{font-family:var(--mono);font-size:11px;padding:2px 9px;border-radius:999px;border:1px solid var(--line2)}
.fcard ul{list-style:none;margin:6px 0 0;padding:0}
.fcard li{padding:6px 0;border-top:1px solid var(--line);font-size:12.5px}
.fcard li a:hover{color:var(--teal)}
.fcard li span{color:var(--mut);font-size:11px}
footer{padding:34px 0 70px;color:var(--mut);font-size:12px;max-width:74ch}
.reveal{opacity:0;transform:translateY(14px);animation:rise .6s cubic-bezier(.2,.7,.2,1) forwards}
@keyframes rise{to{opacity:1;transform:none}}
/* ---------- responsive: tablet & phone ---------- */
@media(max-width:900px){
 .wrap{padding:0 18px}
 .hero{padding:48px 0 26px}
 section{padding:40px 0}
}
@media(max-width:760px){
 .wrap{padding:0 15px}
 .bar{padding:11px 15px}
 .bar .logo{font-size:15px;white-space:nowrap}
 .nav{min-width:0;flex:1;justify-content:flex-end;overflow-x:auto;-webkit-overflow-scrolling:touch;scrollbar-width:none}
 .nav::-webkit-scrollbar{display:none}
 .nav a{padding:6px 8px;font-size:10px;letter-spacing:.08em}
 .h2{font-size:22px;flex-wrap:wrap}
 .note{font-size:12px}
 .row{display:flex;align-items:center;gap:10px;padding:12px 13px}
 .rk{flex:0 0 auto;width:20px}
 .tm{flex:1 1 auto;min-width:0}
 .bar2{display:none}
 .blend{flex:0 0 auto;justify-content:flex-end;color:var(--fg);overflow:visible}
 .chev{flex:0 0 auto;font-size:13px}
 .tools{align-items:stretch}
 .search{width:100%}
 .bracket{min-height:600px}
}
@media(max-width:480px){
 .bar .logo{font-size:13px}
 .nav a{padding:5px 7px;font-size:9.5px;letter-spacing:.05em}
 .hero h1{font-size:clamp(40px,15vw,72px)}
 .hero .pct{font-size:24px}
 .hero .sub{gap:6px 14px}
 section{padding:32px 0}
 .groups{grid-template-columns:1fr}
 .gcard{padding:11px 13px}
}
</style></head>
<body>
<div class="bar">
 <div class="logo">WC<b>26</b> · Forecast Terminal</div>
 <nav class="nav"><a href="#forecast">Forecast</a><a href="#groups">Groups</a><a href="#bracket">Bracket</a><a href="#divergence">Crowd</a><a href="#pulse">Pulse</a></nav>
</div>
<div class="wrap">
 <header class="hero reveal">
  <div class="lab"><span class="dot"></span><span class="kicker" id="kicker"></span></div>
  <h1 id="pick"></h1>
  <div class="sub"><span class="pct" id="pct"></span><span class="status" id="status"></span></div>
  <div class="badges" id="badges"></div>
 </header>

 <section id="forecast">
  <div class="h2"><span class="n">01</span> Title forecast</div>
  <p class="note">Structural simulator blended with the betting market. Click any team to see <b>why</b> — the
   Analyst→Market→Contrarian→Judge reasoning, decomposed and sourced. <b>±</b> is real parameter uncertainty.</p>
  <div class="tools">
   <input class="search" id="search" placeholder="filter teams…">
   <div class="legend hide-sm"><span>model</span><span>market</span><span>blended</span><span>crowd vs model</span></div>
  </div>
  <div class="tbl" id="tbl"></div>
 </section>

 <section id="groups">
  <div class="h2"><span class="n">02</span> The draw</div>
  <p class="note">All 12 groups. Each team's number is its simulated chance of reaching the knockout
   (finishing top-2); the top two are highlighted. Click a team for its reasoning.</p>
  <div class="groups" id="groupsEl"></div>
 </section>

 <section id="bracket">
  <div class="h2"><span class="n">03</span> Projected bracket</div>
  <p class="note">Group standings (from the simulator, incl. altitude) fill the official Round-of-32 template;
   knockout advancement uses the <b>blended, market-anchored</b> title probability, so the stronger side
   advances rather than whoever drew the easier path. Badges show each team's group; click a node for its reasoning.</p>
  <div class="bracket" id="bracketEl"></div>
 </section>

 <section id="divergence">
  <div class="h2"><span class="n">04</span> Crowd vs Model</div>
  <p class="note">Where the simulator and the betting market disagree — the most actionable signal.</p>
  <div class="grid2" id="divEl"></div>
 </section>

 <section id="pulse">
  <div class="h2"><span class="n">05</span> News Pulse <span class="badge" style="align-self:center">display only — not in the forecast</span></div>
  <p class="note">Live headlines pulled from Google News at generation time (re-run <span class="mono">predict</span> to
   refresh). The latest stream is below; per-team mood cards follow. Deliberately excluded from the math.</p>
  <h3 style="font-family:var(--display);font-weight:600;font-size:16px;margin:0 0 10px">Latest across all teams</h3>
  <div class="latest" id="latestEl"></div>
  <h3 style="font-family:var(--display);font-weight:600;font-size:16px;margin:8px 0 10px">By team</h3>
  <div class="feed" id="feedEl"></div>
 </section>

 <section id="track">
  <div class="h2"><span class="n">06</span> Track record</div>
  <p class="note">How this forecast actually performs — out-of-sample, scored with proper rules. Lower RPS is
   better; uniform (1/3 each) is the no-skill baseline.</p>
  <div class="card">
   <div id="trackLive" class="meta" style="margin-bottom:12px"></div>
   <p class="note" style="margin:0">
    <b>Historical validation.</b> 9-tournament out-of-sample W/D/L <b>RPS 0.195 vs 0.234 uniform</b> (skill
    +0.039), positive on <b>8 of 9</b> tournaments (399 matches). Deep-run stage reliability <b>Brier 0.104,
    ECE 0.020</b> across four World Cups; match-level calibration ECE ~0.03. Honest caveat: the betting market
    still edges the model alone (0.190 vs 0.204) — which is why the headline blends <b>25% model / 75% market</b>.
   </p>
  </div>
 </section>

 <footer>
  Generated <span id="gen"></span> &middot; updates automatically through the tournament. Fundamentals + market
  only; venues neutral except hosts; champion probabilities are uncertain estimates (you cannot calibrate a
  champion number on ~3 tournaments). Odds are a snapshot. An analytics tool, not betting advice.
 </footer>
</div>
<script>
const D = __DATA__;
function pctv(x){return (x*100).toFixed(1)+'%';}
const el = (h)=>{const t=document.createElement('template');t.innerHTML=h.trim();return t.content.firstChild;};

// hero
document.getElementById('kicker').textContent = D.kicker;
document.getElementById('pick').textContent = D.pick;
document.getElementById('pct').innerHTML = pctv(D.pickProb)+' <small>± '+pctv(D.pickSd)+' to lift the trophy</small>';
document.getElementById('status').textContent = '· '+D.status;
document.getElementById('badges').innerHTML = D.badges.map(b=>'<span class="badge">'+b+'</span>').join('');
document.getElementById('gen').textContent = D.generated;

// track record (live, display only)
(function(){
  const t = D.track || {}, e = document.getElementById('trackLive');
  if (!e) return;
  if (t.status === 'live'){
    e.innerHTML = '<b>Live &middot; '+t.n_matches+' WC2026 matches scored</b> &mdash; RPS <b>'+t.rps+'</b> vs '
      + t.uniform_rps + ' uniform (skill '+(t.skill>=0?'+':'')+t.skill+') &middot; correct calls '
      + t.calls_correct + '/' + t.n_matches;
  } else {
    e.innerHTML = '<b>Live scoring begins at kickoff</b> &mdash; ' + (t.kickoff || '2026-06-11')
      + '. Each WC2026 match is graded against a rating frozen before the tournament.';
  }
})();

// forecast table
const maxB = Math.max(...D.teams.map(t=>t.blended));
const tbl = document.getElementById('tbl');
function divChip(d){const v=(d*100); const c=v>0.5?'up':v<-0.5?'dn':'eq'; const s=v>0?'+':''; return '<span class="chip '+c+'">'+s+v.toFixed(1)+'pp</span>';}
D.teams.forEach((t,i)=>{
 const row = el('<div class="row" data-team="'+t.team+'" style="animation-delay:'+(i*22)+'ms">'+
  '<div class="rk reveal">'+(i+1)+'</div>'+
  '<div class="tm"><span class="swatch" style="--g1:'+t.g1+';--g2:'+t.g2+'"></span>'+t.team+' <span class="gb">'+t.group+'</span></div>'+
  '<div class="num hide-sm">'+pctv(t.model)+'</div>'+
  '<div class="num hide-sm">'+pctv(t.market)+'</div>'+
  '<div class="blend"><span class="bar2" style="width:'+Math.round(56*t.blended/maxB)+'px"></span>'+pctv(t.blended)+'</div>'+
  '<div class="hide-sm">'+divChip(t.div)+'</div>'+
  '<div class="chev">›</div></div>');
 const drawer = el('<div class="drawer" data-d="'+t.team+'"><div class="inner">'+
  '<dl class="facts">'+
   fact('Coach',t.coach)+fact('Key players',t.keyPlayers.join(', ')||'—')+
   fact('Chemistry',t.chemistry)+fact('Altitude',t.altitude)+
   (t.injuries&&t.injuries.length?fact('Out',t.injuries.join(', '),true):'')+
   fact('Reach SF / Final',pctv(t.reachSF)+' / '+pctv(t.reachFinal))+
  '</dl>'+
  '<div class="pipe">'+t.claims.map(c=>'<div class="claim '+(c.role==="Judge"?"judge":"")+'">'+
    '<div class="role">'+c.role+'</div><div class="txt">'+c.text+'</div><div class="ev">↳ '+c.evidence+'</div></div>').join('')+
  '</div></div></div>');
 row.addEventListener('click',()=>toggle(t.team));
 tbl.appendChild(row); tbl.appendChild(drawer);
});
function fact(k,v,warn){return '<dt>'+k+'</dt><dd'+(warn?' class="warn"':'')+'>'+v+'</dd>';}
function toggle(team){
 const r=document.querySelector('.row[data-team="'+CSS.escape(team)+'"]');
 const d=document.querySelector('.drawer[data-d="'+CSS.escape(team)+'"]');
 const open=d.classList.contains('open');
 document.querySelectorAll('.drawer.open').forEach(x=>x.classList.remove('open'));
 document.querySelectorAll('.row.open').forEach(x=>x.classList.remove('open'));
 if(!open){d.classList.add('open');r.classList.add('open');}
}
function openTeam(team){
 const d=document.querySelector('.drawer[data-d="'+CSS.escape(team)+'"]');
 if(!d){return;} if(!d.classList.contains('open')) toggle(team);
 document.querySelector('.row[data-team="'+CSS.escape(team)+'"]').scrollIntoView({behavior:'smooth',block:'center'});
}
// search
document.getElementById('search').addEventListener('input',e=>{
 const q=e.target.value.toLowerCase();
 document.querySelectorAll('.row').forEach(r=>{
  const show=r.dataset.team.toLowerCase().includes(q);
  r.style.display=show?'':'none';
  const d=document.querySelector('.drawer[data-d="'+CSS.escape(r.dataset.team)+'"]');
  if(!show){d.classList.remove('open');r.classList.remove('open');}
 });
});

// groups
const gel=document.getElementById('groupsEl');
D.groups.forEach(g=>{
 const card=el('<div class="gcard"><div class="gh"><span class="gb">'+g.group+'</span>Group '+g.group+' <small>qualify%</small></div></div>');
 const maxq=Math.max(...g.teams.map(t=>t.qualify),1e-6);
 g.teams.forEach((t,i)=>{
  const row=el('<div class="gt '+(i<2?'adv':'out')+'"><span class="tn"><span class="qbar" style="width:'+Math.round(26*t.qualify/maxq)+'px"></span>'+t.team+'</span><span class="q">'+pctv(t.qualify)+'</span></div>');
  row.addEventListener('click',()=>openTeam(t.team));
  card.appendChild(row);
 });
 gel.appendChild(card);
});

// bracket — pair nodes into matches so rounds align between their feeders
const labels=['Round of 32','Round of 16','Quarterfinals','Semifinals','Final'];
const bel=document.getElementById('bracketEl');
D.bracket.forEach((round,ri)=>{
 const col=el('<div class="bcol"></div>');
 col.appendChild(el('<div class="ch">'+(labels[ri]||'')+'</div>'));
 const matches=el('<div class="matches"></div>');
 for(let i=0;i<round.length;i+=2){
  const m=el('<div class="bmatch"></div>');
  [round[i],round[i+1]].forEach(n=>{
   if(!n)return;
   const node=el('<div class="bnode '+(n.win?'win':'lose')+'"><span class="nm"><span class="gb">'+n.group+'</span>'+n.team+'</span><span class="bp">'+pctv(n.prob)+'</span></div>');
   node.addEventListener('click',()=>openTeam(n.team));
   m.appendChild(node);
  });
  matches.appendChild(m);
 }
 col.appendChild(matches);
 bel.appendChild(col);
});
const cc=el('<div class="bcol champcol"><div class="ch">Champion</div><div class="matches" style="justify-content:center"><div class="champ">'+D.pick+'<span class="bp">'+pctv(D.pickProb)+'</span></div></div></div>');
bel.appendChild(cc);

// divergence
document.getElementById('divEl').innerHTML =
 divCard('Market higher than model','Possible overhype — the crowd rates these above the fundamentals.',D.crowdOver,'up')+
 divCard('Model higher than market','Possible value — the model likes these more than the crowd.',D.crowdUnder,'dn');
function divCard(title,desc,rows,cls){
 return '<div class="card"><h3>'+title+'</h3><p class="cn">'+desc+'</p><ul>'+
  rows.map(r=>'<li><a href="#" onclick="openTeam(\''+r.team.replace(/'/g,"\\'")+'\');return false">'+r.team+'</a>'+
   '<span class="v">mkt '+pctv(r.market)+' · mdl '+pctv(r.model)+' '+divChip(r.div)+'</span></li>').join('')+'</ul></div>';
}
// latest stream (the flowing feed across all teams)
const lat=document.getElementById('latestEl');
if(D.latest && D.latest.length){
 D.latest.forEach(it=>{
  const col=it.score>0?'var(--green)':it.score<0?'var(--red)':'var(--mut)';
  lat.appendChild(el('<div class="litem"><span class="mood" style="background:'+col+'"></span>'+
   '<div class="lt"><a href="'+it.link+'" target="_blank">'+it.title+'</a>'+
   '<div class="meta"><span class="tt">'+it.team+'</span> &middot; '+it.source+' &middot; '+it.date+'</div></div></div>'));
 });
}else{lat.innerHTML='<div class="empty">No headlines loaded — run <b>predict</b> to fetch live news.</div>';}

// news
const feed=document.getElementById('feedEl');
if(!D.news||!D.news.length){feed.innerHTML='<div class="empty">No team news loaded.</div>';}
D.news.forEach(n=>{
 const col=n.mood==='positive'?'var(--green)':n.mood==='negative'?'var(--red)':'var(--mut)';
 const items=n.items.length?n.items.map(it=>'<li><a href="'+it.link+'" target="_blank">'+it.title+'</a><br><span>'+it.source+'</span></li>').join(''):'<li><span>No recent headlines.</span></li>';
 feed.appendChild(el('<div class="fcard"><div class="fh"><b>'+n.team+'</b><span class="ring" style="color:'+col+'">Pulse '+n.pulse+' · '+n.mood+'</span></div><ul>'+items+'</ul></div>'));
});
</script>
</body></html>"""


def build_app(data: dict, generated: str, reload_secs: int = 300) -> str:
    data = {**data, "generated": generated}
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return (_TEMPLATE
            .replace("__RELOAD__", str(int(reload_secs)))
            .replace("__DATA__", blob))
