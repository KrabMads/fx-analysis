import { useState, useRef, useEffect, useCallback } from "react";

const RAW_DATA = [{"ccy":"GBP","strategy":"Unhedged","mean":79.57,"std":5.311,"var10":72.374,"cvar10":71.098,"worst":69.591,"dd3m":212.409,"sharpe":0.0,"avg_sigma":6.61,"avg_carry":-1.46},{"ccy":"GBP","strategy":"Forward","mean":79.824,"std":5.352,"var10":72.146,"cvar10":70.802,"worst":69.222,"dd3m":211.374,"sharpe":0.0474,"avg_sigma":6.61,"avg_carry":-1.46},{"ccy":"GBP","strategy":"Risk Reversal","mean":79.564,"std":5.135,"var10":72.266,"cvar10":70.984,"worst":69.591,"dd3m":212.409,"sharpe":-0.0011,"avg_sigma":6.61,"avg_carry":-1.46},{"ccy":"GBP","strategy":"Blend 50/50","mean":79.694,"std":5.198,"var10":71.997,"cvar10":71.096,"worst":69.9,"dd3m":212.434,"sharpe":0.0238,"avg_sigma":6.61,"avg_carry":-1.46},{"ccy":"GBP","strategy":"Fwd Layered","mean":79.717,"std":5.261,"var10":72.065,"cvar10":71.27,"worst":70.587,"dd3m":212.337,"sharpe":0.028,"avg_sigma":6.61,"avg_carry":-1.46},{"ccy":"GBP","strategy":"RR Layered","mean":79.55,"std":5.168,"var10":72.512,"cvar10":71.255,"worst":70.47,"dd3m":212.657,"sharpe":-0.0038,"avg_sigma":6.61,"avg_carry":-1.46},{"ccy":"GBP","strategy":"Blend Layered","mean":79.634,"std":5.192,"var10":72.222,"cvar10":71.295,"worst":70.906,"dd3m":213.082,"sharpe":0.0123,"avg_sigma":6.61,"avg_carry":-1.46},{"ccy":"SEK","strategy":"Unhedged","mean":55.289,"std":3.919,"var10":50.302,"cvar10":49.595,"worst":48.981,"dd3m":148.217,"sharpe":0.0,"avg_sigma":7.91,"avg_carry":1.32},{"ccy":"SEK","strategy":"Forward","mean":55.482,"std":4.018,"var10":50.447,"cvar10":49.676,"worst":49.095,"dd3m":148.507,"sharpe":0.0479,"avg_sigma":7.91,"avg_carry":1.32},{"ccy":"SEK","strategy":"Risk Reversal","mean":55.377,"std":3.846,"var10":50.518,"cvar10":49.946,"worst":49.227,"dd3m":149.551,"sharpe":0.0229,"avg_sigma":7.91,"avg_carry":1.32},{"ccy":"SEK","strategy":"Blend 50/50","mean":55.43,"std":3.887,"var10":50.412,"cvar10":49.946,"worst":49.161,"dd3m":149.348,"sharpe":0.0361,"avg_sigma":7.91,"avg_carry":1.32},{"ccy":"SEK","strategy":"Fwd Layered","mean":55.391,"std":3.828,"var10":50.492,"cvar10":49.899,"worst":49.464,"dd3m":149.105,"sharpe":0.0266,"avg_sigma":7.91,"avg_carry":1.32},{"ccy":"SEK","strategy":"RR Layered","mean":55.338,"std":3.798,"var10":50.337,"cvar10":49.925,"worst":49.424,"dd3m":149.605,"sharpe":0.0128,"avg_sigma":7.91,"avg_carry":1.32},{"ccy":"SEK","strategy":"Blend Layered","mean":55.365,"std":3.792,"var10":50.326,"cvar10":49.987,"worst":49.606,"dd3m":149.527,"sharpe":0.0198,"avg_sigma":7.91,"avg_carry":1.32},{"ccy":"USD","strategy":"Unhedged","mean":66.939,"std":2.719,"var10":62.816,"cvar10":61.215,"worst":60.312,"dd3m":182.231,"sharpe":0.0,"avg_sigma":7.2,"avg_carry":-1.25},{"ccy":"USD","strategy":"Forward","mean":67.096,"std":2.351,"var10":63.918,"cvar10":62.243,"worst":60.063,"dd3m":183.026,"sharpe":0.0665,"avg_sigma":7.2,"avg_carry":-1.25},{"ccy":"USD","strategy":"Risk Reversal","mean":66.918,"std":2.287,"var10":64.186,"cvar10":62.086,"worst":60.679,"dd3m":183.279,"sharpe":-0.0093,"avg_sigma":7.2,"avg_carry":-1.25},{"ccy":"USD","strategy":"Blend 50/50","mean":67.007,"std":2.23,"var10":64.002,"cvar10":62.27,"worst":60.711,"dd3m":183.152,"sharpe":0.0303,"avg_sigma":7.2,"avg_carry":-1.25},{"ccy":"USD","strategy":"Fwd Layered","mean":67.036,"std":2.298,"var10":64.005,"cvar10":62.049,"worst":60.573,"dd3m":182.46,"sharpe":0.0422,"avg_sigma":7.2,"avg_carry":-1.25},{"ccy":"USD","strategy":"RR Layered","mean":66.937,"std":2.394,"var10":63.558,"cvar10":61.739,"worst":60.679,"dd3m":183.017,"sharpe":-0.0009,"avg_sigma":7.2,"avg_carry":-1.25},{"ccy":"USD","strategy":"Blend Layered","mean":66.987,"std":2.307,"var10":63.754,"cvar10":61.967,"worst":60.735,"dd3m":182.84,"sharpe":0.0205,"avg_sigma":7.2,"avg_carry":-1.25},{"ccy":"NOK","strategy":"Unhedged","mean":54.244,"std":4.604,"var10":49.262,"cvar10":48.825,"worst":48.421,"dd3m":146.215,"sharpe":0.0,"avg_sigma":8.37,"avg_carry":0.07},{"ccy":"NOK","strategy":"Forward","mean":54.913,"std":4.884,"var10":49.645,"cvar10":48.846,"worst":48.366,"dd3m":146.426,"sharpe":0.1369,"avg_sigma":8.37,"avg_carry":0.07},{"ccy":"NOK","strategy":"Risk Reversal","mean":54.702,"std":4.673,"var10":49.878,"cvar10":49.366,"worst":48.859,"dd3m":148.108,"sharpe":0.0979,"avg_sigma":8.37,"avg_carry":0.07},{"ccy":"NOK","strategy":"Blend 50/50","mean":54.807,"std":4.736,"var10":49.605,"cvar10":49.343,"worst":48.948,"dd3m":148.141,"sharpe":0.1189,"avg_sigma":8.37,"avg_carry":0.07},{"ccy":"NOK","strategy":"Fwd Layered","mean":54.687,"std":4.727,"var10":49.687,"cvar10":49.179,"worst":48.785,"dd3m":147.452,"sharpe":0.0938,"avg_sigma":8.37,"avg_carry":0.07},{"ccy":"NOK","strategy":"RR Layered","mean":54.541,"std":4.652,"var10":49.794,"cvar10":49.288,"worst":48.783,"dd3m":147.781,"sharpe":0.0637,"avg_sigma":8.37,"avg_carry":0.07},{"ccy":"NOK","strategy":"Blend Layered","mean":54.614,"std":4.668,"var10":49.909,"cvar10":49.326,"worst":48.978,"dd3m":147.975,"sharpe":0.0792,"avg_sigma":8.37,"avg_carry":0.07},{"ccy":"AUD","strategy":"Unhedged","mean":68.728,"std":7.593,"var10":59.745,"cvar10":59.17,"worst":58.739,"dd3m":178.116,"sharpe":0.0,"avg_sigma":9.0,"avg_carry":-0.32},{"ccy":"AUD","strategy":"Forward","mean":68.017,"std":7.012,"var10":59.737,"cvar10":59.18,"worst":58.732,"dd3m":178.255,"sharpe":-0.1014,"avg_sigma":9.0,"avg_carry":-0.32},{"ccy":"AUD","strategy":"Risk Reversal","mean":68.306,"std":7.142,"var10":60.112,"cvar10":59.37,"worst":59.022,"dd3m":178.555,"sharpe":-0.059,"avg_sigma":9.0,"avg_carry":-0.32},{"ccy":"AUD","strategy":"Blend 50/50","mean":68.162,"std":7.031,"var10":59.901,"cvar10":59.452,"worst":58.96,"dd3m":179.146,"sharpe":-0.0805,"avg_sigma":9.0,"avg_carry":-0.32},{"ccy":"AUD","strategy":"Fwd Layered","mean":68.203,"std":7.03,"var10":60.131,"cvar10":59.822,"worst":59.411,"dd3m":179.367,"sharpe":-0.0746,"avg_sigma":9.0,"avg_carry":-0.32},{"ccy":"AUD","strategy":"RR Layered","mean":68.383,"std":7.185,"var10":59.893,"cvar10":59.611,"worst":59.213,"dd3m":179.342,"sharpe":-0.048,"avg_sigma":9.0,"avg_carry":-0.32},{"ccy":"AUD","strategy":"Blend Layered","mean":68.293,"std":7.084,"var10":60.039,"cvar10":59.872,"worst":59.724,"dd3m":179.748,"sharpe":-0.0614,"avg_sigma":9.0,"avg_carry":-0.32},{"ccy":"CAD","strategy":"Unhedged","mean":101.638,"std":8.262,"var10":87.85,"cvar10":83.28,"worst":79.219,"dd3m":240.464,"sharpe":0.0,"avg_sigma":7.6,"avg_carry":-0.92},{"ccy":"CAD","strategy":"Forward","mean":99.965,"std":9.665,"var10":84.371,"cvar10":78.911,"worst":76.309,"dd3m":233.454,"sharpe":-0.1731,"avg_sigma":7.6,"avg_carry":-0.92},{"ccy":"CAD","strategy":"Risk Reversal","mean":100.828,"std":8.963,"var10":86.325,"cvar10":80.715,"worst":78.176,"dd3m":238.488,"sharpe":-0.0903,"avg_sigma":7.6,"avg_carry":-0.92},{"ccy":"CAD","strategy":"Blend 50/50","mean":100.396,"std":9.277,"var10":85.348,"cvar10":79.813,"worst":77.243,"dd3m":235.971,"sharpe":-0.1338,"avg_sigma":7.6,"avg_carry":-0.92},{"ccy":"CAD","strategy":"Fwd Layered","mean":100.525,"std":9.107,"var10":85.88,"cvar10":80.421,"worst":77.879,"dd3m":235.939,"sharpe":-0.1221,"avg_sigma":7.6,"avg_carry":-0.92},{"ccy":"CAD","strategy":"RR Layered","mean":101.135,"std":8.679,"var10":87.103,"cvar10":81.732,"worst":78.871,"dd3m":239.305,"sharpe":-0.0579,"avg_sigma":7.6,"avg_carry":-0.92},{"ccy":"CAD","strategy":"Blend Layered","mean":100.83,"std":8.875,"var10":86.492,"cvar10":81.076,"worst":78.375,"dd3m":237.622,"sharpe":-0.091,"avg_sigma":7.6,"avg_carry":-0.92}];

const METRICS = [
  { key:"mean",      label:"Mean Monthly DKK",    unit:"M DKK", hi:true  },
  { key:"std",       label:"Std Dev",              unit:"M DKK", hi:false },
  { key:"var10",     label:"VaR-10%",              unit:"M DKK", hi:true  },
  { key:"cvar10",    label:"CVaR-10%",             unit:"M DKK", hi:true  },
  { key:"worst",     label:"Worst Month",          unit:"M DKK", hi:true  },
  { key:"dd3m",      label:"Max 3M Drawdown",      unit:"M DKK", hi:true  },
  { key:"sharpe",    label:"Sharpe vs Unhedged",   unit:"",      hi:true  },
  { key:"avg_sigma", label:"Avg Realised Vol",     unit:"%",     hi:false },
  { key:"avg_carry", label:"Avg Carry",            unit:"%",     hi:true  },
];

const CCY_COLORS   = { GBP:"#60a5fa", SEK:"#34d399", USD:"#fbbf24", NOK:"#f472b6", AUD:"#a78bfa", CAD:"#fb923c" };
const STRAT_COLORS = {
  "Unhedged":      "#64748b",
  "Forward":       "#2563eb",
  "Risk Reversal": "#059669",
  "Blend 50/50":   "#d97706",
  "Fwd Layered":   "#93c5fd",
  "RR Layered":    "#6ee7b7",
  "Blend Layered": "#fcd34d",
};

// Dot labels — 2-char abbreviations
const STRAT_ABBR = {
  "Unhedged":"UH","Forward":"FW","Risk Reversal":"RR","Blend 50/50":"BL",
  "Fwd Layered":"FL","RR Layered":"RL","Blend Layered":"BX",
};

// Whether a strategy is "layered" (dashed border)
const IS_LAYERED = s => s.includes("Layered");

const ALL_CCYS   = Object.keys(CCY_COLORS);
const ALL_STRATS = Object.keys(STRAT_COLORS);

function fmt(v, key) {
  const m = METRICS.find(x => x.key === key);
  if (!m) return v.toFixed(2);
  if (m.unit === "%") return v.toFixed(2) + "%";
  if (m.unit === "M DKK") return v.toFixed(1) + "M";
  return v.toFixed(4);
}

// ── Scatter chart rendered in SVG ──────────────────────────────────────────
function ScatterSVG({ data, xKey, yKey, colorBy, hoveredId, setHoveredId }) {
  const [size, setSize] = useState({ w: 700, h: 460 });
  const wrapRef = useRef(null);
  const P = { t: 36, r: 20, b: 60, l: 80 };

  useEffect(() => {
    if (!wrapRef.current) return;
    const ro = new ResizeObserver(([e]) =>
      setSize({ w: e.contentRect.width, h: e.contentRect.height })
    );
    ro.observe(wrapRef.current);
    return () => ro.disconnect();
  }, []);

  const { w, h } = size;
  const cw = w - P.l - P.r, ch = h - P.t - P.b;

  const xs = data.map(d => d[xKey]), ys = data.map(d => d[yKey]);
  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  const yMin = Math.min(...ys), yMax = Math.max(...ys);
  const xSpan = (xMax - xMin) || 1, ySpan = (yMax - yMin) || 1;
  const xPad = xSpan * 0.14, yPad = ySpan * 0.14;
  const x0 = xMin - xPad, x1 = xMax + xPad, y0 = yMin - yPad, y1 = yMax + yPad;

  const px = v => P.l + ((v - x0) / (x1 - x0)) * cw;
  const py = v => P.t + (1 - (v - y0) / (y1 - y0)) * ch;

  const getColor = d => colorBy === "currency" ? CCY_COLORS[d.ccy] : STRAT_COLORS[d.strategy];
  const getId    = d => `${d.ccy}||${d.strategy}`;

  const xMeta = METRICS.find(m => m.key === xKey);
  const yMeta = METRICS.find(m => m.key === yKey);

  // Grid ticks
  const nT = 5;
  const xTicks = Array.from({length:nT}, (_,i) => x0 + (i+0.5)*(x1-x0)/(nT-0.5));
  const yTicks = Array.from({length:nT}, (_,i) => y0 + (i+0.5)*(y1-y0)/(nT-0.5));

  // Group by strategy for connector lines
  const byStrat = {};
  data.forEach(d => {
    const s = d.strategy;
    if (!byStrat[s]) byStrat[s] = [];
    byStrat[s].push(d);
  });

  const hov = hoveredId ? data.find(d => getId(d) === hoveredId) : null;

  return (
    <div ref={wrapRef} style={{ width:"100%", height:"100%", position:"relative" }}>
      <svg width={w} height={h} style={{ display:"block" }}>
        {/* BG */}
        <rect width={w} height={h} fill="#0a0f1e" rx="10"/>
        <rect x={P.l} y={P.t} width={cw} height={ch} fill="#111827" rx="6"/>

        {/* Grid */}
        {xTicks.map((v,i) => (
          <g key={`xg${i}`}>
            <line x1={px(v)} y1={P.t} x2={px(v)} y2={P.t+ch} stroke="#1e293b" strokeWidth="1"/>
            <text x={px(v)} y={P.t+ch+16} textAnchor="middle" fill="#475569" fontSize="9.5" fontFamily="monospace">
              {fmt(v,xKey)}
            </text>
          </g>
        ))}
        {yTicks.map((v,i) => (
          <g key={`yg${i}`}>
            <line x1={P.l} y1={py(v)} x2={P.l+cw} y2={py(v)} stroke="#1e293b" strokeWidth="1"/>
            <text x={P.l-8} y={py(v)+4} textAnchor="end" fill="#475569" fontSize="9.5" fontFamily="monospace">
              {fmt(v,yKey)}
            </text>
          </g>
        ))}

        {/* Axes */}
        <line x1={P.l} y1={P.t} x2={P.l} y2={P.t+ch} stroke="#334155" strokeWidth="1.5"/>
        <line x1={P.l} y1={P.t+ch} x2={P.l+cw} y2={P.t+ch} stroke="#334155" strokeWidth="1.5"/>

        {/* Axis labels */}
        <text x={P.l+cw/2} y={h-6} textAnchor="middle" fill="#94a3b8" fontSize="11" fontWeight="600">
          {xMeta?.label}{xMeta?.unit ? ` (${xMeta.unit})` : ""} {xMeta?.hi ? "↑ better" : "↓ better"}
        </text>
        <text x={13} y={P.t+ch/2} textAnchor="middle" fill="#94a3b8" fontSize="11" fontWeight="600"
          transform={`rotate(-90,13,${P.t+ch/2})`}>
          {yMeta?.label}{yMeta?.unit ? ` (${yMeta.unit})` : ""} {yMeta?.hi ? "↑ better" : "↓ better"}
        </text>

        {/* Strategy connectors */}
        {Object.entries(byStrat).map(([strat, pts]) => {
          if (pts.length < 2) return null;
          const sorted = [...pts].sort((a,b)=>a[xKey]-b[xKey]);
          const d = sorted.map((p,i)=>`${i?"L":"M"}${px(p[xKey])},${py(p[yKey])}`).join(" ");
          return <path key={`c${strat}`} d={d} fill="none"
            stroke={STRAT_COLORS[strat]} strokeWidth="1"
            strokeOpacity="0.2"
            strokeDasharray={IS_LAYERED(strat) ? "4,3" : ""}/>
        })}

        {/* Dots — layered behind non-layered */}
        {[...data].sort((a,b) => IS_LAYERED(a.strategy) - IS_LAYERED(b.strategy)).map(d => {
          const id = getId(d), cx = px(d[xKey]), cy = py(d[yKey]);
          const col = getColor(d), isHov = id === hoveredId;
          const r = isHov ? 10 : 7;
          const layered = IS_LAYERED(d.strategy);
          return (
            <g key={id} onMouseEnter={() => setHoveredId(id)} style={{cursor:"pointer"}}>
              {isHov && <circle cx={cx} cy={cy} r={18} fill={col} fillOpacity="0.12"/>}
              {layered
                ? <rect x={cx-r} y={cy-r} width={r*2} height={r*2} rx="3"
                    fill={col} fillOpacity={isHov?1:0.8}
                    stroke={isHov?"#fff":col} strokeWidth={isHov?2:1.5}
                    strokeDasharray="3,2"/>
                : <circle cx={cx} cy={cy} r={r}
                    fill={col} fillOpacity={isHov?1:0.85}
                    stroke={isHov?"#fff":col} strokeWidth={isHov?2:1}/>
              }
              <text x={cx} y={cy+3.5} textAnchor="middle" fill="#0a0f1e"
                fontSize={isHov?"8":"6.5"} fontWeight="800" fontFamily="monospace" pointerEvents="none">
                {STRAT_ABBR[d.strategy]}
              </text>
            </g>
          );
        })}

        {/* Tooltip */}
        {hov && (() => {
          const cx = px(hov[xKey]), cy = py(hov[yKey]);
          const tw = 200, th = 120;
          const tx = cx+(cx>w*0.66?-(tw+14):14), ty = cy+(cy>h*0.6?-(th+10):10);
          return (
            <g pointerEvents="none">
              <rect x={tx} y={ty} width={tw} height={th} rx="8"
                fill="#1e293b" stroke={getColor(hov)} strokeWidth="1.5" opacity="0.98"/>
              <text x={tx+12} y={ty+20} fill="#f1f5f9" fontSize="12" fontWeight="700">
                {hov.ccy} — {hov.strategy}
              </text>
              {[
                [`${xMeta?.label}`, fmt(hov[xKey],xKey)],
                [`${yMeta?.label}`, fmt(hov[yKey],yKey)],
                ["Mean:", fmt(hov.mean,"mean")],
                ["Sharpe:", hov.sharpe.toFixed(4)],
                [`Vol ${hov.avg_sigma}%`, `Carry ${hov.avg_carry>0?"+":""}${hov.avg_carry}%`],
              ].map(([l,r],i) => (
                <g key={i}>
                  <text x={tx+12} y={ty+37+i*15} fill="#94a3b8" fontSize="9.5" fontFamily="monospace">{l}</text>
                  <text x={tx+tw-10} y={ty+37+i*15} fill="#e2e8f0" fontSize="9.5"
                    textAnchor="end" fontFamily="monospace" fontWeight="600">{r}</text>
                </g>
              ))}
            </g>
          );
        })()}
      </svg>
    </div>
  );
}

// ── Panel: metric selector ─────────────────────────────────────────────────
function AxisPanel({ axis, current, onSelect }) {
  const [open, setOpen] = useState(false);
  const meta = METRICS.find(m => m.key === current);
  return (
    <div style={{ background:"#0d1424", borderRadius:10, border:"1px solid #1e293b", overflow:"hidden" }}>
      <div onClick={() => setOpen(o=>!o)} style={{
        padding:"9px 13px", cursor:"pointer", display:"flex",
        justifyContent:"space-between", alignItems:"center",
        background: open?"#161f30":"transparent",
        borderBottom: open?"1px solid #1e293b":"none",
      }}>
        <span style={{ fontSize:10, color:"#475569", letterSpacing:1, fontFamily:"monospace" }}>
          {axis.toUpperCase()}-AXIS
        </span>
        <span style={{ fontSize:11, color:"#93c5fd", fontWeight:600 }}>
          {meta?.label} ▾
        </span>
      </div>
      {open ? (
        <div style={{ padding:"6px 5px", display:"flex", flexDirection:"column", gap:2 }}>
          {METRICS.map(m => (
            <div key={m.key} onClick={() => { onSelect(m.key); setOpen(false); }} style={{
              padding:"6px 10px", borderRadius:6, cursor:"pointer",
              background: current===m.key?"rgba(59,130,246,0.14)":"transparent",
              color: current===m.key?"#93c5fd":"#94a3b8",
              fontSize:11, fontFamily:"monospace",
              display:"flex", justifyContent:"space-between",
            }}>
              <span>{m.label}</span>
              <span style={{ fontSize:9, color:m.hi?"#34d399":"#f87171" }}>{m.hi?"↑ better":"↓ better"}</span>
            </div>
          ))}
        </div>
      ) : (
        <div style={{ padding:"5px 13px 8px", fontSize:9.5, color:"#475569", fontFamily:"monospace",
          display:"flex", justifyContent:"space-between" }}>
          <span>{meta?.unit || "dimensionless"}</span>
          <span style={{ color:meta?.hi?"#34d399":"#f87171" }}>{meta?.hi?"↑ higher better":"↓ lower better"}</span>
        </div>
      )}
    </div>
  );
}

// ── Chip toggle ─────────────────────────────────────────────────────────────
function Chip({ label, active, color, onToggle, square }) {
  return (
    <button onClick={onToggle} style={{
      padding: square?"5px 8px":"4px 9px",
      borderRadius: square?5:5,
      border: active?`1.5px solid ${color}`:"1px solid #1e293b",
      background: active?`${color}1e`:"#060d1a",
      color: active?color:"#475569",
      fontSize:10.5, fontFamily:"monospace", fontWeight:600,
      cursor:"pointer", display:"flex", alignItems:"center", gap:5,
    }}>
      {square && <span style={{ width:8, height:8, background:active?color:"#1e293b",
        display:"inline-block", borderRadius:1.5, flexShrink:0 }}/>}
      {label}
    </button>
  );
}

// ── Main app ────────────────────────────────────────────────────────────────
export default function App() {
  const [xKey, setXKey] = useState("std");
  const [yKey, setYKey] = useState("mean");
  const [colorBy, setColorBy] = useState("strategy");
  const [selCcy,   setSelCcy]   = useState(new Set(ALL_CCYS));
  const [selStrat, setSelStrat] = useState(new Set(ALL_STRATS));
  const [hoveredId, setHoveredId] = useState(null);

  const toggleCcy   = c => setSelCcy(s   => { const n=new Set(s); n.has(c)?n.delete(c):n.add(c); return n; });
  const toggleStrat = s => setSelStrat(p => { const n=new Set(p); n.has(s)?n.delete(s):n.add(s); return n; });

  const data = RAW_DATA.filter(d => selCcy.has(d.ccy) && selStrat.has(d.strategy));

  return (
    <div style={{ background:"#060d1a", minHeight:"100vh", padding:"18px 22px",
      fontFamily:"'Syne', sans-serif", color:"#e2e8f0" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
        * { box-sizing:border-box; }
        button { transition:all 0.13s; }
        button:hover { opacity:0.85; }
      `}</style>

      {/* Header */}
      <div style={{ marginBottom:16, display:"flex", justifyContent:"space-between", alignItems:"flex-end" }}>
        <div>
          <div style={{ fontSize:19, fontWeight:800, letterSpacing:"-0.5px", color:"#f8fafc" }}>
            FX Strategy Explorer
          </div>
          <div style={{ fontSize:11, color:"#475569", marginTop:2, fontFamily:"JetBrains Mono" }}>
            DKK · GBP SEK USD NOK AUD CAD · 3M + layered entry · 2020–2025 demo
          </div>
        </div>
        <div style={{ fontSize:10, color:"#334155", fontFamily:"JetBrains Mono", textAlign:"right" }}>
          ○ = single entry &nbsp;·&nbsp; □ = layered (1/3 each @ 1M 2M 3M)
        </div>
      </div>

      <div style={{ display:"grid", gridTemplateColumns:"210px 1fr", gap:14 }}>

        {/* ── Left controls ── */}
        <div style={{ display:"flex", flexDirection:"column", gap:11 }}>
          <AxisPanel axis="x" current={xKey} onSelect={setXKey}/>
          <AxisPanel axis="y" current={yKey} onSelect={setYKey}/>

          {/* Color by */}
          <div style={{ background:"#0d1424", borderRadius:10, border:"1px solid #1e293b", padding:"10px 12px" }}>
            <div style={{ fontSize:10, color:"#475569", letterSpacing:1, marginBottom:8, fontFamily:"monospace" }}>COLOR BY</div>
            <div style={{ display:"flex", gap:6 }}>
              {["currency","strategy"].map(opt => (
                <button key={opt} onClick={() => setColorBy(opt)} style={{
                  flex:1, padding:"5px 0", borderRadius:6, cursor:"pointer",
                  border: colorBy===opt?"1.5px solid #6366f1":"1px solid #1e293b",
                  background: colorBy===opt?"rgba(99,102,241,0.15)":"#060d1a",
                  color: colorBy===opt?"#a5b4fc":"#475569",
                  fontSize:10, fontFamily:"monospace", textTransform:"capitalize",
                }}>
                  {opt}
                </button>
              ))}
            </div>
          </div>

          {/* Currencies */}
          <div style={{ background:"#0d1424", borderRadius:10, border:"1px solid #1e293b", padding:"10px 12px" }}>
            <div style={{ fontSize:10, color:"#475569", letterSpacing:1, marginBottom:8, fontFamily:"monospace" }}>CURRENCIES</div>
            <div style={{ display:"flex", flexWrap:"wrap", gap:5 }}>
              {ALL_CCYS.map(c => (
                <Chip key={c} label={c} active={selCcy.has(c)} color={CCY_COLORS[c]} onToggle={() => toggleCcy(c)}/>
              ))}
            </div>
          </div>

          {/* Strategies — grouped */}
          <div style={{ background:"#0d1424", borderRadius:10, border:"1px solid #1e293b", padding:"10px 12px" }}>
            <div style={{ fontSize:10, color:"#475569", letterSpacing:1, marginBottom:8, fontFamily:"monospace" }}>
              STRATEGIES
            </div>
            <div style={{ fontSize:9.5, color:"#334155", marginBottom:6, fontFamily:"monospace" }}>Single entry (3M)</div>
            <div style={{ display:"flex", flexDirection:"column", gap:4, marginBottom:10 }}>
              {["Unhedged","Forward","Risk Reversal","Blend 50/50"].map(s => (
                <Chip key={s} label={`${STRAT_ABBR[s]}  ${s}`} active={selStrat.has(s)}
                  color={STRAT_COLORS[s]} onToggle={() => toggleStrat(s)}/>
              ))}
            </div>
            <div style={{ fontSize:9.5, color:"#334155", marginBottom:6, fontFamily:"monospace" }}>Layered entry □</div>
            <div style={{ display:"flex", flexDirection:"column", gap:4 }}>
              {["Fwd Layered","RR Layered","Blend Layered"].map(s => (
                <Chip key={s} label={`${STRAT_ABBR[s]}  ${s}`} active={selStrat.has(s)}
                  color={STRAT_COLORS[s]} onToggle={() => toggleStrat(s)} square/>
              ))}
            </div>
          </div>

          {/* Point count */}
          <div style={{ fontSize:10, color:"#334155", fontFamily:"monospace", textAlign:"center", padding:"4px 0" }}>
            {data.length} data points visible
          </div>
        </div>

        {/* ── Chart ── */}
        <div style={{ background:"#0d1424", borderRadius:12, border:"1px solid #1e293b",
          height:540, position:"relative", overflow:"hidden" }}
          onMouseLeave={() => setHoveredId(null)}>
          <ScatterSVG data={data} xKey={xKey} yKey={yKey}
            colorBy={colorBy} hoveredId={hoveredId} setHoveredId={setHoveredId}/>
        </div>
      </div>

      {/* Footer hint */}
      <div style={{ marginTop:12, display:"flex", gap:20, flexWrap:"wrap",
        fontSize:10, color:"#334155", fontFamily:"monospace", padding:"8px 12px",
        background:"#0d1424", borderRadius:8, border:"1px solid #1e293b" }}>
        <span>Classic view: X=Std Dev ↓ · Y=Mean ↑ (risk/return frontier)</span>
        <span style={{color:"#1e293b"}}>|</span>
        <span>Try: X=VaR-10% · Y=Sharpe to see downside-adjusted performance</span>
        <span style={{color:"#1e293b"}}>|</span>
        <span>Hover any dot for full metric breakdown</span>
        <span style={{color:"#1e293b"}}>|</span>
        <span style={{color:"#6ee7b7"}}>□ = layered entry smooths timing risk</span>
      </div>
    </div>
  );
}
