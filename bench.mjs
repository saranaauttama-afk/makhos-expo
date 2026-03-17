// bench.mjs — Quick benchmark of the search engine at various game phases
// Run with:  node bench.mjs

// ── minimal bitboard helpers (mirror of bitboards.ts) ────────────────────────
const SQUARE_TO_RC = new Array(32);
const RC_TO_INDEX  = new Int16Array(64).fill(-1);
let idx = 0;
for (let r = 0; r < 8; r++)
  for (let c = 0; c < 8; c++)
    if (((r + c) & 1) === 1) {
      SQUARE_TO_RC[idx] = { r, c };
      RC_TO_INDEX[r * 8 + c] = idx++;
    }

const toRC    = i => SQUARE_TO_RC[i];
const toIndex = (r, c) => RC_TO_INDEX[r * 8 + c];
const B1      = i => (1 << i) >>> 0;
function* bits(bb) {
  let x = bb >>> 0;
  while (x) { const lsb = x & -x; yield 31 - Math.clz32(lsb); x = (x ^ lsb) >>> 0; }
}
function bitCount(x) { let c = 0; x = x >>> 0; while (x) { x &= x-1; c++; } return c; }

const STEPS = Array.from({ length: 32 }, () => []);
const dirs  = [['UL',-1,-1],['UR',-1,+1],['DL',+1,-1],['DR',+1,+1]];
for (let i = 0; i < 32; i++) {
  const { r, c } = toRC(i);
  for (const [dir, dr, dc] of dirs) {
    const r1 = r+dr, c1 = c+dc;
    if (r1>=0 && r1<8 && c1>=0 && c1<8) {
      const to = toIndex(r1,c1);
      if (to >= 0) STEPS[i].push({ to, dir });
    }
  }
}

// ── position helpers ─────────────────────────────────────────────────────────
const occ = p => (p.p1Men|p.p1Kings|p.p2Men|p.p2Kings) >>> 0;

function nextInDir(from, dir) {
  const s = STEPS[from].find(s => s.dir === dir);
  return s ? s.to : -1;
}
function* ray(from, dir) {
  let cur = from;
  while (true) { const n = nextInDir(cur, dir); if (n < 0) return; yield n; cur = n; }
}

// ── movegen (simplified, faithful to movegen.ts) ─────────────────────────────
const LAST_P1 = new Set([0,1,2,3]);
const LAST_P2 = new Set([28,29,30,31]);
const DIRS4   = ['UL','UR','DL','DR'];

function applyMove(p, m) {
  const q = { ...p };
  const myMen = p.side===1?'p1Men':'p2Men', myKings=p.side===1?'p1Kings':'p2Kings';
  const opMen = p.side===1?'p2Men':'p1Men', opKings=p.side===1?'p2Kings':'p1Kings';
  const fromBit=B1(m.from), toBit=B1(m.to);
  const isKing=(q[myKings]&fromBit)!==0;
  q[isKing?myKings:myMen] = ((q[isKing?myKings:myMen]&~fromBit)|toBit)>>>0;
  for (const c of m.captured) {
    const cb=B1(c);
    if (q[opMen]&cb) q[opMen]=(q[opMen]&~cb)>>>0; else q[opKings]=(q[opKings]&~cb)>>>0;
  }
  if (m.promote && !isKing) { q[myMen]=(q[myMen]&~toBit)>>>0; q[myKings]=(q[myKings]|toBit)>>>0; }
  q.side = p.side===1?-1:1;
  q.halfmoveClock = m.captured.length>0 ? 0 : p.halfmoveClock+1;
  return q;
}

function generateMoves(p) {
  const occBB = occ(p);
  const empty = (~occBB)>>>0;
  const myMen = p.side===1?p.p1Men:p.p2Men;
  const myKings = p.side===1?p.p1Kings:p.p2Kings;
  const caps = [];
  for (const from of bits(myMen)) _menCaps(p, from, caps);
  for (const from of bits(myKings)) _kingCaps(p, from, caps);
  if (caps.length) return caps;
  const q = [];
  for (const from of bits(myMen))
    for (const st of STEPS[from]) {
      if (p.side===1&&(st.dir==='DL'||st.dir==='DR')) continue;
      if (p.side===-1&&(st.dir==='UL'||st.dir==='UR')) continue;
      if (empty&B1(st.to)) q.push({from,to:st.to,captured:[],promote:p.side===1?LAST_P1.has(st.to):LAST_P2.has(st.to)});
    }
  for (const from of bits(myKings))
    for (const dir of DIRS4)
      for (const sq of ray(from, dir)) {
        if (occBB&B1(sq)) break;
        q.push({from,to:sq,captured:[],promote:false});
      }
  return q;
}

function _menCaps(p, from, out) {
  const myMen=p.side===1?p.p1Men:p.p2Men, myKings=p.side===1?p.p1Kings:p.p2Kings;
  const opMen=p.side===1?p.p2Men:p.p1Men, opKings=p.side===1?p.p2Kings:p.p1Kings;
  const path=[], caps=[];
  function dfs(cur, mm, mk, om, ok) {
    let ext=false;
    for (const st of STEPS[cur]) {
      if (p.side===1&&(st.dir==='DL'||st.dir==='DR')) continue;
      if (p.side===-1&&(st.dir==='UL'||st.dir==='UR')) continue;
      const over=st.to; if (over<0) continue;
      const ob=B1(over), occN=(mm|mk|om|ok)>>>0;
      if (!((om|ok)&ob)) continue;
      const landing=nextInDir(over,st.dir); if (landing<0) continue;
      const lb=B1(landing);
      if (!((~occN)>>>0&lb)) continue;
      const fb=B1(cur), ck=(ok&ob)!==0;
      let mmN=mm,mkN=mk,omN=om,okN=ok;
      if (mkN&fb) mkN=((mkN&~fb)|lb)>>>0; else mmN=((mmN&~fb)|lb)>>>0;
      if (ck) okN=(okN&~ob)>>>0; else omN=(omN&~ob)>>>0;
      path.push(landing); caps.push(over);
      dfs(landing,mmN,mkN,omN,okN);
      path.pop(); caps.pop(); ext=true;
    }
    if (!ext&&caps.length>0) {
      const lt=path.length?path[path.length-1]:cur;
      out.push({from,to:lt,captured:[...caps],promote:p.side===1?LAST_P1.has(lt):LAST_P2.has(lt)});
    }
  }
  dfs(from,myMen,myKings,opMen,opKings);
}

function _kingCaps(p, from, out) {
  const myMen=p.side===1?p.p1Men:p.p2Men, myKings=p.side===1?p.p1Kings:p.p2Kings;
  const opMen=p.side===1?p.p2Men:p.p1Men, opKings=p.side===1?p.p2Kings:p.p1Kings;
  if (!(myKings&B1(from))) return;
  const path=[], caps=[];
  function dfs(cur, mm, mk, om, ok) {
    let ext=false;
    for (const dir of DIRS4) {
      let seenEnemy=false, enemyIdx=-1;
      for (const sq of ray(cur,dir)) {
        const bit=B1(sq), occN=(mm|mk|om|ok)>>>0;
        if ((mm|mk)&bit) break;
        if (!seenEnemy) {
          if ((om|ok)&bit) { seenEnemy=true; enemyIdx=sq; continue; }
          continue;
        } else {
          if ((~occN)>>>0&bit) {
            const landing=sq, fb=B1(cur), lb=B1(landing), eb=B1(enemyIdx), ck=(ok&eb)!==0;
            let mmN=mm,mkN=mk,omN=om,okN=ok;
            mkN=((mkN&~fb)|lb)>>>0;
            if (ck) okN=(okN&~eb)>>>0; else omN=(omN&~eb)>>>0;
            path.push(landing); caps.push(enemyIdx);
            dfs(landing,mmN,mkN,omN,okN);
            path.pop(); caps.pop(); ext=true; break;
          } else break;
        }
      }
    }
    if (!ext&&caps.length>0) {
      const lt=path.length?path[path.length-1]:cur;
      out.push({from,to:lt,captured:[...caps],promote:false});
    }
  }
  dfs(from,myMen,myKings,opMen,opKings);
}

// ── simple eval ──────────────────────────────────────────────────────────────
function evaluate(p) {
  const s=p.side;
  return 100*(bitCount(s===1?p.p1Men:p.p2Men)-bitCount(s===1?p.p2Men:p.p1Men))
       + 280*(bitCount(s===1?p.p1Kings:p.p2Kings)-bitCount(s===1?p.p2Kings:p.p1Kings));
}

// ── simple alpha-beta (no TT, no pruning) just to count nodes ─────────────────
let nodes = 0;
function ab(p, depth, alpha, beta, deadline) {
  nodes++;
  if (Date.now() > deadline) return evaluate(p);
  if (depth <= 0) return evaluate(p);
  const moves = generateMoves(p);
  if (!moves.length) return -1_000_000;
  let best = -Infinity;
  for (const m of moves) {
    const score = -ab(applyMove(p, m), depth-1, -beta, -alpha, deadline);
    if (score > best) best = score;
    if (score > alpha) alpha = score;
    if (alpha >= beta) break;
  }
  return best;
}

// ── test positions ────────────────────────────────────────────────────────────
const INITIAL = {
  side: 1,
  p1Men:   [24,25,26,27,28,29,30,31].reduce((b,i)=>b|(1<<i)>>>0,0),
  p1Kings: 0,
  p2Men:   [0,1,2,3,4,5,6,7].reduce((b,i)=>b|(1<<i)>>>0,0),
  p2Kings: 0,
  halfmoveClock: 0,
};

// Simulate ~10 random moves from initial to get a mid-game position
function randomMidgame(seed = 42) {
  let pos = { ...INITIAL };
  let rng = seed;
  function rand() { rng = (rng * 1664525 + 1013904223) >>> 0; return rng; }
  for (let i = 0; i < 14; i++) {
    const moves = generateMoves(pos);
    if (!moves.length) break;
    pos = applyMove(pos, moves[rand() % moves.length]);
  }
  return pos;
}

// ── run benchmark ────────────────────────────────────────────────────────────
console.log('=== Makhos Engine Benchmark (node.js, no TT) ===\n');

const positions = [
  { label: 'Start (8v8)',   pos: INITIAL },
  { label: 'Mid-game ~7v7', pos: randomMidgame(42) },
  { label: 'Mid-game ~6v6', pos: randomMidgame(99) },
  { label: 'Mid-game ~5v6', pos: randomMidgame(777) },
];

for (const { label, pos } of positions) {
  const p1 = bitCount(pos.p1Men|pos.p1Kings);
  const p2 = bitCount(pos.p2Men|pos.p2Kings);
  console.log(`--- ${label} (actual: ${p1}v${p2}) ---`);

  for (const depth of [4, 6, 8]) {
    nodes = 0;
    const t0 = Date.now();
    const deadline = t0 + 5000; // 5s max per depth
    const score = ab(pos, depth, -Infinity, Infinity, deadline);
    const ms = Date.now() - t0;
    const timeout = ms >= 4900;
    console.log(`  depth ${depth}: ${ms.toString().padStart(5)}ms  ${nodes.toLocaleString().padStart(12)} nodes  score=${score}${timeout?' TIMEOUT':''}`);
  }
  console.log();
}
