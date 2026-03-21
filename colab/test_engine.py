"""
test_engine.py — ตรวจสอบว่า Python engine ถูกต้องไหม
=====================================================

รัน: python colab/test_engine.py
"""

import sys
sys.path.insert(0, 'colab')

from makhos_engine import *

PASS = 0
FAIL = 0

def check(name, got, expected):
    global PASS, FAIL
    if got == expected:
        print(f'  ✅ {name}: {got}')
        PASS += 1
    else:
        print(f'  ❌ {name}: got {got}, expected {expected}')
        FAIL += 1

def check_true(name, cond):
    check(name, cond, True)

print('── Test 1: Initial position ──────────────────────────────────────────')
pos = initial_position()
check('P1 men count', bit_count(pos.p1_men), 8)
check('P2 men count', bit_count(pos.p2_men), 8)
check('P1 kings', pos.p1_kings, 0)
check('P2 kings', pos.p2_kings, 0)
check('side', pos.side, 1)
check('halfmove_clock', pos.halfmove_clock, 0)

# P1 ต้องอยู่ที่ 24-31
for i in range(24, 32):
    check_true(f'P1 man at sq {i}', bool(pos.p1_men & b1(i)))
# P2 ต้องอยู่ที่ 0-7
for i in range(0, 8):
    check_true(f'P2 man at sq {i}', bool(pos.p2_men & b1(i)))

print('\n── Test 2: Initial moves (P1 ต้องมี 7 ตา) ────────────────────────────')
moves = generate_moves(pos)
check('P1 initial move count', len(moves), 7)
# ต้องไม่มี captures (เกมเพิ่งเริ่ม)
check_true('no captures at start', all(len(m.captured) == 0 for m in moves))
# P1 เดินขึ้นเท่านั้น (ไม่มีทิศ DL/DR)
from makhos_engine import STEPS
for m in moves:
    r_from, c_from = SQUARE_TO_RC[m.from_sq]
    r_to,   c_to   = SQUARE_TO_RC[m.to_sq]
    check_true(f'P1 moves up: sq{m.from_sq}→{m.to_sq}', r_to < r_from)

print('\n── Test 3: P2 initial moves ──────────────────────────────────────────')
# flip side
pos2 = Position(-1, pos.p1_men, pos.p1_kings, pos.p2_men, pos.p2_kings, 0)
moves2 = generate_moves(pos2)
check('P2 initial move count', len(moves2), 7)
for m in moves2:
    r_from, _ = SQUARE_TO_RC[m.from_sq]
    r_to,   _ = SQUARE_TO_RC[m.to_sq]
    check_true(f'P2 moves down: sq{m.from_sq}→{m.to_sq}', r_to > r_from)

print('\n── Test 4: apply_move ────────────────────────────────────────────────')
m0   = moves[0]
pos3 = apply_move(pos, m0)
check('side flipped after move', pos3.side, -1)
check_true('piece moved from', not bool(pos3.p1_men & b1(m0.from_sq)))
check_true('piece moved to',       bool(pos3.p1_men & b1(m0.to_sq)))
check('halfmove_clock incremented (quiet)', pos3.halfmove_clock, 1)

print('\n── Test 5: Mandatory capture ─────────────────────────────────────────')
# สร้าง position ที่มี capture บังคับ
# P1 man ที่ sq 16, P2 man ที่ sq 9 (adjacent UL), sq 2 empty (landing)
# sq 16 = (4,1), sq 9 = (3,2)→ wait let me recalc
# sq 16=(4,1), sq 9=(2,3)... ไม่ adjacent
# ใช้ STEPS แทน — หา pair ที่ adjacent จริงๆ
# sq 21=(5,2), STEPS[21] มี UL=(4,1)=16, UR=(4,3)=17
# วาง P1 ที่ 21, P2 ที่ 16, empty ที่ 12=(3,0)
cap_pos = Position(
    side=1,
    p1_men=b1(21),
    p1_kings=0,
    p2_men=b1(16),
    p2_kings=0,
    halfmove_clock=0
)
cap_moves = generate_moves(cap_pos)
check('must capture (no quiet moves)', len(cap_moves) >= 1, True)
check_true('all moves are captures', all(len(m.captured) > 0 for m in cap_moves))
cap_result = apply_move(cap_pos, cap_moves[0])
check('captured piece removed', bit_count(cap_result.p2_men), 0)
check('halfmove_clock reset after capture', cap_result.halfmove_clock, 0)

print('\n── Test 6: Promotion ─────────────────────────────────────────────────')
# P1 man ที่ row 1 → เดินไป row 0 = promote
# sq 4=(1,0) → step UR=(0,1)=0 → LAST_RANK_P1
promo_pos = Position(
    side=1,
    p1_men=b1(4),
    p1_kings=0,
    p2_men=0,
    p2_kings=0,
    halfmove_clock=0
)
promo_moves = generate_moves(promo_pos)
promo_to_rank0 = [m for m in promo_moves if m.to_sq in LAST_RANK_P1]
if promo_to_rank0:
    m_promo = promo_to_rank0[0]
    check_true('promote=True when reaching row 0', m_promo.promote)
    result = apply_move(promo_pos, m_promo)
    check_true('piece becomes king', bool(result.p1_kings & b1(m_promo.to_sq)))
    check('no longer a man', bit_count(result.p1_men), 0)
else:
    print('  ⚠️  ไม่มีตาเดินไป rank 0 จาก sq 4 — ตรวจ STEPS')

print('\n── Test 7: King movement ─────────────────────────────────────────────')
# King ที่กลางกระดาน sq 17=(3,2) ไม่มีหมากอื่น → เดินได้หลายช่อง
king_pos = Position(
    side=1,
    p1_men=0,
    p1_kings=b1(17),
    p2_men=0,
    p2_kings=0,
    halfmove_clock=0
)
king_moves = generate_moves(king_pos)
check_true('king has multiple moves', len(king_moves) > 4)

print('\n── Test 8: Draw by inactivity ────────────────────────────────────────')
draw_pos = Position(1, b1(0), 0, b1(31), 0, 20)
check_true('draw when ≤2 pieces each + clock≥20', is_draw_by_inactivity(draw_pos))
no_draw = Position(1, b1(0), 0, b1(31), 0, 19)
check_true('no draw when clock=19', not is_draw_by_inactivity(no_draw))

print('\n── Test 9: Terminal ──────────────────────────────────────────────────')
term_pos = Position(1, 0, 0, b1(15), 0, 0)  # P1 has no pieces
check_true('terminal when side has no pieces', is_terminal(term_pos))

print('\n── Test 10: Features ─────────────────────────────────────────────────')
feat = get_features(initial_position())
check('feature shape', feat.shape, (128,))
check('my men features', int(feat[:32].sum()), 8)    # 8 P1 men
check('enemy men features', int(feat[64:96].sum()), 8)  # 8 P2 men
check('no kings', int(feat[32:64].sum()) + int(feat[96:128].sum()), 0)

print(f'\n{"─"*55}')
print(f'Results: {PASS} passed, {FAIL} failed')
if FAIL == 0:
    print('✅ Engine ถูกต้องครับ! พร้อมใช้งาน')
else:
    print('❌ มีบางอย่างผิดพลาด — ต้องแก้ก่อน train')
