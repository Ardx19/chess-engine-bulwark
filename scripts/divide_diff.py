#!/usr/bin/env python3
"""Bisect a perft mismatch: divide_diff.py '<shredder fen>' <depth> -> descends into the first differing move."""
import subprocess, sys, chess
fen, depth = sys.argv[1], int(sys.argv[2])
def mine(fen, d):
    out = subprocess.run(["./build/perft", "divide", str(d), fen], env={"CHESS960": "1"}, capture_output=True, text=True).stdout
    return {l.split(": ")[0]: int(l.split(": ")[1]) for l in out.splitlines() if ": " in l and not l.startswith("Nodes")}
def ref(b, d):
    r = {}
    for m in b.legal_moves:
        b.push(m)
        n = 1 if d == 1 else sum(1 for _ in [0]) and perft(b, d-1)
        b.pop(); r[m.uci()] = n
    return r
def perft(b, d):
    if d == 0: return 1
    if d == 1: return b.legal_moves.count()
    n = 0
    for m in b.legal_moves:
        b.push(m); n += perft(b, d-1); b.pop()
    return n
b = chess.Board(fen, chess960=True)
while depth >= 1:
    a = mine(b.fen(shredder=True), depth); r = ref(b, depth)
    # python-chess prints castling as king-to-destination in 960 ucis? normalise to king-takes-rook
    r2 = {}
    for m in b.legal_moves:
        u = m.uci()
        if b.is_castling(m):
            u = chess.square_name(m.from_square) + chess.square_name(b.castling_rights and m.to_square)
        r2[u] = r.get(m.uci())
    diff = [k for k in set(a) | set(r2) if a.get(k) != r2.get(k)]
    if not diff: print("no diff at depth", depth, b.fen()); break
    k = sorted(diff)[0]
    print(f"depth {depth} {b.fen(shredder=True)}: move {k} mine={a.get(k)} ref={r2.get(k)} (all diffs: {sorted(diff)})")
    mv = next((m for m in b.legal_moves if m.uci() == k or (b.is_castling(m) and chess.square_name(m.from_square)+chess.square_name(m.to_square) == k)), None)
    if mv is None: break
    b.push(mv); depth -= 1
