#!/usr/bin/env python3
"""Generate Chess960 perft reference lines with python-chess (independent implementation).
Usage: gen_perft960.py <count> <depth> [seed] >> tests/data/perft_960.epd
Also useful: DFRC-style positions (independent random back ranks per side)."""
import random, sys, chess

def back_rank(rng):
    # random legal 960 back rank: bishops on opposite colors, king between rooks
    sq = list(range(8))
    r = [None]*8
    r[rng.choice([0,2,4,6])] = 'b'
    r[rng.choice([1,3,5,7])] = 'b'
    free = [i for i in range(8) if r[i] is None]
    r[free.pop(rng.randrange(len(free)))] = 'q'
    r[free.pop(rng.randrange(len(free)))] = 'n'
    r[free.pop(rng.randrange(len(free)))] = 'n'
    r[free[0]] = 'r'; r[free[1]] = 'k'; r[free[2]] = 'r'
    return ''.join(r)

def perft(b, d):
    if d == 0: return 1
    if d == 1: return b.legal_moves.count()
    n = 0
    for m in b.legal_moves:
        b.push(m); n += perft(b, d-1); b.pop()
    return n

count, depth = int(sys.argv[1]), int(sys.argv[2])
rng = random.Random(int(sys.argv[3]) if len(sys.argv) > 3 else 1)
for i in range(count):
    w, bl = back_rank(rng), back_rank(rng)   # independent ranks => DFRC
    fen = f"{bl}/pppppppp/8/8/8/8/PPPPPPPP/{w.upper()} w KQkq - 0 1"
    board = chess.Board(fen, chess960=True)
    # play a few random plies so castling, pins and ep show up in the tests
    for _ in range(rng.randrange(0, 12)):
        ms = list(board.legal_moves)
        if not ms: break
        board.push(rng.choice(ms))
    if board.is_game_over(): continue
    fen = board.fen(shredder=True)   # castling letters as files
    parts = ";".join(f"D{d} {perft(board, d)}" for d in range(1, depth + 1))
    print(f"{fen} ;{parts.replace(';', ' ;')}")
