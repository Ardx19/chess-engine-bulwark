"""Write random Double Fischer Random start positions (independent Chess960 back ranks per side) as Shredder-FEN EPD.

usage: python3 scripts/gen_dfrc.py <out.epd> [count=1000] [seed=1]
"""
import random
import sys

import chess


def back_rank(n):
    return chess.Board.from_chess960_pos(n).board_fen().split("/")[7]   # white's rank 1, uppercase


def rook_files(rank):
    return "".join(chr(ord("A") + f) for f in sorted((f for f, c in enumerate(rank) if c == "R"), reverse=True))


def main():
    out = sys.argv[1]
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    rng = random.Random(int(sys.argv[3]) if len(sys.argv) > 3 else 1)
    pairs = set()
    while len(pairs) < count:
        pairs.add((rng.randrange(960), rng.randrange(960)))
    with open(out, "w") as f:
        for w, b in sorted(pairs):
            wr, br = back_rank(w), back_rank(b)
            fen = f"{br.lower()}/pppppppp/8/8/8/8/PPPPPPPP/{wr} w {rook_files(wr)}{rook_files(br).lower()} - 0 1"
            board = chess.Board(fen, chess960=True)
            assert board.is_valid() and board.fen(shredder=True) == fen, fen
            f.write(fen + "\n")
    print(f"{count} positions -> {out}")


if __name__ == "__main__":
    main()
