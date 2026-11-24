"""UCI protocol tests against the built engine (Phase 2/3 gate).

Run: python3 -m unittest tests/test_uci.py   (engine path from $BULWARK, default build/bulwark)
Checks position/go/stop/ucinewgame/isready/setoption, searchmoves, time limits, and Chess960 castling spelling
(king-takes-rook with UCI_Chess960, e1g1 otherwise). Every bestmove is checked for legality with python-chess.
"""
from __future__ import annotations

import os
import queue
import subprocess
import threading
import time
import unittest

import chess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE = os.environ.get("BULWARK", os.path.join(ROOT, "build", "bulwark"))


class Engine:
    def __init__(self):
        self.p = subprocess.Popen([ENGINE], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.q: queue.Queue[str] = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.p.stdout:
            self.q.put(line.rstrip("\n"))

    def send(self, cmd: str):
        self.p.stdin.write(cmd + "\n")
        self.p.stdin.flush()

    def expect(self, prefix: str, timeout: float = 10.0) -> tuple[str, list[str]]:
        """Read lines until one starts with prefix; returns (that line, all lines read)."""
        seen = []
        end = time.time() + timeout
        while time.time() < end:
            try:
                line = self.q.get(timeout=max(0.01, end - time.time()))
            except queue.Empty:
                break
            seen.append(line)
            if line.startswith(prefix):
                return line, seen
        raise AssertionError(f"no {prefix!r} within {timeout}s; got {seen[-5:]}")

    def ready(self):
        self.send("isready")
        self.expect("readyok")

    def close(self):
        try:
            self.send("quit")
            self.p.wait(timeout=5)
        except Exception:      # noqa: BLE001
            self.p.kill()


@unittest.skipUnless(os.path.exists(ENGINE), f"engine not built: {ENGINE}")
class TestUci(unittest.TestCase):
    def setUp(self):
        self.e = Engine()
        self.e.ready()          # startup (attack tables) is not part of any timed check

    def tearDown(self):
        self.e.close()

    def bestmove(self, timeout=20.0) -> str:
        line, _ = self.e.expect("bestmove", timeout)
        return line.split()[1]

    def test_handshake_and_options(self):
        self.e.send("uci")
        _, lines = self.e.expect("uciok")
        text = "\n".join(lines)
        self.assertTrue(any(l.startswith("id name") for l in lines))
        for opt in ("Hash", "Threads", "Move Overhead", "UCI_Chess960", "SyzygyPath", "EvalFile", "IIR",
                    "D1_WDLUtility", "DefenseMode", "DrawValue", "D3_Rule50Scale", "D4_Syzygy", "D5_StalemateGuard",
                    "D6_PerpetualSeek", "D7_FortressDetect", "UCI_ShowWDL"):
            self.assertIn(f"option name {opt} type", text, opt)
        self.e.ready()

    def test_position_moves_and_go_depth(self):
        moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"]
        self.e.send("ucinewgame")
        self.e.send("position startpos moves " + " ".join(moves))
        self.e.send("go depth 6")
        mv = self.bestmove()
        b = chess.Board()
        for m in moves:
            b.push_uci(m)
        self.assertIn(chess.Move.from_uci(mv), b.legal_moves)

    def test_position_fen_and_nodes(self):
        fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
        self.e.send(f"position fen {fen} moves f1c4")
        self.e.send("go nodes 5000")
        mv = self.bestmove()
        b = chess.Board(fen)
        b.push_uci("f1c4")
        self.assertIn(chess.Move.from_uci(mv), b.legal_moves)

    def test_stop_infinite(self):
        self.e.send("position startpos")
        self.e.send("go infinite")
        time.sleep(0.3)
        t = time.time()
        self.e.send("stop")
        mv = self.bestmove(timeout=2.0)
        self.assertLess(time.time() - t, 1.0)
        self.assertIn(chess.Move.from_uci(mv), chess.Board().legal_moves)

    def test_clock_respected(self):
        self.e.send("position startpos")
        t = time.time()
        self.e.send("go wtime 1000 btime 1000 winc 0 binc 0")
        self.bestmove(timeout=3.0)
        self.assertLess(time.time() - t, 0.9)

    def test_movetime(self):
        self.e.send("position startpos")
        t = time.time()
        self.e.send("go movetime 200")
        self.bestmove(timeout=3.0)
        self.assertLess(time.time() - t, 0.6)

    def test_isready_during_search_and_repeated_go(self):
        self.e.send("position startpos")
        for _ in range(3):
            self.e.send("go depth 4")
            self.bestmove()
        self.e.send("go infinite")
        self.e.send("isready")
        self.e.expect("readyok", 2.0)
        self.e.send("stop")
        self.bestmove(2.0)

    def test_searchmoves(self):
        self.e.send("position startpos")
        self.e.send("go depth 5 searchmoves a2a3 h2h3")
        self.assertIn(self.bestmove(), ("a2a3", "h2h3"))

    def test_castling_spelling_standard(self):
        fen = "4rkr1/4p1p1/8/8/8/8/8/4K2R w K - 0 1"
        self.e.send(f"position fen {fen}")
        self.e.send("go depth 3 searchmoves e1g1")
        self.assertEqual(self.bestmove(), "e1g1")
        # Input in either spelling is accepted outside 960 mode.
        self.e.send(f"position fen {fen} moves e1h1")
        self.e.send("d")
        line, _ = self.e.expect("4rkr1")
        self.assertTrue(line.startswith("4rkr1/4p1p1/8/8/8/8/8/5RK1 b"), line)

    def test_castling_spelling_960(self):
        self.e.send("setoption name UCI_Chess960 value true")
        fen = "1r1k1r2/8/8/8/8/8/8/1R1K1R2 w FBfb - 0 1"
        b = chess.Board(fen, chess960=True)
        castles = [m for m in b.legal_moves if b.is_castling(m)]
        self.assertTrue(castles)
        uci = castles[0].uci()                      # python-chess 960 mode: king-takes-rook
        self.e.send(f"position fen {fen}")
        self.e.send(f"go depth 3 searchmoves {uci}")
        self.assertEqual(self.bestmove(), uci)
        self.e.send(f"position fen {fen} moves {uci}")
        self.e.send("go depth 4")
        b.push(castles[0])
        mv = self.bestmove()
        self.assertIn(chess.Move.from_uci(mv), b.legal_moves)

    def test_ucinewgame_and_options(self):
        self.e.send("setoption name Hash value 32")
        self.e.send("setoption name IIR value false")
        self.e.send("setoption name DrawValue value 600")
        self.e.send("setoption name D1_WDLUtility value true")
        self.e.send("ucinewgame")
        self.e.ready()
        self.e.send("position startpos moves d2d4")
        self.e.send("go depth 5")
        b = chess.Board()
        b.push_uci("d2d4")
        self.assertIn(chess.Move.from_uci(self.bestmove()), b.legal_moves)

    def test_bad_evalfile_keeps_engine_alive(self):
        self.e.send("setoption name EvalFile value /nonexistent.nnue")
        self.e.expect("info string EvalFile")
        self.e.send("position startpos")
        self.e.send("go depth 3")
        self.assertIn(chess.Move.from_uci(self.bestmove()), chess.Board().legal_moves)

    def test_terminal_positions(self):
        # Mated and stalemated sides still get a (null) bestmove line without crashing.
        for fen in ("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1", "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"):
            self.e.send(f"position fen {fen}")
            self.e.send("go depth 3")
            self.e.expect("bestmove")
        self.e.ready()


if __name__ == "__main__":
    unittest.main()
