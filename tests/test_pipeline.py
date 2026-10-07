"""Pipeline tests: record formats, splits, label perspective, feature symmetry, loader cursor, resume.
Run: python3 -m unittest discover -s tests -p 'test_pipeline.py'   (LC0_TAR=<tar> adds a real-data check)"""
import glob
import json
import os
import shutil
import sys
import tempfile
import unittest
import urllib.request

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pipeline import records as R                      # noqa: E402
from nnue import targets as T                          # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def color_flip(fen):
    pl, stm, cas, ep, *rest = fen.split()
    rows = pl.split("/")[::-1]
    sw = lambda s: s.swapcase()                        # noqa: E731
    return " ".join(["/".join(sw(r) for r in rows), "b" if stm == "w" else "w",
                     "".join(sorted(sw(cas), key="KQkq".index)) if cas != "-" else "-", ep, *rest])


class Records(unittest.TestCase):
    def test_sizes_and_header(self):
        self.assertEqual(R.MASTER_DTYPE.itemsize, 56)
        self.assertEqual(R.TRAIN_DTYPE.itemsize, 40)
        offs = {n: R.MASTER_DTYPE.fields[n][1] for n in R.MASTER_DTYPE.names}
        self.assertEqual((offs["sf_cp"], offs["sf_w"], offs["lc_q"], offs["result"], offs["ply"], offs["gid"]),
                         (27, 29, 33, 39, 41, 43))        # docs/interfaces.md
        with tempfile.TemporaryDirectory() as d:
            rec = R.from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
            p = os.path.join(d, "a.bin")
            R.write_shard(p, rec, 7)
            back = R.read_shard(p)
            self.assertEqual(back.tobytes(), rec.tobytes())

    def test_fen_roundtrip(self):
        for fen in ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b Kq - 3 1"]:
            self.assertEqual(R.to_fen(R.from_fen(fen)), fen.rsplit(" ", 1)[0] + " 1")

    def test_split_by_game(self):
        gid = (np.arange(20000, dtype=np.uint64) // 40) + (np.uint64(3) << np.uint64(32))   # 40 positions per game
        s = R.split_of(gid)
        for g in np.unique(gid):
            self.assertEqual(len(np.unique(s[gid == g])), 1)               # a game never straddles splits
        frac = np.bincount(s, minlength=3) / len(s)
        self.assertTrue(0.9 < frac[0] < 1.0 and 0.0 < frac[1] < 0.1 and 0.0 < frac[2] < 0.1)
        np.testing.assert_array_equal(s, R.split_of(gid))                   # deterministic


class Targets(unittest.TestCase):
    def test_lc0_wdl_perspective(self):
        w = T.lc0_wdl(np.array([0.5, -0.5, 0.0, 1.0]), np.array([0.2, 0.2, 1.0, 0.0]))
        np.testing.assert_allclose(w[0], [0.65, 0.2, 0.15], atol=1e-6)      # good for the side to move
        np.testing.assert_allclose(w[1], [0.15, 0.2, 0.65], atol=1e-6)      # bad for the side to move
        np.testing.assert_allclose(w[2], [0.0, 1.0, 0.0], atol=1e-6)        # dead drawn
        np.testing.assert_allclose(w[3], [1.0, 0.0, 0.0], atol=1e-6)

    def test_blend(self):
        rec = R.from_fen("4k3/8/8/8/8/8/8/4K2R w - - 0 1", lc_q=5000, lc_d=2000, sf_w=700, sf_d=200, result=2)
        t = T.blend_targets(rec)[0]
        want = 0.6 * np.array([0.65, 0.2, 0.15]) + 0.25 * np.array([0.7, 0.2, 0.1]) + 0.15 * np.array([1, 0, 0])
        np.testing.assert_allclose(t, want, atol=1e-5)
        self.assertAlmostEqual(float(t.sum()), 1.0, places=5)

    def test_missing_labels_renormalise(self):
        rec = R.from_fen("4k3/8/8/8/8/8/8/4K2R w - - 0 1", result=1)          # only the game result is present
        np.testing.assert_allclose(T.blend_targets(rec)[0], [0, 1, 0])
        rec = R.from_fen("4k3/8/8/8/8/8/8/4K2R w - - 0 1", lc_q=0, lc_d=10000, result=2)
        np.testing.assert_allclose(T.blend_targets(rec)[0], [0.15 / 0.75, 0.6 / 0.75, 0], atol=1e-5)

    def test_tablebase_override(self):
        rec = R.from_fen("4k3/8/8/8/8/8/8/4K2R w - - 0 1", lc_q=-9000, lc_d=500, sf_w=0, sf_d=0, result=2, flags=1)
        np.testing.assert_allclose(T.blend_targets(rec)[0], [1, 0, 0])


class Features(unittest.TestCase):
    FENS = ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"]

    def test_color_flip_symmetry(self):
        from nnue.data import features, PAD
        for fen in self.FENS:
            a = R.from_fen(fen)
            b = R.from_fen(color_flip(fen))
            ua, ta = features(a)
            ub, tb = features(b)
            np.testing.assert_array_equal(np.sort(ua), np.sort(ub))          # same features from the mover's view
            np.testing.assert_array_equal(np.sort(ta), np.sort(tb))
            self.assertEqual(int((ua != PAD).sum()), bin(int(a["occ"][0])).count("1"))

    def test_start_position_features(self):
        from nnue.data import features
        us, them = features(R.from_fen(self.FENS[0]))
        self.assertIn(0 * 64 + 8, us[0].tolist())         # own pawn on a2 (white to move): rel 0, type pawn, sq 8
        self.assertIn(6 * 64 + 48, us[0].tolist())        # enemy pawn on a7
        self.assertIn(0 * 64 + 8, them[0].tolist())       # symmetric start: black's own a-pawn seen from black (a2 after the flip)
        self.assertIn(6 * 64 + 48, them[0].tolist())


class Policy(unittest.TestCase):
    def test_policy_table_matches_lc0(self):
        exe = os.path.join(ROOT, "build", "lc0_convert")
        if not os.path.exists(exe):
            self.skipTest("lc0_convert not built")
        import subprocess
        mine = subprocess.run([exe, "--dump-policy"], capture_output=True, text=True).stdout.split()
        try:
            src = urllib.request.urlopen("https://raw.githubusercontent.com/LeelaChessZero/lczero-training/master/tf/policy_index.py", timeout=20).read().decode()
        except Exception:
            self.skipTest("offline")
        ns = {}
        exec(src, ns)
        self.assertEqual(mine, ns["policy_index"])


@unittest.skipUnless(os.environ.get("LC0_TAR"), "set LC0_TAR to a real Lc0 tar to check conversion")
class RealData(unittest.TestCase):
    def test_convert_perspective(self):
        from pipeline.convert import convert
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "s.bin")
            st = convert(os.environ["LC0_TAR"], out, os.path.join(ROOT, "build", "lc0_convert"), 1, 0, max_games=150)
            self.assertEqual(st["malformed"], 0)
            self.assertGreater(st["kept"], 5000)
            r = R.read_shard(out)
            rows, cols, codes = R.decode_board(r)
            val = np.array([1, 3, 3.2, 5, 9, 0, -1, -3, -3.2, -5, -9, 0])
            mat = np.zeros(len(r)); np.add.at(mat, rows, val[codes])
            mat_stm = np.where(r["stm_cas"] & 1, -mat, mat)
            self.assertGreater(np.corrcoef(mat_stm, r["lc_q"] / 1e4)[0, 1], 0.25)      # eval agrees with material
            self.assertGreater(np.corrcoef(r["result"].astype(float) - 1, r["lc_q"] / 1e4)[0, 1], 0.4)


class Training(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch  # noqa: F401
        cls.tmp = tempfile.mkdtemp()
        rng = np.random.default_rng(0)
        fens = Features.FENS
        recs = np.concatenate([R.from_fen(f) for f in fens] * 700)
        n = len(recs)
        recs["lc_q"] = rng.integers(-8000, 8000, n); recs["lc_d"] = rng.integers(0, 6000, n)
        recs["result"] = rng.integers(0, 3, n)
        os.makedirs(os.path.join(cls.tmp, "train")); os.makedirs(os.path.join(cls.tmp, "val"))
        for i in range(2):
            R.write_shard(os.path.join(cls.tmp, "train", f"s{i}.bin"), R.to_training(recs[i::2]), i)
        R.write_shard(os.path.join(cls.tmp, "val", "v0.bin"), R.to_training(recs[:600]), 0)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def cfg(self, run_id, **kw):
        from nnue.train import DEFAULTS
        c = dict(DEFAULTS, run_id=run_id, ckpt_dir=os.path.join(self.tmp, "ckpt"), n=16, head=3, batch_size=128, steps=40,
                 warmup=5, val_every=10, ckpt_every_steps=10, train_glob=os.path.join(self.tmp, "train", "*.bin"),
                 val_glob=os.path.join(self.tmp, "val", "*.bin"), device="cpu", alpha=0.1)
        c.update(kw)
        return c

    def losses(self, run_id):
        path = os.path.join(self.tmp, "ckpt", f"run-{run_id}", "metrics.jsonl")
        return {j["step"]: j["loss"] for j in map(json.loads, open(path).read().splitlines())}

    def test_loader_cursor_resumes_exactly(self):
        from nnue.data import ShardStream
        paths = sorted(glob.glob(os.path.join(self.tmp, "train", "*.bin")))
        full = ShardStream(paths, 128, seed=3, prefetch=0)
        it = iter(full)
        batches = [next(it) for _ in range(30)]
        resumed = iter(ShardStream(paths, 128, seed=3, cursor=batches[9]["cursor"], prefetch=0))
        for k in range(10, 30):
            np.testing.assert_array_equal(next(resumed)["us"], batches[k]["us"])

    def test_resume_gives_identical_loss_curve(self):
        from nnue.train import train
        train(self.cfg("straight"), resume="none")
        train(self.cfg("interrupted", stop_after=20), resume="none")     # killed at step 20 ...
        train(self.cfg("interrupted"), resume="auto")                    # ... then resumed
        a, b = self.losses("straight"), self.losses("interrupted")
        for s in range(21, 41):
            self.assertAlmostEqual(a[s], b[s], places=6, msg=f"step {s}")

    def test_config_mismatch_refused(self):
        from nnue.train import train
        train(self.cfg("mm", steps=10), resume="none")
        with self.assertRaises(SystemExit):
            train(self.cfg("mm", lr=5e-3, steps=20), resume="auto")

    def test_training_reduces_loss_and_export_roundtrip(self):
        from nnue.train import train
        from nnue.export import export_net, load_checkpoint_model, write_reference
        import torch
        out = train(self.cfg("learn", steps=60, lr=3e-3), resume="none")
        l = self.losses("learn")
        self.assertLess(np.mean([l[s] for s in range(51, 61)]), np.mean([l[s] for s in range(1, 11)]))
        m = load_checkpoint_model(out["final"])
        info = export_net(m, os.path.join(self.tmp, "nets"))
        self.assertTrue(os.path.getsize(info["quantized"]) > 768 * 16 * 2)
        write_reference(m, glob.glob(os.path.join(self.tmp, "val", "*.bin"))[0], os.path.join(self.tmp, "ref.jsonl"), 50)
        lines = [json.loads(x) for x in open(os.path.join(self.tmp, "ref.jsonl")).read().splitlines()]
        self.assertEqual(len(lines), 50)
        self.assertEqual(len(lines[0]["out"]), 3)

    def test_engine_parity_with_pytorch(self):
        """Float C++ inference must match PyTorch; the quantised path must stay close to float."""
        exe = os.path.join(ROOT, "build", "bulwark")
        if not os.path.exists(exe):
            self.skipTest("bulwark not built")
        import subprocess
        from nnue.train import train
        from nnue.export import export_net, load_checkpoint_model, write_reference
        out = train(self.cfg("parity", steps=30, lr=3e-3, n=32), resume="none")
        m = load_checkpoint_model(out["final"])
        info = export_net(m, os.path.join(self.tmp, "nets_p"))
        ref = os.path.join(self.tmp, "ref_p.jsonl")
        write_reference(m, glob.glob(os.path.join(self.tmp, "val", "*.bin"))[0], ref, 300)
        errs = {}
        for kind in ("float", "quantized"):
            res = subprocess.run([exe], input=f"verifynet {info[kind]} {ref}\nquit\n", capture_output=True, text=True).stdout
            line = [l for l in res.splitlines() if l.startswith("verifynet")][0]
            errs[kind] = float(line.split("max abs error ")[1].split(",")[0])
        self.assertLess(errs["float"], 1e-4)
        self.assertLess(errs["quantized"], 0.15)

    def test_model_color_flip_invariance(self):
        import torch
        from nnue.model import NNUE
        from nnue.data import features
        torch.manual_seed(0)
        m = NNUE(32, 3)
        for fen in Features.FENS:
            u1, t1 = features(R.from_fen(fen)); u2, t2 = features(R.from_fen(color_flip(fen)))
            o1 = m(torch.from_numpy(u1), torch.from_numpy(t1)); o2 = m(torch.from_numpy(u2), torch.from_numpy(t2))
            torch.testing.assert_close(o1, o2, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
