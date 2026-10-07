import argparse, json
from .relabel import relabel

ap = argparse.ArgumentParser()
ap.add_argument("inp"); ap.add_argument("out"); ap.add_argument("--stockfish", required=True)
ap.add_argument("--nodes", type=int, default=2000); ap.add_argument("--workers", type=int, default=None)
ap.add_argument("--limit", type=int, default=0); ap.add_argument("--syzygy")
a = ap.parse_args()
print(json.dumps(relabel(a.inp, a.out, a.stockfish, a.nodes, a.workers, syzygy=a.syzygy, limit=a.limit), indent=1))
