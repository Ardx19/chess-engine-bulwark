# Phase 4 pipeline (data + training)

```
Lc0 tar (storage.lczero.org, ODbL) --pipeline/convert.py--> tools/lc0_convert --> master shard (56 B records)
   --pipeline/relabel.py (Stockfish 19, W/D/L)--> labeled master shard
   --pipeline/make_training.py (split by game id, dedup)--> train/ val/ test/ shards (40 B records)
   --nnue/train.py (checkpoint/resume)--> final.pt --nnue/export.py--> net-<sha8>.nnue + .nnuef + ref-<sha8>.jsonl
```

* `kaggle/pilot.sh` runs one CPU session of the data round (pilot: few tars; production: more tars, `RELABEL_LIMIT=0`).
* `kaggle/train.sh` trains and exports on a GPU session.
* Facts measured on real data (test91 run, input format 1, version 6): records are 8,356 B; every plane byte is
  bit-reversed; planes are from the side to move's view (ranks mirrored for black); a `.gz` member is one game
  (about 120 records); about 5,250 games per tar (~635k records, ~130 MB); the run mixes in Chess960 games.
* Filters: <8 plies of history, in check, best move capture or promotion, |root_q| > 0.98, malformed/illegal, deletion mark.
  Castling rights that do not match the standard king/rook squares are dropped (Chess960 games).
* Tablebase override: <= 5 pieces, rule50 == 0, no castling -> exact outcome replaces the target (flag bit 0).
  Positions with a running halfmove clock are not overridden (WDL tables assume rule50 = 0).
* Feature layout (shared with the engine): `(rel_color*6 + type)*64 + sq`, `sq ^= 56` for black's perspective, pad = 768.
* Tests: `python3 -m unittest discover -s tests -p test_pipeline.py` (set `LC0_TAR` for the real-data check).
* Data checks before scaling (WP-B):
  - `python3 -m pipeline.tb_check 'master/*.bin' --oracle lichess` (or `--oracle local --syzygy DIR`): tablebase-overridden
    labels against the Lichess tablebase API, per category (cursed wins / blessed losses must be stored as draws).
  - `python3 -m pipeline.check_splits --master 'master/*.bin' --training data/training --max-overlap 0.05`: session-id
    reuse, and exact-board overlap of val/test with train (`--drop-overlap` removes it from val/test).
  - `python3 -m pipeline.to_bullet 'training/train/*.bin' bullet.txt`: bullet text (`FEN | score | result`, White's
    view) for the N1 cross-check with bullet's `simple` example (eval scale 400, WDL weight 0).
