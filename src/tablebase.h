#pragma once
#include "movegen.h"
#include <string>

// Syzygy probing through Pyrrhic (MIT). WDL probes assume a zero halfmove clock, so callers only use probe_wdl
// when rule50 == 0 and there are no castling rights; the root filter combines DTZ with the real clock.
namespace TB {
enum Wdl { LOSS = 0, BLESSED_LOSS = 1, DRAW = 2, CURSED_WIN = 3, WIN = 4, FAIL = -1 };

bool init(const std::string& path);   // true if at least one table was loaded
int largest();                        // largest piece count with loaded tables (0 = none)
int probe_wdl(const Position& pos);   // Wdl or FAIL
// Restrict the root moves: won -> moves that keep the (50-move-aware) win; drawn -> moves that keep the draw;
// lost -> no restriction (DTZ is the distance to the next capture or pawn move, not survival time).
// Returns false (and leaves `allowed` empty) if the position is not covered or probing failed.
bool root_filter(const Position& pos, MoveList& allowed);
}
