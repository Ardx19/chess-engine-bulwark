#pragma once
#include "position.h"

U64 perft(const Position& pos, int depth);                 // bulk counting at depth 1
void perft_divide(const Position& pos, int depth, bool chess960);
