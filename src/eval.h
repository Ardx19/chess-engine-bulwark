#pragma once
#include "position.h"

namespace Eval {
// Static evaluation from the side to move's view, in internal units (centipawn-like).
int evaluate(const Position& pos);
}
