#pragma once
#include "search.h"
// Fixed-depth search over a set of FENs (standard + Chess960). Returns the total node count.
uint64_t run_bench(Searcher& s, int depth, bool print_each = false);
#include <vector>
std::vector<const char*> bench_fens();
