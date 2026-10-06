#pragma once
#include "types.h"
#include <cstddef>
#include <vector>

enum Bound : uint8_t { BOUND_NONE = 0, BOUND_UPPER = 1, BOUND_LOWER = 2, BOUND_EXACT = 3 };

struct TTData {
    Move move;
    int score, eval, depth;
    Bound bound;
};

struct TTEntry {
    uint32_t key32;
    uint16_t move;
    int16_t score;
    int16_t eval;
    uint8_t depth;
    uint8_t gen_bound;   // generation << 2 | bound
};

struct alignas(64) TTCluster {
    TTEntry e[5];
    char pad[4];
};

class TT {
public:
    void resize(size_t mb);
    void clear();
    void new_search() { gen_ = uint8_t((gen_ + 1) & 63); }
    bool probe(U64 key, TTData& out) const;
    void store(U64 key, Move m, int score, int eval, int depth, Bound b);
    int hashfull() const;
    void prefetch(U64 key) const { __builtin_prefetch(&table_[index(key)]); }
private:
    size_t index(U64 key) const { return (size_t)(((unsigned __int128)key * n_) >> 64); }
    std::vector<TTCluster> table_;
    size_t n_ = 0;
    uint8_t gen_ = 0;
};
