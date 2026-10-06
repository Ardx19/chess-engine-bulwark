#include "tt.h"
#include <cstring>

void TT::resize(size_t mb) {
    n_ = mb * 1024 * 1024 / sizeof(TTCluster);
    if (n_ < 1) n_ = 1;
    table_.assign(n_, TTCluster{});
    gen_ = 0;
}

void TT::clear() { std::memset(table_.data(), 0, table_.size() * sizeof(TTCluster)); gen_ = 0; }

bool TT::probe(U64 key, TTData& out) const {
    const TTCluster& c = table_[index(key)];
    uint32_t k = uint32_t(key >> 32);
    for (const TTEntry& e : c.e)
        if ((e.gen_bound & 3) && e.key32 == k) {
            out.move = Move(e.move);
            out.score = e.score;
            out.eval = e.eval;
            out.depth = e.depth;
            out.bound = Bound(e.gen_bound & 3);
            return true;
        }
    return false;
}

void TT::store(U64 key, Move m, int score, int eval, int depth, Bound b) {
    TTCluster& c = table_[index(key)];
    uint32_t k = uint32_t(key >> 32);
    TTEntry* rep = &c.e[0];
    for (TTEntry& e : c.e) {
        if (!(e.gen_bound & 3) || e.key32 == k) { rep = &e; break; }
        // replacement value: depth minus an age penalty
        auto val = [&](const TTEntry& x) { return int(x.depth) - 8 * ((gen_ - (x.gen_bound >> 2)) & 63); };
        if (val(e) < val(*rep)) rep = &e;
    }
    bool same = (rep->gen_bound & 3) && rep->key32 == k;
    if (same && !m) m = Move(rep->move);   // keep the old move if we have none
    if (same && b != BOUND_EXACT && depth < rep->depth - 3 && ((gen_ - (rep->gen_bound >> 2)) & 63) == 0) return;
    rep->key32 = k;
    rep->move = m.v;
    rep->score = int16_t(score);
    rep->eval = int16_t(eval);
    rep->depth = uint8_t(depth < 0 ? 0 : depth);
    rep->gen_bound = uint8_t(gen_ << 2 | b);
}

int TT::hashfull() const {
    int cnt = 0;
    size_t n = n_ < 1000 ? n_ : 1000;
    for (size_t i = 0; i < n; ++i)
        for (const TTEntry& e : table_[i].e)
            if ((e.gen_bound & 3) && (e.gen_bound >> 2) == gen_) ++cnt;
    return cnt * 1000 / int(n * 5);
}
