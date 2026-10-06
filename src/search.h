#pragma once
#include "movegen.h"
#include "tt.h"
#include <atomic>
#include <chrono>
#include <string>
#include <vector>

constexpr int MAX_PLY = 128;
constexpr int INF = 32001;
constexpr int MATE = 32000;
constexpr int MATE_IN_MAX = MATE - MAX_PLY;
constexpr int NO_EVAL = -INF - 1;

struct Limits {
    int depth = 0;                 // 0 = none
    uint64_t nodes = 0;            // 0 = none
    int64_t movetime = 0;          // ms, 0 = none
    int64_t time[2] = {0, 0};      // remaining ms per color
    int64_t inc[2] = {0, 0};
    int movestogo = 0;
    bool infinite = false;
    bool use_clock = false;        // wtime/btime given
};

struct Options {
    int hash_mb = 16;
    int move_overhead = 10;
    bool chess960 = false;
};

class Searcher {
public:
    Searcher() { tt.resize(16); }
    TT tt;
    Options opt;
    std::atomic<bool> stop{false};

    void new_game();
    // Search from `root`; `history` holds the keys of all earlier game positions (excluding root).
    Move think(const Position& root, const std::vector<U64>& history, const Limits& lim, bool print = true);
    uint64_t nodes() const { return nodes_; }

private:
    template <bool PV> int search(int ply, int depth, int alpha, int beta);
    int qsearch(int ply, int alpha, int beta);
    bool is_draw(int ply) const;
    void check_time();
    int score_moves(const Position& pos, int ply, Move tt_move, MoveList& ml, int* scores);
    void update_history(Color c, Move m, int bonus);
    void print_info(int depth, int score);
    int64_t elapsed_ms() const;

    Position stack_[MAX_PLY + 4];
    std::vector<U64> keys_;
    size_t hist_n_ = 0;
    Move pv_[MAX_PLY + 4][MAX_PLY + 4];
    int pv_len_[MAX_PLY + 4];
    Move killers_[MAX_PLY + 4][2];
    int16_t history_[COLOR_NB][64][64];
    int static_eval_[MAX_PLY + 4];

    Limits lim_;
    bool print_ = true;
    uint64_t nodes_ = 0;
    int seldepth_ = 0;
    int root_depth_ = 0;
    int64_t soft_ms_ = 0, hard_ms_ = 0;
    std::chrono::steady_clock::time_point start_;
    bool stopped_ = false;
};
