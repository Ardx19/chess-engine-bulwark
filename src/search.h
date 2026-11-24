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
constexpr int TB_WIN = 20000;                      // tablebase win score, minus ply
constexpr int DECISIVE = TB_WIN - MAX_PLY;       // |score| >= DECISIVE: forced mate or tablebase result
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

// Every ladder step / defensive feature is a switch so one binary can be SPRT-tested against itself.
struct Options {
    int hash_mb = 16;
    int move_overhead = 10;
    bool chess960 = false;
    bool aspiration = true;        // ladder 1
    bool rfp = true;               // 2  reverse futility
    bool nmp = true;               // 3  null move
    bool lmr = true;               // 4  late move reductions
    bool killers_history = true;   // 5
    bool lmp_futility = true;      // 6
    bool iir = true;               // 7  ladder -15 +/- 22 (600 g); re-test +21.7 +/- 14.1 (1200 g) -> on
    bool improving = true;         // 8
    bool see_prune = true;         // 9
    bool tm_scaling = true;        // time management: node fraction + best-move stability
    bool d3_rule50 = true;         // D3: scale eval toward draw with the halfmove clock
    bool d5_stalemate = true;      // D5: stalemate guard in qsearch at low material
    bool d4_syzygy = true;         // D4: tablebase probes (needs SyzygyPath)
    std::string syzygy_path;
    bool* flag(const std::string& name);   // UCI option name -> switch, or nullptr
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
    bool null_prev_[MAX_PLY + 4];
    uint64_t root_move_nodes_[64][64];
    MoveList root_allowed_;
    bool root_filtered_ = false;
    uint64_t tb_hits_ = 0;
    int eval_of(const Position& pos) const;

    Limits lim_;
    bool print_ = true;
    uint64_t nodes_ = 0;
    int seldepth_ = 0;
    int root_depth_ = 0;
    int64_t soft_ms_ = 0, hard_ms_ = 0;
    std::chrono::steady_clock::time_point start_;
    bool stopped_ = false;
};
