#pragma once
#include "movegen.h"
#include "nnue.h"
#include "tt.h"
#include <atomic>
#include <chrono>
#include <memory>
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
    std::vector<Move> searchmoves; // restrict the root moves (empty = all)
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
    // D1/D2: utility U = W + d*D (WDL net) and in-tree draws scored 400*logit(d); d = 0.5 is neutral.
    bool d1_utility = false;       // D1: use d = DrawValue in every search
    int draw_value = 500;          // per mille
    bool d2_defense = false;       // D2 (DefenseMode): d = DefenseDrawValue when the last root P(loss) >= threshold
    int defense_draw_value = 650;  // per mille
    int defense_loss_threshold = 400;   // per mille
    bool d6_perpetual = false;     // D6: quiet checks at the first qsearch ply when far below alpha
    bool d7_fortress = false;      // D7: flat large score + no zeroing move in the PV -> scale toward the draw
    bool show_wdl = false;         // UCI_ShowWDL (WDL nets: net W/D/L at the end of the PV)
    std::string syzygy_path;
    bool* flag(const std::string& name);   // UCI option name -> switch, or nullptr
};

class Searcher {
public:
    Searcher() : acc_(MAX_PLY + 4), finny_(std::make_unique<NNUE::Finny>()) { tt.resize(16); }
    TT tt;
    Options opt;
    std::atomic<bool> stop{false};

    void new_game();
    // Search from `root`; `history` holds the keys of all earlier game positions (excluding root).
    Move think(const Position& root, const std::vector<U64>& history, const Limits& lim, bool print = true);
    uint64_t nodes() const { return nodes_; }
    // Evaluation: HCE when `net` is null, else the net (quantised: incremental; float: full forward). Not owned.
    void set_net(const NNUE::Net* net) { net_ = net; ++net_gen_; tt.clear(); finny_->clear(); }
    const NNUE::Net* net() const { return net_; }
    double draw_weight() const { return d_; }          // d used by the last search
    // Static evaluation of `pos` in internal units from its side to move's view, at neutral utility (d = 0.5).
    int static_eval(const Position& pos);

private:
    template <bool PV> int search(int ply, int depth, int alpha, int beta);
    int qsearch(int ply, int alpha, int beta, bool first = false);
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
    int evaluate(int ply);                     // raw static eval (utility-transformed for WDL nets)
    int adjust(int ply, int raw) const;        // D3 / D7 scaling toward the draw score
    int draw_score(int ply) const { return stack_[ply].stm == root_stm_ ? draw_us_ : -draw_us_; }
    int utility_units(const double* wdl, Color stm) const;
    bool leaf_wdl(int* wdl_permille);          // net W/D/L at the end of the root PV, root side's view

    const NNUE::Net* net_ = nullptr;
    uint64_t net_gen_ = 0;
    std::vector<NNUE::Accumulator> acc_;
    std::unique_ptr<NNUE::Finny> finny_;
    double d_ = 0.5;                           // draw weight for the root side this search
    int draw_us_ = 0;                          // 400*logit(d)
    Color root_stm_ = WHITE;
    uint64_t util_tag_ = ~0ULL;
    double last_wdl_[3] = {0, 1, 0};           // net W/D/L after the last search, from last_color_'s view
    Color last_color_ = WHITE;
    bool fortress_ = false;
    int fortress_moves_ = 0;

    Limits lim_;
    bool print_ = true;
    uint64_t nodes_ = 0;
    int seldepth_ = 0;
    int root_depth_ = 0;
    int64_t soft_ms_ = 0, hard_ms_ = 0;
    std::chrono::steady_clock::time_point start_;
    bool stopped_ = false;
};
