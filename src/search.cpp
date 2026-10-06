#include "search.h"
#include "eval.h"
#include <algorithm>
#include <cstdio>
#include <cstring>

namespace {
const int PieceVal[7] = {100, 320, 330, 500, 900, 0, 0};
}

void Searcher::new_game() {
    tt.clear();
    std::memset(history_, 0, sizeof(history_));
    std::memset(killers_, 0, sizeof(killers_));
}

int64_t Searcher::elapsed_ms() const {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_).count();
}

void Searcher::check_time() {
    if (root_depth_ <= 1) return;   // always finish depth 1 so there is a move to play
    if (stop.load(std::memory_order_relaxed)) { stopped_ = true; return; }
    if (lim_.nodes && nodes_ >= lim_.nodes) { stopped_ = true; return; }
    if (hard_ms_ && elapsed_ms() >= hard_ms_) stopped_ = true;
}

bool Searcher::is_draw(int ply) const {
    const Position& pos = stack_[ply];
    if (pos.insufficient_material()) return true;
    size_t p = hist_n_ + ply;
    int lim = std::min<int>(pos.rule50, int(p));
    int count = 0;
    for (int i = 4; i <= lim; i += 2) {
        if (keys_[p - i] == keys_[p]) {
            if (p - i >= hist_n_) return true;   // repeated inside the search tree: one repetition is a draw
            if (++count >= 2) return true;       // against game history: threefold
        }
    }
    return false;
}

void Searcher::update_history(Color c, Move m, int bonus) {
    int16_t& h = history_[c][m.from()][m.to()];
    int b = std::clamp(bonus, -2000, 2000);
    h += b - h * std::abs(b) / 16384;
}

int Searcher::score_moves(const Position& pos, int ply, Move tt_move, MoveList& ml, int* scores) {
    for (int i = 0; i < ml.size(); ++i) {
        Move m = ml.moves[i];
        int s;
        if (m == tt_move) s = 1 << 30;
        else if (m.is_capture() || (m.is_promo() && m.promo_type() == QUEEN)) {
            int victim = m.is_ep() ? PAWN : (pos.board[m.to()] == NO_PIECE ? NO_PT : type_of(pos.board[m.to()]));
            int attacker = type_of(pos.board[m.from()]);
            int mvv = 16 * PieceVal[victim] - PieceVal[attacker] / 8 + (m.is_promo() ? 800 : 0);
            s = pos.see_ge(m, 0) ? 1000000 + mvv : -1000000 + mvv;
        } else if (m == killers_[ply][0]) s = 900000;
        else if (m == killers_[ply][1]) s = 800000;
        else s = history_[pos.stm][m.from()][m.to()];
        scores[i] = s;
    }
    return ml.size();
}

int Searcher::qsearch(int ply, int alpha, int beta) {
    if ((++nodes_ & 2047) == 0) check_time();
    if (stopped_) return 0;
    seldepth_ = std::max(seldepth_, ply);
    const Position& pos = stack_[ply];
    keys_[hist_n_ + ply] = pos.key;
    pv_len_[ply] = ply;

    if (ply > 0 && is_draw(ply)) return 0;
    if (pos.rule50 >= 100) { MoveList l; generate_legal(pos, l); return (pos.in_check() && !l.size()) ? -MATE + ply : 0; }
    if (ply >= MAX_PLY) return Eval::evaluate(pos);

    TTData tte;
    bool found = tt.probe(pos.key, tte);
    if (found) {
        int s = tte.score;
        if (s >= MATE_IN_MAX) s -= ply; else if (s <= -MATE_IN_MAX) s += ply;
        if (pos.rule50 < 90 && ((tte.bound == BOUND_EXACT) || (tte.bound == BOUND_LOWER && s >= beta) ||
                                (tte.bound == BOUND_UPPER && s <= alpha)))
            return s;
    }

    bool in_check = pos.in_check();
    int best = -INF, eval = NO_EVAL;
    if (!in_check) {
        eval = found && tte.eval != NO_EVAL ? tte.eval : Eval::evaluate(pos);
        best = eval;
        if (best >= beta) return best;
        alpha = std::max(alpha, best);
    }

    MoveList ml;
    if (in_check) generate<GEN_ALL>(pos, ml); else generate<GEN_CAPTURES>(pos, ml);
    int scores[256];
    score_moves(pos, ply, found ? tte.move : MOVE_NONE, ml, scores);

    Move best_move = MOVE_NONE;
    int legal = 0;
    for (int i = 0; i < ml.size(); ++i) {
        int bi = i;
        for (int j = i + 1; j < ml.size(); ++j) if (scores[j] > scores[bi]) bi = j;
        std::swap(ml.moves[i], ml.moves[bi]); std::swap(scores[i], scores[bi]);
        Move m = ml.moves[i];
        if (!pos.is_legal(m)) continue;
        ++legal;
        if (!in_check && scores[i] < -500000) continue;   // skip losing captures (SEE < 0) in qsearch
        stack_[ply + 1] = pos.make(m);
        int score = -qsearch(ply + 1, -beta, -alpha);
        if (stopped_) return 0;
        if (score > best) {
            best = score;
            best_move = m;
            if (score > alpha) {
                alpha = score;
                if (score >= beta) break;
            }
        }
    }
    if (in_check && legal == 0) return -MATE + ply;

    int stored = best;
    if (stored >= MATE_IN_MAX) stored += ply; else if (stored <= -MATE_IN_MAX) stored -= ply;
    tt.store(pos.key, best_move, stored, eval == NO_EVAL ? NO_EVAL : eval, 0,
             best >= beta ? BOUND_LOWER : BOUND_UPPER);
    return best;
}

template <bool PV>
int Searcher::search(int ply, int depth, int alpha, int beta) {
    if (depth <= 0) return qsearch(ply, alpha, beta);
    if ((++nodes_ & 2047) == 0) check_time();
    if (stopped_) return 0;
    seldepth_ = std::max(seldepth_, ply);
    const Position& pos = stack_[ply];
    keys_[hist_n_ + ply] = pos.key;
    pv_len_[ply] = ply;
    const bool root = ply == 0;
    const bool in_check = pos.in_check();

    if (!root) {
        if (pos.rule50 >= 100) {
            MoveList l; generate_legal(pos, l);
            return (in_check && !l.size()) ? -MATE + ply : 0;
        }
        if (is_draw(ply)) return 0;
        if (ply >= MAX_PLY) return in_check ? 0 : Eval::evaluate(pos);
        alpha = std::max(alpha, -MATE + ply);
        beta = std::min(beta, MATE - ply - 1);
        if (alpha >= beta) return alpha;
    }

    TTData tte;
    bool found = tt.probe(pos.key, tte);
    Move tt_move = found ? tte.move : MOVE_NONE;
    if (found && !PV && tte.depth >= depth && pos.rule50 < 90) {
        int s = tte.score;
        if (s >= MATE_IN_MAX) s -= ply; else if (s <= -MATE_IN_MAX) s += ply;
        if ((tte.bound == BOUND_EXACT) || (tte.bound == BOUND_LOWER && s >= beta) ||
            (tte.bound == BOUND_UPPER && s <= alpha))
            return s;
    }

    int eval = in_check ? NO_EVAL : (found && tte.eval != NO_EVAL ? tte.eval : Eval::evaluate(pos));
    static_eval_[ply] = eval;

    MoveList ml;
    generate<GEN_ALL>(pos, ml);
    int scores[256];
    score_moves(pos, ply, tt_move, ml, scores);

    int best = -INF;
    Move best_move = MOVE_NONE;
    int legal = 0;
    Move quiets[64];
    int nq = 0;
    const int orig_alpha = alpha;

    for (int i = 0; i < ml.size(); ++i) {
        int bi = i;
        for (int j = i + 1; j < ml.size(); ++j) if (scores[j] > scores[bi]) bi = j;
        std::swap(ml.moves[i], ml.moves[bi]); std::swap(scores[i], scores[bi]);
        Move m = ml.moves[i];
        if (!pos.is_legal(m)) continue;
        ++legal;
        bool quiet = !m.is_capture() && !m.is_promo();

        stack_[ply + 1] = pos.make(m);
        tt.prefetch(stack_[ply + 1].key);
        int ext = stack_[ply + 1].in_check() ? 1 : 0;
        int nd = depth - 1 + ext;
        int score;
        if (legal == 1) {
            score = -search<PV>(ply + 1, nd, -beta, -alpha);
        } else {
            score = -search<false>(ply + 1, nd, -alpha - 1, -alpha);
            if (PV && score > alpha && score < beta) score = -search<true>(ply + 1, nd, -beta, -alpha);
        }
        if (stopped_) return 0;

        if (quiet && nq < 64) quiets[nq++] = m;
        if (score > best) {
            best = score;
            best_move = m;
            if (score > alpha) {
                alpha = score;
                if (PV) {   // triangular PV update
                    pv_[ply][ply] = m;
                    for (int k = ply + 1; k < pv_len_[ply + 1]; ++k) pv_[ply][k] = pv_[ply + 1][k];
                    pv_len_[ply] = std::max(pv_len_[ply + 1], ply + 1);
                }
                if (score >= beta) {
                    if (quiet) {
                        if (killers_[ply][0] != m) { killers_[ply][1] = killers_[ply][0]; killers_[ply][0] = m; }
                        int bonus = depth * depth;
                        update_history(pos.stm, m, bonus);
                        for (int k = 0; k < nq - 1; ++k) update_history(pos.stm, quiets[k], -bonus);
                    }
                    break;
                }
            }
        }
    }

    if (legal == 0) return in_check ? -MATE + ply : 0;

    int stored = best;
    if (stored >= MATE_IN_MAX) stored += ply; else if (stored <= -MATE_IN_MAX) stored -= ply;
    Bound b = best >= beta ? BOUND_LOWER : (PV && best > orig_alpha) ? BOUND_EXACT : BOUND_UPPER;
    tt.store(pos.key, best_move, stored, eval, depth, b);
    return best;
}

void Searcher::print_info(int depth, int score) {
    int64_t ms = std::max<int64_t>(elapsed_ms(), 1);
    std::string sc;
    if (std::abs(score) >= MATE_IN_MAX) {
        int n = (MATE - std::abs(score) + 1) / 2;
        sc = "mate " + std::to_string(score > 0 ? n : -n);
    } else sc = "cp " + std::to_string(score);
    std::string pv;
    for (int i = 0; i < pv_len_[0]; ++i) pv += " " + stack_[0].move_to_uci(pv_[0][i], opt.chess960);
    std::printf("info depth %d seldepth %d score %s nodes %llu nps %llu hashfull %d time %lld pv%s\n", depth,
                seldepth_, sc.c_str(), (unsigned long long)nodes_, (unsigned long long)(nodes_ * 1000 / ms),
                tt.hashfull(), (long long)ms, pv.c_str());
    std::fflush(stdout);
}

Move Searcher::think(const Position& root, const std::vector<U64>& history, const Limits& lim, bool print) {
    lim_ = lim;
    print_ = print;
    start_ = std::chrono::steady_clock::now();
    nodes_ = 0;
    stopped_ = false;
    seldepth_ = 0;
    tt.new_search();
    hist_n_ = history.size();
    keys_.assign(hist_n_ + MAX_PLY + 8, 0);
    std::copy(history.begin(), history.end(), keys_.begin());
    stack_[0] = root;
    std::memset(killers_, 0, sizeof(killers_));

    // Time management: soft = remaining/20 + inc/2; hard a few times soft, capped well below remaining.
    soft_ms_ = hard_ms_ = 0;
    if (lim.movetime) soft_ms_ = hard_ms_ = lim.movetime;
    else if (lim.use_clock) {
        int c = root.stm;
        int64_t rem = lim.time[c], inc = lim.inc[c];
        int mtg = lim.movestogo ? lim.movestogo : 20;
        soft_ms_ = rem / mtg + inc / 2;
        hard_ms_ = std::min<int64_t>(soft_ms_ * 4, rem * 6 / 10);
        soft_ms_ = std::min(soft_ms_, hard_ms_);
        int64_t oh = opt.move_overhead;
        hard_ms_ = std::max<int64_t>(hard_ms_ - oh, 1);
        soft_ms_ = std::max<int64_t>(soft_ms_ - oh, 1);
    }

    Move best = MOVE_NONE;
    int max_depth = lim.depth ? std::min(lim.depth, MAX_PLY - 1) : MAX_PLY - 1;
    for (int d = 1; d <= max_depth; ++d) {
        root_depth_ = d;
        int score = search<true>(0, d, -INF, INF);
        if (stopped_) break;
        if (pv_len_[0] > 0) best = pv_[0][0];
        if (print) print_info(d, score);
        if (soft_ms_ && elapsed_ms() >= soft_ms_) break;
        if (!lim.infinite && score >= MATE_IN_MAX && MATE - score <= d) break;   // shortest mate already proven
    }
    if (!best) {   // no completed iteration or no legal moves: fall back to any legal move
        MoveList l; generate_legal(root, l);
        if (l.size()) best = l.moves[0];
    }
    return best;
}

template int Searcher::search<true>(int, int, int, int);
template int Searcher::search<false>(int, int, int, int);
