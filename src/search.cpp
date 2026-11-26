#include "search.h"
#include "eval.h"
#include "tablebase.h"
#include <algorithm>
#include <cstdio>
#include <cmath>
#include <cstring>

namespace {
const int PieceVal[7] = {100, 320, 330, 500, 900, 0, 0};

int LmrTable[64][64];
struct LmrInit {
    LmrInit() {
        for (int d = 1; d < 64; ++d)
            for (int m = 1; m < 64; ++m) LmrTable[d][m] = int(0.78 + std::log(d) * std::log(m) / 2.47);
    }
} lmr_init;
}

bool* Options::flag(const std::string& n) {
    if (n == "Aspiration") return &aspiration;
    if (n == "RFP") return &rfp;
    if (n == "NMP") return &nmp;
    if (n == "LMR") return &lmr;
    if (n == "KillersHistory") return &killers_history;
    if (n == "LMPFutility") return &lmp_futility;
    if (n == "IIR") return &iir;
    if (n == "Improving") return &improving;
    if (n == "SEEPrune") return &see_prune;
    if (n == "ContHist") return &cont_hist;
    if (n == "CaptHist") return &capt_hist;
    if (n == "HistBonus2") return &hist_bonus2;
    if (n == "HistPrune") return &hist_prune;
    if (n == "HistPruneSoft") return &hist_prune_soft;
    if (n == "SingularExt") return &singular_ext;
    if (n == "CorrHist") return &corr_hist;
    if (n == "Razoring") return &razoring;
    if (n == "ProbCut") return &probcut;
    if (n == "TMScaling") return &tm_scaling;
    if (n == "D3_Rule50Scale") return &d3_rule50;
    if (n == "D5_StalemateGuard") return &d5_stalemate;
    if (n == "D4_Syzygy") return &d4_syzygy;
    if (n == "D1_WDLUtility") return &d1_utility;
    if (n == "DefenseMode") return &d2_defense;
    if (n == "D6_PerpetualSeek") return &d6_perpetual;
    if (n == "D7_FortressDetect") return &d7_fortress;
    if (n == "UCI_ShowWDL") return &show_wdl;
    return nullptr;
}

constexpr int EVAL_CAP = 10000;   // static evals stay well inside the mate / tablebase bands

int Searcher::utility_units(const double* p, Color stm) const {
    double d = stm == root_stm_ ? d_ : 1.0 - d_;          // opponent nodes weight draws by 1 - d (negamax-consistent)
    double u = std::clamp(p[0] + d * p[1], 1e-6, 1.0 - 1e-6);
    return int(std::lround(4.0 * opt.wdl_scale * std::log(u / (1.0 - u))));
}

int Searcher::evaluate(int ply) {
    const Position& pos = stack_[ply];
    if (!net_) return Eval::evaluate(pos);
    float out[NNUE::MAX_HEADS];
    if (net_->quantized) {
        int q = ply;
        while (!acc_[q].computed) --q;                     // acc_[0] is refreshed at the root
        for (int i = q + 1; i <= ply; ++i) NNUE::update(*net_, stack_[i - 1], stack_[i], acc_[i - 1], acc_[i], finny_.get());
        NNUE::output(*net_, acc_[ply], pos, out);
    } else {
        net_->forward(pos, out);
    }
    if (net_->heads == 1) return std::clamp(int(std::lround(400.0 * out[0])), -EVAL_CAP, EVAL_CAP);
    double p[3];
    NNUE::wdl_probs(out, net_->heads, p);
    if (net_->dual()) {
        // Dual head: the scalar head is the search eval; a non-neutral draw weight shifts it by the utility change
        // the W/D/L head implies, u(W + d*D) - u(W + D/2), so d = 0.5 is exactly the scalar net.
        int base = int(std::lround(400.0 * out[0]));
        if (d_ != 0.5) {
            double neutral = std::clamp(p[0] + 0.5 * p[1], 1e-6, 1.0 - 1e-6);
            base += utility_units(p, pos.stm) - int(std::lround(4.0 * opt.wdl_scale * std::log(neutral / (1.0 - neutral))));
        }
        return std::clamp(base, -EVAL_CAP, EVAL_CAP);
    }
    return utility_units(p, pos.stm);
}

int Searcher::adjust(int ply, int e) const {
    const int dr = draw_score(ply);
    if (opt.d3_rule50) e = dr + (e - dr) * (212 - std::min(stack_[ply].rule50, 211)) / 212;   // D3: 50-move awareness
    if (fortress_) e = dr + (e - dr) / 2;                                                       // D7
    return e;
}

int Searcher::static_eval(const Position& pos) {
    stack_[0] = pos;
    root_stm_ = pos.stm;
    d_ = 0.5;
    draw_us_ = 0;
    if (net_ && net_->quantized) NNUE::refresh(*net_, pos, acc_[0], finny_.get());
    return evaluate(0);
}

bool Searcher::leaf_wdl(int* wdl) {
    if (!net_ || !net_->wdl()) return false;
    Position p = stack_[0];
    for (int i = 0; i < pv_len_[0]; ++i) {
        MoveList l; generate_legal(p, l);
        bool ok = false;
        for (Move m : l) ok |= m == pv_[0][i];
        if (!ok) break;
        p = p.make(pv_[0][i]);
    }
    float out[NNUE::MAX_HEADS];
    double pr[3];
    net_->forward(p, out);
    NNUE::wdl_probs(out, net_->heads, pr);
    if (p.stm != root_stm_) std::swap(pr[0], pr[2]);
    for (int k = 0; k < 3; ++k) wdl[k] = int(std::lround(pr[k] * 1000));
    return true;
}

void Searcher::new_game() {
    tt.clear();
    util_tag_ = ~0ULL;
    last_wdl_[0] = 0; last_wdl_[1] = 1; last_wdl_[2] = 0;
    fortress_moves_ = 0;
    std::memset(history_, 0, sizeof(history_));
    std::memset(killers_, 0, sizeof(killers_));
    std::fill(cont_.get(), cont_.get() + CONT_SIZE, int16_t(0));
    std::memset(capt_, 0, sizeof(capt_));
    std::memset(corr_, 0, sizeof(corr_));
    std::fill(std::begin(excluded_), std::end(excluded_), MOVE_NONE);
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

int Searcher::cont_score(int ply, Piece pc, int to) const {
    int s = 0;
    for (int k = 1; k <= 2; ++k)
        if (ply - k >= 0 && cm_[ply - k]) s += cm_[ply - k][pc * 64 + to];
    return s;
}

void Searcher::update_cont(int ply, Piece pc, int to, int bonus) {
    int b = std::clamp(bonus, -2000, 2000);
    for (int k = 1; k <= 2; ++k)
        if (ply - k >= 0 && cm_[ply - k]) {
            int16_t& h = cm_[ply - k][pc * 64 + to];
            h += b - h * std::abs(b) / 16384;
        }
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
            if (opt.capt_hist) mvv += capt_entry(pos, m) / 8;
            s = pos.see_ge(m, 0) ? 1000000 + mvv : -1000000 + mvv;
        } else if (!opt.killers_history) s = 0;
        else if (m == killers_[ply][0]) s = 900000;
        else if (m == killers_[ply][1]) s = 800000;
        else s = history_[pos.stm][m.from()][m.to()] + (opt.cont_hist ? cont_score(ply, pos.board[m.from()], m.to()) : 0);
        scores[i] = s;
    }
    return ml.size();
}

int Searcher::qsearch(int ply, int alpha, int beta, bool first) {
    if ((++nodes_ & 2047) == 0) check_time();
    if (stopped_) return 0;
    seldepth_ = std::max(seldepth_, ply);
    const Position& pos = stack_[ply];
    keys_[hist_n_ + ply] = pos.key;
    pv_len_[ply] = ply;

    if (ply > 0 && is_draw(ply)) return draw_score(ply);
    if (pos.rule50 >= 100) { MoveList l; generate_legal(pos, l); return (pos.in_check() && !l.size()) ? -MATE + ply : draw_score(ply); }
    if (ply >= MAX_PLY) return adjust(ply, evaluate(ply));

    TTData tte;
    bool found = tt.probe(pos.key, tte);
    if (found) {
        int s = tte.score;
        if (s >= DECISIVE) s -= ply; else if (s <= -DECISIVE) s += ply;
        if (pos.rule50 < 90 && ((tte.bound == BOUND_EXACT) || (tte.bound == BOUND_LOWER && s >= beta) ||
                                (tte.bound == BOUND_UPPER && s <= alpha)))
            return s;
    }

    bool in_check = pos.in_check();
    int best = -INF, eval = NO_EVAL;
    bool quiet_checks = false;
    if (!in_check) {
        // D5: with little material a side with no legal move is stalemated; do not stand pat on it.
        if (opt.d5_stalemate && popcount(pos.pieces()) <= 8) {
            MoveList l; generate_legal(pos, l);
            if (!l.size()) return draw_score(ply);
        }
        eval = found && tte.eval != NO_EVAL ? tte.eval : evaluate(ply);
        best = corrected(pos, adjust(ply, eval));
        if (best >= beta) return best;
        // D6: far below alpha at the first qsearch ply, also try quiet checks (perpetual-check seeking).
        quiet_checks = opt.d6_perpetual && first && best + 250 <= alpha && alpha < DECISIVE;
        alpha = std::max(alpha, best);
    }

    MoveList ml;
    if (in_check || quiet_checks) generate<GEN_ALL>(pos, ml); else generate<GEN_CAPTURES>(pos, ml);
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
        acc_[ply + 1].computed = false;
        if (quiet_checks && !m.is_capture() && !m.is_promo() && !stack_[ply + 1].in_check()) continue;
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
    if (stored >= DECISIVE) stored += ply; else if (stored <= -DECISIVE) stored -= ply;
    tt.store(pos.key, best_move, stored, eval, 0, best >= beta ? BOUND_LOWER : BOUND_UPPER);
    return best;
}

template <bool PV>
int Searcher::search(int ply, int depth, int alpha, int beta) {
    if (depth <= 0) return qsearch(ply, alpha, beta, true);
    if ((++nodes_ & 2047) == 0) check_time();
    if (stopped_) return 0;
    seldepth_ = std::max(seldepth_, ply);
    const Position& pos = stack_[ply];
    keys_[hist_n_ + ply] = pos.key;
    pv_len_[ply] = ply;
    const bool root = ply == 0;
    const bool in_check = pos.in_check();
    const Move excl = excluded_[ply];

    if (!root) {
        if (pos.rule50 >= 100) {
            MoveList l; generate_legal(pos, l);
            return (in_check && !l.size()) ? -MATE + ply : draw_score(ply);
        }
        if (is_draw(ply)) return draw_score(ply);
        if (ply >= MAX_PLY) return in_check ? draw_score(ply) : adjust(ply, evaluate(ply));
        alpha = std::max(alpha, -MATE + ply);
        beta = std::min(beta, MATE - ply - 1);
        if (alpha >= beta) return alpha;
    }

    TTData tte;
    bool found = tt.probe(pos.key, tte);
    Move tt_move = found ? tte.move : MOVE_NONE;
    if (found && !PV && !excl && tte.depth >= depth && pos.rule50 < 90) {
        int s = tte.score;
        if (s >= DECISIVE) s -= ply; else if (s <= -DECISIVE) s += ply;
        if ((tte.bound == BOUND_EXACT) || (tte.bound == BOUND_LOWER && s >= beta) ||
            (tte.bound == BOUND_UPPER && s <= alpha))
            return s;
    }

    // D4: Syzygy WDL probe. Tables assume a zero halfmove clock and no castling rights.
    if (opt.d4_syzygy && !root && !excl && pos.rule50 == 0 && !pos.castling && TB::largest() > 0 &&
        popcount(pos.pieces()) <= TB::largest()) {
        int w = TB::probe_wdl(pos);
        if (w != TB::FAIL) {
            ++tb_hits_;
            int sc = w == TB::WIN ? TB_WIN - ply : w == TB::LOSS ? -TB_WIN + ply : draw_score(ply);   // cursed/blessed = draw
            return sc;
        }
    }

    int raw_eval = in_check ? NO_EVAL : (found && tte.eval != NO_EVAL ? tte.eval : evaluate(ply));
    const int base_eval = in_check ? NO_EVAL : adjust(ply, raw_eval);
    int eval = in_check ? NO_EVAL : corrected(pos, base_eval);
    static_eval_[ply] = eval;
    const bool improving = opt.improving && !in_check && ply >= 2 && static_eval_[ply - 2] != NO_EVAL &&
                           eval > static_eval_[ply - 2];
    const bool not_mated_range = beta > -DECISIVE && beta < DECISIVE;

    if (!PV && !in_check && !root && !excl) {
        // Razoring: hopeless at low depth -> verify with the capture search only
        if (opt.razoring && depth <= 3 && eval + 250 * depth < alpha && not_mated_range) {
            int v = qsearch(ply, alpha, alpha + 1);
            if (stopped_) return 0;
            if (v < alpha) return v;
        }
        // 2. Reverse futility pruning
        if (opt.rfp && depth <= 8 && not_mated_range && eval - 80 * (depth - improving) >= beta) return eval;
        // 3. Null move pruning
        if (opt.nmp && depth >= 3 && !null_prev_[ply] && eval >= beta && not_mated_range &&
            pos.has_non_pawn_material(pos.stm)) {
            int R = 3 + depth / 3 + std::min((eval - beta) / 200, 3);
            stack_[ply + 1] = pos.make_null();
            cm_[ply] = nullptr;
            acc_[ply + 1].computed = false;
            null_prev_[ply + 1] = true;
            int s = -search<false>(ply + 1, depth - R, -beta, -beta + 1);
            null_prev_[ply + 1] = false;
            if (stopped_) return 0;
            if (s >= beta) return s >= DECISIVE ? beta : s;
        }
        // ProbCut: a good capture that beats beta + margin at reduced depth cuts this node
        if (opt.probcut && depth >= 5 && not_mated_range) {
            const int pbeta = beta + 200;
            int tts = found ? tte.score : 0;
            if (!(found && tte.depth >= depth - 3 && tts < pbeta)) {
                MoveList cl;
                generate<GEN_CAPTURES>(pos, cl);
                for (Move m : cl) {
                    if (!pos.is_legal(m) || !pos.see_ge(m, pbeta - eval)) continue;
                    cm_[ply] = opt.cont_hist ? &cont_[size_t(pos.board[m.from()] * 64 + m.to()) * 768] : nullptr;
                    stack_[ply + 1] = pos.make(m);
                    acc_[ply + 1].computed = false;
                    null_prev_[ply + 1] = false;
                    int v = -qsearch(ply + 1, -pbeta, -pbeta + 1);
                    if (v >= pbeta && !stopped_) v = -search<false>(ply + 1, depth - 4, -pbeta, -pbeta + 1);
                    if (stopped_) return 0;
                    if (v >= pbeta) {
                        tt.store(pos.key, m, v, raw_eval, depth - 3, BOUND_LOWER);
                        return v;
                    }
                }
            }
        }
    }
    // 7. Internal iterative reduction
    if (opt.iir && !tt_move && depth >= 4 && !in_check) --depth;

    MoveList ml;
    generate<GEN_ALL>(pos, ml);
    int scores[256];
    score_moves(pos, ply, tt_move, ml, scores);

    int best = -INF;
    Move best_move = MOVE_NONE;
    int legal = 0;
    Move quiets[64], caps[32];
    int nq = 0, nc = 0;
    const int orig_alpha = alpha;

    // Singular extension: if every move but the TT move fails low against a margin below the TT score, the TT move
    // is singular and gets one ply more; if even that search beats beta, cut (multi-cut); if the TT score alone
    // beats beta, reduce the TT move instead (negative extension).
    int sing_ext = 0;
    if (opt.singular_ext && !root && !excl && tt_move && depth >= 7 && found && tte.depth >= depth - 3 &&
        (tte.bound & BOUND_LOWER) && std::abs(tte.score) < DECISIVE && ply < 2 * root_depth_) {
        const int ttv = tte.score;
        const int sbeta = ttv - 2 * depth;
        excluded_[ply] = tt_move;
        int v = search<false>(ply, (depth - 1) / 2, sbeta - 1, sbeta);
        excluded_[ply] = MOVE_NONE;
        if (stopped_) return 0;
        if (v < sbeta) sing_ext = 1;
        else if (sbeta >= beta) return sbeta;
        else if (ttv >= beta) sing_ext = -1;
    }

    for (int i = 0; i < ml.size(); ++i) {
        int bi = i;
        for (int j = i + 1; j < ml.size(); ++j) if (scores[j] > scores[bi]) bi = j;
        std::swap(ml.moves[i], ml.moves[bi]); std::swap(scores[i], scores[bi]);
        Move m = ml.moves[i];
        if (m == excl || !pos.is_legal(m)) continue;
        if (root && root_filtered_) {
            bool allowed = false;
            for (Move a : root_allowed_) if (a == m) { allowed = true; break; }
            if (!allowed) continue;
        }
        ++legal;
        const bool quiet = !m.is_capture() && !m.is_promo();
        const bool prune_ok = !root && best > -DECISIVE && pos.has_non_pawn_material(pos.stm) && !in_check;

        if (prune_ok && legal > 1) {
            // 6. Late move pruning
            if (opt.lmp_futility && quiet && depth <= 8 && legal > (3 + depth * depth) * (1 + improving)) continue;
            // History pruning: quiets whose history says they rarely work
            if ((opt.hist_prune || opt.hist_prune_soft) && quiet && m != killers_[ply][0] && m != killers_[ply][1]) {
                int h = history_[pos.stm][m.from()][m.to()] + (opt.cont_hist ? cont_score(ply, pos.board[m.from()], m.to()) : 0);
                if ((opt.hist_prune && depth <= 4 && h < -2048 * depth) || (opt.hist_prune_soft && depth <= 3 && h < -4096 * depth))
                    continue;
            }
            // 9. SEE pruning
            if (opt.see_prune && depth <= 6 && !m.is_promo()) {
                if (quiet ? !pos.see_ge(m, -25 * depth * depth) : !pos.see_ge(m, -90 * depth)) continue;
            }
        }

        cm_[ply] = opt.cont_hist ? &cont_[size_t(pos.board[m.from()] * 64 + m.to()) * 768] : nullptr;
        stack_[ply + 1] = pos.make(m);
        acc_[ply + 1].computed = false;
        null_prev_[ply + 1] = false;
        tt.prefetch(stack_[ply + 1].key);
        const bool gives_check = stack_[ply + 1].in_check();

        // 6. Futility pruning (needs to know the move does not give check)
        if (prune_ok && legal > 1 && opt.lmp_futility && quiet && !gives_check && depth <= 7 &&
            eval + 90 + 90 * depth <= alpha)
            continue;

        int nd = depth - 1 + (gives_check ? 1 : 0) + (m == tt_move ? sing_ext : 0);
        int score;
        uint64_t before = nodes_;
        if (legal == 1) {
            score = -search<PV>(ply + 1, nd, -beta, -alpha);
        } else {
            int r = 0;
            // 4. Late move reductions
            if (opt.lmr && depth >= 3 && quiet) {
                r = LmrTable[std::min(depth, 63)][std::min(legal, 63)];
                r -= PV;
                r -= improving;
                r -= gives_check;
                if (opt.killers_history) {
                    int hs = history_[pos.stm][m.from()][m.to()];
                    if (opt.cont_hist) r -= std::clamp((hs + cont_score(ply, pos.board[m.from()], m.to())) / 8192, -2, 2);
                    else r -= std::clamp(hs / 4096, -2, 2);
                }
                r = std::clamp(r, 0, nd - 1);
            }
            score = -search<false>(ply + 1, nd - r, -alpha - 1, -alpha);
            if (score > alpha && r > 0) score = -search<false>(ply + 1, nd, -alpha - 1, -alpha);
            if (PV && score > alpha && score < beta) score = -search<true>(ply + 1, nd, -beta, -alpha);
        }
        if (stopped_) return 0;
        if (root) root_move_nodes_[m.from()][m.to()] += nodes_ - before;

        if (quiet && nq < 64) quiets[nq++] = m;
        else if (!quiet && nc < 32) caps[nc++] = m;
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
                    if (opt.capt_hist) {     // reward the refuting capture, penalise captures tried before it
                        int cb = opt.hist_bonus2 ? std::min(170 * depth - 100, 1700) : depth * depth;
                        auto upd = [&](Move c, int b) {
                            int16_t& h = capt_entry(pos, c);
                            b = std::clamp(b, -2000, 2000);
                            h += b - h * std::abs(b) / 16384;
                        };
                        if (!quiet) upd(m, cb);
                        for (int k = 0; k < nc; ++k) if (caps[k] != m) upd(caps[k], -cb);
                    }
                    if (quiet && opt.killers_history) {
                        if (killers_[ply][0] != m) { killers_[ply][1] = killers_[ply][0]; killers_[ply][0] = m; }
                        int bonus = opt.hist_bonus2 ? std::min(170 * depth - 100, 1700) : depth * depth;
                        update_history(pos.stm, m, bonus);
                        for (int k = 0; k < nq - 1; ++k) update_history(pos.stm, quiets[k], -bonus);
                        if (opt.cont_hist) {
                            update_cont(ply, pos.board[m.from()], m.to(), bonus);
                            for (int k = 0; k < nq - 1; ++k) update_cont(ply, pos.board[quiets[k].from()], quiets[k].to(), -bonus);
                        }
                    }
                    break;
                }
            }
        }
    }

    if (legal == 0) return excl ? alpha : in_check ? -MATE + ply : draw_score(ply);
    if (excl) return best;                    // singular verification search: no TT store, no history update

    // Correction history: move this pawn structure's correction toward (search result - static eval).
    if (opt.corr_hist && !in_check && (!best_move || !best_move.is_capture()) && std::abs(best) < DECISIVE &&
        !(best >= beta && best <= eval) && !(best <= orig_alpha && best >= eval)) {
        int& c = corr_[pos.stm][pawn_index(pos)];
        int w = std::min(depth + 1, 16);
        int diff = std::clamp(best - base_eval, -400, 400) * 256;
        c = std::clamp((c * (256 - w) + diff * w) / 256, -64 * 256, 64 * 256);
    }

    int stored = best;
    if (stored >= DECISIVE) stored += ply; else if (stored <= -DECISIVE) stored -= ply;
    Bound b = best >= beta ? BOUND_LOWER : (PV && best > orig_alpha) ? BOUND_EXACT : BOUND_UPPER;
    tt.store(pos.key, best_move, stored, raw_eval, depth, b);
    return best;
}

void Searcher::print_info(int depth, int score, const char* bound) {
    int64_t ms = std::max<int64_t>(elapsed_ms(), 1);
    std::string sc;
    if (std::abs(score) >= MATE_IN_MAX) {
        int n = (MATE - std::abs(score) + 1) / 2;
        sc = "mate " + std::to_string(score > 0 ? n : -n);
    } else sc = "cp " + std::to_string(score);
    sc += bound;
    std::string pv;
    for (int i = 0; i < pv_len_[0]; ++i) pv += " " + stack_[0].move_to_uci(pv_[0][i], opt.chess960);
    int wdl[3];
    if (opt.show_wdl && leaf_wdl(wdl)) sc += " wdl " + std::to_string(wdl[0]) + " " + std::to_string(wdl[1]) + " " + std::to_string(wdl[2]);
    std::printf("info depth %d seldepth %d score %s nodes %llu nps %llu hashfull %d tbhits %llu time %lld pv%s\n", depth,
                seldepth_, sc.c_str(), (unsigned long long)nodes_, (unsigned long long)(nodes_ * 1000 / ms),
                tt.hashfull(), (unsigned long long)tb_hits_, (long long)ms, pv.c_str());
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
    std::memset(root_move_nodes_, 0, sizeof(root_move_nodes_));
    null_prev_[0] = false;
    tb_hits_ = 0;
    root_allowed_.count = 0;
    root_filtered_ = opt.d4_syzygy && TB::root_filter(root, root_allowed_);
    if (!lim.searchmoves.empty()) {   // UCI searchmoves, intersected with the tablebase filter when both apply
        MoveList keep;
        for (Move m : lim.searchmoves) {
            bool in_tb = !root_filtered_;
            for (Move a : root_allowed_) in_tb |= a == m;
            if (in_tb) keep.add(m);
        }
        if (!keep.size()) for (Move m : lim.searchmoves) keep.add(m);
        root_allowed_ = keep;
        root_filtered_ = true;
    }

    // D1/D2: draw weight for the root side. Scores and static evals depend on it (and on the root side when it is
    // not neutral), so the TT is cleared whenever that configuration or the net changes.
    root_stm_ = root.stm;
    double d = opt.d1_utility ? opt.draw_value / 1000.0 : 0.5;
    if (opt.d2_defense && net_ && net_->wdl()) {
        double p_loss = root.stm == last_color_ ? last_wdl_[2] : last_wdl_[0];
        if (p_loss * 1000.0 >= opt.defense_loss_threshold) d = opt.defense_draw_value / 1000.0;
    }
    d_ = std::clamp(d, 0.001, 0.999);
    draw_us_ = int(std::lround(4.0 * opt.wdl_scale * std::log(d_ / (1.0 - d_))));
    int dm = int(std::lround(d_ * 1000));
    uint64_t tag = (net_gen_ * 4096 + uint64_t(dm) * 2 + (dm != 500 ? uint64_t(root.stm) : 0)) * 1024 + uint64_t(opt.wdl_scale);
    if (tag != util_tag_) { if (util_tag_ != ~0ULL) tt.clear(); util_tag_ = tag; }
    if (net_ && net_->quantized) NNUE::refresh(*net_, root, acc_[0], finny_.get());
    else acc_[0].computed = true;

    // D7: a fortress verdict stays on for a few moves unless the game makes progress (zeroing move at the root).
    if (!opt.d7_fortress || root.rule50 == 0) fortress_moves_ = 0;
    fortress_ = fortress_moves_ > 0;
    if (fortress_moves_ > 0) --fortress_moves_;
    int iter_scores[MAX_PLY];

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
    int score = 0, stable = 0;
    for (int d = 1; d <= max_depth; ++d) {
        root_depth_ = d;
        Move prev_best = best;
        // 1. Aspiration windows from depth 4
        int delta = 25, alpha = -INF, beta = INF;
        if (opt.aspiration && d >= 4 && std::abs(score) < DECISIVE) { alpha = score - delta; beta = score + delta; }
        int s;
        for (;;) {
            s = search<true>(0, d, alpha, beta);
            if (stopped_) break;
            if (s > alpha && pv_len_[0] > 0) best = pv_[0][0];   // fail-high or exact: PV is valid
            if (s <= alpha) { beta = (alpha + beta) / 2; alpha = std::max(s - delta, -INF); }
            else if (s >= beta) {
                // The fail-high move becomes the move we would play: report it now, so that if time runs out
                // during the re-search the last info line's PV starts with the bestmove we send.
                if (print && pv_len_[0] > 0 && best != prev_best) print_info(d, s, " lowerbound");
                beta = std::min(s + delta, INF);
            }
            else break;
            delta += delta / 2;
            if (delta > 600) { alpha = -INF; beta = INF; }
        }
        if (stopped_) break;
        score = s;
        iter_scores[d] = s;
        if (opt.d7_fortress && !fortress_ && d >= 10 && std::abs(s) >= 150 && std::abs(s) < DECISIVE && pv_len_[0] >= 8) {
            int lo = s, hi = s;
            for (int k = d - 5; k < d; ++k) { lo = std::min(lo, iter_scores[k]); hi = std::max(hi, iter_scores[k]); }
            bool zeroing = false;
            Position p = root;
            for (int k = 0; k < pv_len_[0] && !zeroing; ++k) {
                Move m = pv_[0][k];
                zeroing = m.is_capture() || type_of(p.board[m.from()]) == PAWN;
                p = p.make(m);
            }
            if (hi - lo <= 8 && !zeroing) { fortress_ = true; fortress_moves_ = 8; }
        }
        if (print) print_info(d, score);
        stable = (best == prev_best) ? stable + 1 : 0;
        if (soft_ms_) {
            double factor = 1.0;
            if (opt.tm_scaling && d >= 6) {
                uint64_t tot = 0;
                for (auto& r : root_move_nodes_) for (uint64_t v : r) tot += v;
                double frac = tot ? double(root_move_nodes_[best.from()][best.to()]) / double(tot) : 0.5;
                factor = std::clamp(1.8 - 1.4 * frac, 0.5, 1.5) * std::clamp(1.2 - 0.05 * stable, 0.8, 1.2);
            }
            if (elapsed_ms() >= soft_ms_ * factor) break;
        }
        if (!lim.infinite && score >= MATE_IN_MAX && MATE - score <= d) break;   // shortest mate already proven
    }
    int wdl[3];
    if (leaf_wdl(wdl)) {
        for (int k = 0; k < 3; ++k) last_wdl_[k] = wdl[k] / 1000.0;
        last_color_ = root.stm;
    }
    if (!best) {   // no completed iteration or no legal moves: fall back to any allowed legal move
        MoveList l; generate_legal(root, l);
        if (root_filtered_ && root_allowed_.size()) best = root_allowed_.moves[0];
        else if (l.size()) best = l.moves[0];
    }
    return best;
}

template int Searcher::search<true>(int, int, int, int);
template int Searcher::search<false>(int, int, int, int);
