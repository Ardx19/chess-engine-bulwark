// UCI front end.
#include "bench.h"
#include "eval.h"
#include "nnue.h"
#include "perft.h"
#include "tablebase.h"
#include <algorithm>
#include <cstdio>
#include <iostream>
#include <memory>
#include <sstream>
#include <thread>

#ifdef BULWARK_EMBEDDED_NET
extern "C" const char bulwark_net_data[];
extern "C" const char bulwark_net_end[];
#endif

namespace {

struct Engine {
    Searcher searcher;
    Position pos;
    std::vector<U64> history;   // keys of positions before `pos`
    std::thread worker;
    std::unique_ptr<NNUE::Net> net;   // null = handcrafted eval

    // Load the evaluation net. "<default>" = the net embedded at build time (handcrafted eval if there is none);
    // "none", "<empty>" or "" = handcrafted eval. On failure the previous evaluation stays in use.
    bool load_net(const std::string& path, bool quiet = false) {
        if (path.empty() || path == "<empty>" || path == "none") { searcher.set_net(nullptr); net.reset(); return true; }
        auto n = std::make_unique<NNUE::Net>();
        std::string err;
        bool ok;
        if (path == "<default>") {
#ifdef BULWARK_EMBEDDED_NET
            ok = n->load_bytes(bulwark_net_data, size_t(bulwark_net_end - bulwark_net_data), BULWARK_EMBEDDED_NET, &err);
#else
            searcher.set_net(nullptr); net.reset(); return true;
#endif
        } else {
            ok = n->load(path, &err);
        }
        if (!ok) { std::printf("info string EvalFile %s: %s\n", path.c_str(), err.c_str()); return false; }
        if (quiet) { searcher.set_net(n.get()); net = std::move(n); return true; }
        std::printf("info string EvalFile %s: arch %d, L1 %d, %s head, %s%s\n", path.c_str(), n->arch, n->l1,
                    n->wdl() ? "W/D/L" : "scalar", n->quantized ? "quantised" : "float (slow, parity only)",
                    n->quantized && !n->fast_out ? ", scalar output kernel" : "");
        searcher.set_net(n.get());
        net = std::move(n);
        return true;
    }

    Engine() { pos.set_fen(Position::START_FEN); load_net("<default>", true); }
    void join() { if (worker.joinable()) worker.join(); }

    void set_position(std::istringstream& is) {
        std::string tok, fen;
        is >> tok;
        if (tok == "startpos") { fen = Position::START_FEN; is >> tok; }
        else if (tok == "fen") {
            while (is >> tok && tok != "moves") fen += tok + " ";
        }
        Position p;
        if (!p.set_fen(fen)) return;
        std::vector<U64> hist;
        if (tok == "moves") {
            std::string mv;
            while (is >> mv) {
                Move m = p.parse_move(mv);
                if (!m) break;
                hist.push_back(p.key);
                p = p.make(m);
                if (p.rule50 == 0) hist.clear();   // nothing before an irreversible move can repeat
            }
        }
        pos = p;
        history = hist;
    }

    void go(std::istringstream& is) {
        Limits lim;
        std::string tok;
        while (is >> tok) {
            if (tok == "wtime") { is >> lim.time[WHITE]; lim.use_clock = true; }
            else if (tok == "btime") { is >> lim.time[BLACK]; lim.use_clock = true; }
            else if (tok == "winc") is >> lim.inc[WHITE];
            else if (tok == "binc") is >> lim.inc[BLACK];
            else if (tok == "movestogo") is >> lim.movestogo;
            else if (tok == "movetime") is >> lim.movetime;
            else if (tok == "depth") is >> lim.depth;
            else if (tok == "nodes") is >> lim.nodes;
            else if (tok == "infinite") lim.infinite = true;
            else if (tok == "searchmoves") {
                MoveList legal; generate_legal(pos, legal);
                std::streampos at = is.tellg();
                while (is >> tok) {
                    Move m = pos.parse_move(tok);
                    bool ok = false;
                    for (Move x : legal) ok |= x == m;
                    if (!m || !ok) { is.clear(); is.seekg(at); break; }   // next keyword
                    lim.searchmoves.push_back(m);
                    at = is.tellg();
                }
            }
            else if (tok == "perft") { int d; is >> d; perft_divide(pos, d, searcher.opt.chess960); return; }
        }
        join();
        searcher.stop = false;
        Position root = pos;
        std::vector<U64> hist = history;
        worker = std::thread([this, root, hist, lim]() {
            Move best = searcher.think(root, hist, lim);
            std::printf("bestmove %s\n", root.move_to_uci(best, searcher.opt.chess960).c_str());
            std::fflush(stdout);
        });
    }

    void setoption(std::istringstream& is) {
        std::string tok, name, value;
        is >> tok;   // "name"
        while (is >> tok && tok != "value") name += (name.empty() ? "" : " ") + tok;
        while (is >> tok) value += (value.empty() ? "" : " ") + tok;
        if (name == "Hash") searcher.tt.resize(std::max(1, std::stoi(value)));
        else if (name == "Move Overhead") searcher.opt.move_overhead = std::stoi(value);
        else if (name == "UCI_Chess960") searcher.opt.chess960 = value == "true";
        else if (name == "SyzygyPath") { searcher.opt.syzygy_path = value; TB::init(value); }
        else if (name == "EvalFile") load_net(value);
        else if (name == "DrawValue") searcher.opt.draw_value = std::clamp(std::stoi(value), 0, 1000);
        else if (name == "DefenseDrawValue") searcher.opt.defense_draw_value = std::clamp(std::stoi(value), 0, 1000);
        else if (name == "WDLScale") searcher.opt.wdl_scale = std::clamp(std::stoi(value), 10, 400);
        else if (name == "DefenseLossThreshold") searcher.opt.defense_loss_threshold = std::clamp(std::stoi(value), 0, 1000);
        else if (bool* f = searcher.opt.flag(name)) *f = value == "true";
        // Threads is fixed at 1 (accepted and ignored).
    }

    void loop() {
        std::string line;
        while (std::getline(std::cin, line)) {
            std::istringstream is(line);
            std::string cmd;
            is >> cmd;
            if (cmd == "uci") {
                std::puts("id name Bulwark 0.3\nid author Aryan");
                std::puts("option name Hash type spin default 16 min 1 max 65536");
                std::puts("option name Threads type spin default 1 min 1 max 1");
                std::puts("option name Move Overhead type spin default 10 min 0 max 5000");
                std::puts("option name UCI_Chess960 type check default false");
                std::puts("option name CaptHist type check default true");
                std::puts("option name HistBonus2 type check default true");
                std::puts("option name HistPrune type check default false");
                for (const char* n : {"Aspiration", "RFP", "NMP", "LMR", "KillersHistory", "LMPFutility", "IIR", "Improving",
                                      "SEEPrune", "ContHist", "TMScaling", "D3_Rule50Scale", "D4_Syzygy", "D5_StalemateGuard"})
                    std::printf("option name %s type check default true\n", n);
                for (const char* n : {"D1_WDLUtility", "DefenseMode", "D6_PerpetualSeek", "D7_FortressDetect", "UCI_ShowWDL"})
                    std::printf("option name %s type check default false\n", n);
                std::puts("option name DrawValue type spin default 500 min 0 max 1000");
                std::puts("option name DefenseDrawValue type spin default 650 min 0 max 1000");
                std::puts("option name DefenseLossThreshold type spin default 400 min 0 max 1000");
                std::puts("option name WDLScale type spin default 100 min 10 max 400");
                std::puts("option name SyzygyPath type string default <empty>");
                std::puts("option name EvalFile type string default <default>");
                std::puts("uciok");
            } else if (cmd == "isready") std::puts("readyok");
            else if (cmd == "ucinewgame") { join(); searcher.new_game(); }
            else if (cmd == "setoption") { join(); setoption(is); }
            else if (cmd == "position") { join(); set_position(is); }
            else if (cmd == "go") go(is);
            else if (cmd == "stop") { searcher.stop = true; join(); }
            else if (cmd == "quit") { searcher.stop = true; join(); return; }
            else if (cmd == "bench") { int d = 6; is >> d; join(); run_bench(searcher, d, true); }
            else if (cmd == "perft") { int d = 1; is >> d; perft_divide(pos, d, searcher.opt.chess960); }
            else if (cmd == "verifynet") {   // verifynet <net.nnue|net.nnuef> <ref.jsonl> [limit]
                std::string net_path, ref_path; size_t lim = 0;
                is >> net_path >> ref_path >> lim;
                NNUE::Net net; std::string err;
                if (!net.load(net_path, &err)) std::printf("verifynet: %s\n", err.c_str());
                else NNUE::verify(net, ref_path, lim);
            }
            else if (cmd == "eval") {
                join();
                std::printf("hce %d (side to move)\n", Eval::evaluate(pos));
                if (net) {
                    float out[NNUE::MAX_HEADS]; double p[3];
                    net->forward(pos, out);
                    NNUE::wdl_probs(out, net->heads, p);
                    std::printf("nnue %d (side to move, neutral) logits", searcher.static_eval(pos));
                    for (int h = 0; h < net->heads; ++h) std::printf(" %.4f", out[h]);
                    std::printf(" wdl %.3f %.3f %.3f\n", p[0], p[1], p[2]);
                }
            }
            else if (cmd == "d") std::printf("%s\nkey %016llx\n", pos.fen(searcher.opt.chess960).c_str(), (unsigned long long)pos.key);
            std::fflush(stdout);
        }
        searcher.stop = true;
        join();
    }
};

}  // namespace

int main(int argc, char** argv) {
    Bitboards::init();
    Zobrist::init();
    Engine e;
    if (argc > 1 && std::string(argv[1]) == "bench") {   // bench [depth] [evalfile|none|-] [verbose]
        if (argc > 3 && std::string(argv[3]) != "-" && !e.load_net(argv[3])) return 1;
        run_bench(e.searcher, argc > 2 ? std::atoi(argv[2]) : 6, argc > 4);
        return 0;
    }
    e.loop();
}
