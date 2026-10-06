// UCI front end.
#include "bench.h"
#include "eval.h"
#include "perft.h"
#include <cstdio>
#include <iostream>
#include <sstream>
#include <thread>

namespace {

struct Engine {
    Searcher searcher;
    Position pos;
    std::vector<U64> history;   // keys of positions before `pos`
    std::thread worker;

    Engine() { pos.set_fen(Position::START_FEN); }
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
        // Threads, SyzygyPath, EvalFile, D* switches: accepted, wired up in later phases.
    }

    void loop() {
        std::string line;
        while (std::getline(std::cin, line)) {
            std::istringstream is(line);
            std::string cmd;
            is >> cmd;
            if (cmd == "uci") {
                std::puts("id name Bulwark 0.2\nid author Aryan");
                std::puts("option name Hash type spin default 16 min 1 max 65536");
                std::puts("option name Threads type spin default 1 min 1 max 1");
                std::puts("option name Move Overhead type spin default 10 min 0 max 5000");
                std::puts("option name UCI_Chess960 type check default false");
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
            else if (cmd == "eval") std::printf("eval %d (side to move)\n", Eval::evaluate(pos));
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
    if (argc > 1 && std::string(argv[1]) == "bench") {
        run_bench(e.searcher, argc > 2 ? std::atoi(argv[2]) : 6, argc > 3);
        return 0;
    }
    e.loop();
}
