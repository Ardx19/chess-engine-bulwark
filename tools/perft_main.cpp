// Usage: perft <depth> [fen...]   |  perft divide <depth> [fen...]   (append "960" as env CHESS960=1)
#include "perft.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

int main(int argc, char** argv) {
    bool divide = argc > 1 && std::string(argv[1]) == "divide";
    int a = divide ? 2 : 1;
    if (argc <= a) { std::puts("usage: perft [divide] <depth> [fen]"); return 1; }
    int depth = std::atoi(argv[a]);
    std::string fen = argc > a + 1 ? argv[a + 1] : Position::START_FEN;
    for (int i = a + 2; i < argc; ++i) fen += std::string(" ") + argv[i];
    Position pos;
    if (!pos.set_fen(fen)) { std::puts("bad fen"); return 1; }
    bool c960 = std::getenv("CHESS960") != nullptr;
    if (divide) { perft_divide(pos, depth, c960); return 0; }
    auto t0 = std::chrono::steady_clock::now();
    U64 n = perft(pos, depth);
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("nodes %llu  time %.3fs  %.1f Mnps\n", (unsigned long long)n, s, n / s / 1e6);
}
