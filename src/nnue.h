#pragma once
#include "position.h"
#include <string>
#include <vector>

// N1 / N1-WDL inference (768 -> L1 per perspective, SCReLU, concat -> head). File formats are written by nnue/export.py:
// ".nnuef" float32 (parity with PyTorch), ".nnue" int16/int32 quantised (QA = 255, QB = 64).
//
// Two quantised paths that must agree exactly (integer arithmetic):
//   * Net::forward: scalar full refresh, the reference used by verifynet;
//   * refresh/update/output: the search path, a per-ply accumulator stack updated incrementally from board diffs,
//     with AVX2 kernels when compiled for it (scalar fallback otherwise, or when the output weights are too large
//     for the int16 v*w trick).
namespace NNUE {

constexpr int FEATURES = 768;
constexpr int QA = 255, QB = 64;
constexpr int MAX_L1 = 1024;

struct Net {
    int l1 = 0, heads = 0, arch = 0;
    bool quantized = false;
    bool fast_out = false;                                         // |out_w| * QA fits int16 (SIMD v*w trick is exact)
    std::vector<float> ft_w, ft_b, out_w, out_b;                   // float
    std::vector<int16_t> q_ft_w, q_ft_b, q_out_w;                  // quantised
    std::vector<int32_t> q_out_b;
    std::string path;
    bool load(const std::string& path, std::string* err = nullptr);
    void finalize();      // derive fast_out after the weights are filled in (load calls it; tests that build nets too)
    bool wdl() const { return heads == 3; }
    // Raw head outputs (logits): 1 value for N1, 3 (W, D, L) for N1-WDL, from the side to move's view.
    void forward(const Position& pos, float* out) const;
};

// Accumulator for both absolute perspectives (v[WHITE], v[BLACK]); quantised nets only.
struct alignas(64) Accumulator {
    int16_t v[COLOR_NB][MAX_L1];
    bool computed = false;
};

void refresh(const Net& net, const Position& pos, Accumulator& acc);
// `to` = `from` plus the feature changes between parent and child (found by diffing the piece bitboards, so every
// move type, null moves and Chess960 castling are handled the same way).
void update(const Net& net, const Position& parent, const Position& child, const Accumulator& from, Accumulator& to);
// Head outputs (logits) for side to move `stm`. `simd` = false forces the scalar kernel (tests).
void output(const Net& net, const Accumulator& acc, Color stm, float* out, bool simd = true);

// Feature index for a piece seen from `persp`: (rel_color * 6 + type) * 64 + (sq ^ (persp == BLACK ? 56 : 0)).
inline int feature(Color persp, Piece p, int sq) {
    int rel = color_of(p) == persp ? 0 : 1;
    return (rel * 6 + type_of(p)) * 64 + (persp == BLACK ? sq ^ 56 : sq);
}

// W/D/L probabilities from head logits (softmax). For a scalar head, D = 0 and W = sigmoid(out).
void wdl_probs(const float* logits, int heads, double* wdl);

// Compare a net against reference outputs (jsonl with "fen" and "out"); prints the max/mean abs error.
// Returns the max abs error, or a negative number on failure.
double verify(const Net& net, const std::string& ref_jsonl, size_t limit = 0);

}  // namespace NNUE
