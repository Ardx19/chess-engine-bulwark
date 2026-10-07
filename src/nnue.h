#pragma once
#include "position.h"
#include <string>
#include <vector>

// N1 / N1-WDL inference (768 -> L1 per perspective, SCReLU, concat -> head). File formats are written by nnue/export.py:
// ".nnuef" float32 (parity with PyTorch), ".nnue" int16/int32 quantised (QA = 255, QB = 64). Scalar code only here;
// the AVX2 path comes with Phase 5 and is verified against this one.
namespace NNUE {

constexpr int FEATURES = 768;
constexpr int QA = 255, QB = 64;

struct Net {
    int l1 = 0, heads = 0, arch = 0;
    bool quantized = false;
    std::vector<float> ft_w, ft_b, out_w, out_b;                   // float
    std::vector<int16_t> q_ft_w, q_ft_b, q_out_w;                  // quantised
    std::vector<int32_t> q_out_b;
    bool load(const std::string& path, std::string* err = nullptr);
    // Raw head outputs (logits): 1 value for N1, 3 (W, D, L) for N1-WDL, from the side to move's view.
    void forward(const Position& pos, float* out) const;
};

// Feature index for a piece seen from `persp`: (rel_color * 6 + type) * 64 + (sq ^ (persp == BLACK ? 56 : 0)).
inline int feature(Color persp, Piece p, int sq) {
    int rel = color_of(p) == persp ? 0 : 1;
    return (rel * 6 + type_of(p)) * 64 + (persp == BLACK ? sq ^ 56 : sq);
}

// Compare a net against reference outputs (jsonl with "fen" and "out"); prints the max/mean abs error.
// Returns the max abs error, or a negative number on failure.
double verify(const Net& net, const std::string& ref_jsonl, size_t limit = 0);

}  // namespace NNUE
