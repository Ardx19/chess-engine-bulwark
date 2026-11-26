#pragma once
#include "position.h"
#include <algorithm>
#include <memory>
#include <string>
#include <vector>

// Net inference. File formats are written by nnue/export.py:
//   version 1: N1 / N1-WDL (arch 1 / 2): 768 -> L1 per perspective, SCReLU, concat -> head (1 scalar, 3 W/D/L logits,
//              or 4 = dual head [expected-score logit, W, D, L]).
//   version 3: N4 (arch 5): the v2 king-bucket inputs -> L1, SCReLU, concat -> one linear head per output bucket
//              (piece count), quantised exactly like v1 (int16 output weights, QB = 64).
//   version 2: N2 / N3 (arch 3 / 4): 8 mirrored king buckets x 768 -> L1, SCReLU, concat -> K expert stacks
//              (2*L1 -> L2 -> L3 -> head) chosen by no router (K = 1), piece count, or a learned top-1 router.
// ".nnuef" is float32 (parity with PyTorch), ".nnue" quantised: L1 int16 (QA = 255); v1 output int16 (QB = 64);
// v2 L2 / router int8 (QB = 64) on activations x_q = round(clamp(acc, 0, QA)^2 / 512) in 0..127, L3 and head float.
//
// Two quantised paths that must agree exactly (integer arithmetic, then identical float code):
//   * Net::forward: full refresh + scalar kernels, the reference used by verifynet;
//   * refresh/update/output: the search path, a per-ply accumulator stack updated from board diffs (v2: a king
//     bucket or mirror change refreshes that perspective through a Finny table), AVX2 kernels when available.
namespace NNUE {

constexpr int FEATURES = 768;
constexpr int QA = 255, QB = 64;
constexpr int MAX_L1 = 2048;
constexpr int N_BUCKETS = 8;
constexpr int N_LAYOUTS = 2 * N_BUCKETS;   // bucket x mirrored
constexpr int N_SCALARS = 4;
constexpr int MAX_HEADS = 4;              // output buffers passed to forward()/output() need this many floats
enum Router { ROUTER_NONE = 0, ROUTER_COUNT = 1, ROUTER_LEARNED = 2 };

struct Net {
    int version = 1, arch = 0, l1 = 0, heads = 0;
    int l2 = 0, l3 = 0, experts = 1, router = ROUTER_NONE;   // version 2
    int out_buckets = 1;                                       // version 3: output heads selected by piece count
    bool quantized = false;
    bool fast_out = false;                                     // v1: |out_w| * QA fits int16 (SIMD v*w trick is exact)
    // float (.nnuef)
    std::vector<float> ft_w, ft_b, out_w, out_b;               // v1 head: out_w[heads][2*L1]
    std::vector<float> f_l2_w, f_r_w;                          // v2 float L2 / router weights
    // quantised (.nnue)
    std::vector<int16_t> q_ft_w, q_ft_b, q_out_w;
    std::vector<int32_t> q_out_b;
    std::vector<int8_t> q_l2_w, q_r_w;                         // v2
    // v2, shared by both files (float in both)
    std::vector<float> l2_b, l3_w, l3_b, h_w, h_b, r_s, r_b;
    std::string path;

    bool load(const std::string& path, std::string* err = nullptr);
    bool load_bytes(const char* data, size_t size, const std::string& name, std::string* err = nullptr);
    void finalize();      // derived flags after the weights are filled in (load calls it; tests that build nets too)
    bool wdl() const { return heads == 3 || heads == 4; }      // has W/D/L outputs
    bool dual() const { return heads == 4; }                  // scalar search head + W/D/L head
    int inputs() const { return version >= 2 ? N_BUCKETS * FEATURES : FEATURES; }
    int out_bucket(const Position& pos) const {
        return std::min((popcount(pos.pieces()) - 1) * out_buckets / 32, out_buckets - 1);
    }
    // Raw head outputs (logits), side to move's view: 1 (scalar), 3 (W, D, L) or 4 (scalar, W, D, L).
    void forward(const Position& pos, float* out) const;
};

// Accumulator for both absolute perspectives (v[WHITE], v[BLACK]); quantised nets only.
struct alignas(64) Accumulator {
    int16_t v[COLOR_NB][MAX_L1];
    bool computed = false;
};

// Finny table (v2): per perspective and king layout, the last accumulator computed for that layout and the piece
// bitboards it reflects, so a bucket change costs a diff against that state instead of a full refresh.
struct Finny {
    struct Entry {
        alignas(64) int16_t acc[MAX_L1];
        Bitboard bb[12];
        bool valid = false;
    } e[COLOR_NB][N_LAYOUTS];
    void clear() { for (auto& c : e) for (auto& x : c) x.valid = false; }
};

// King layout of a perspective: bucket * 2 + mirrored (v2), always 0 for v1.
int king_layout(const Net& net, Color persp, int ksq);
// Feature index of a piece for a perspective with the given king layout.
int feature_index(const Net& net, Color persp, Piece p, int sq, int layout);

void refresh(const Net& net, const Position& pos, Accumulator& acc, Finny* finny = nullptr);
// `to` = `from` plus the feature changes between parent and child (found by diffing the piece bitboards, so every
// move type, null moves and Chess960 castling are handled the same way).
void update(const Net& net, const Position& parent, const Position& child, const Accumulator& from, Accumulator& to,
            Finny* finny = nullptr);
// Head outputs (logits) for the position the accumulator belongs to. `simd` = false forces the scalar kernels.
void output(const Net& net, const Accumulator& acc, const Position& pos, float* out, bool simd = true);
// Expert chosen for a position (v2), for routing statistics. -1 for v1.
int expert_of(const Net& net, const Accumulator& acc, const Position& pos);

// Feature index for v1 nets (no buckets): (rel_color * 6 + type) * 64 + (sq ^ (persp == BLACK ? 56 : 0)).
inline int feature(Color persp, Piece p, int sq) {
    int rel = color_of(p) == persp ? 0 : 1;
    return (rel * 6 + type_of(p)) * 64 + (persp == BLACK ? sq ^ 56 : sq);
}

// Router scalars: [phase, material balance / 10 (side to move), rule50 / 100, side to move is black].
void scalars(const Position& pos, float* sc);

// W/D/L probabilities from head logits (softmax over the W/D/L logits). Scalar head: D = 0, W = sigmoid(out).
void wdl_probs(const float* logits, int heads, double* wdl);

// Compare a net against reference outputs (jsonl with "fen" and "out"); prints the max/mean abs error.
// Returns the max abs error, or a negative number on failure.
double verify(const Net& net, const std::string& ref_jsonl, size_t limit = 0);

}  // namespace NNUE
