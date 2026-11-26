#include "nnue.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace NNUE {

namespace {
const int BUCKETS[32] = {0, 1, 2, 3,  4, 4, 5, 5,  6, 6, 6, 6,  6, 6, 6, 6,
                         7, 7, 7, 7,  7, 7, 7, 7,  7, 7, 7, 7,  7, 7, 7, 7};
const float PHASE_W[6] = {0, 1, 1, 2, 4, 0};
const float MAT_W[6] = {1, 3, 3, 5, 9, 0};
constexpr float DEQ = 1.0f / (127.0f * QB);   // v2: x_q in 0..127, weights * QB
}  // namespace

bool Net::load(const std::string& file, std::string* err) {
    std::ifstream f(file, std::ios::binary);
    if (!f) { if (err) *err = "cannot open file"; return false; }
    std::vector<char> data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return load_bytes(data.data(), data.size(), file, err);
}

bool Net::load_bytes(const char* data, size_t size, const std::string& name, std::string* err) {
    auto fail = [&](const char* m) { if (err) *err = m; return false; };
    if (size < 12 + 32) return fail("file too small");
    quantized = std::memcmp(data, "NNUEWDL1", 8) == 0;
    bool flt = std::memcmp(data, "NNUEFLT1", 8) == 0;
    if (!quantized && !flt) return fail("bad magic");
    size_t at = 8;
    bool ok = true;
    auto take = [&](void* dst, size_t bytes) {
        if (!ok || at + bytes > size) { ok = false; return; }
        std::memcpy(dst, data + at, bytes);
        at += bytes;
    };
    auto vec = [&](auto& v, size_t n) { v.resize(n); take(v.data(), n * sizeof(v[0])); };
    uint32_t ver = 0;
    take(&ver, 4);
    if (ver == 1) {
        uint32_t h[4];
        take(h, 16);
        arch = int(h[0]); l1 = int(h[1]); heads = int(h[3]);
        if (h[2] != 1 || l1 <= 0 || (heads != 1 && heads != 3 && heads != 4)) return fail("unsupported architecture");
        size_t n_ft = size_t(FEATURES) * l1, n_ow = size_t(heads) * 2 * l1;
        if (quantized) { vec(q_ft_w, n_ft); vec(q_ft_b, l1); vec(q_out_w, n_ow); vec(q_out_b, heads); }
        else { vec(ft_w, n_ft); vec(ft_b, l1); vec(out_w, n_ow); vec(out_b, heads); }
    } else if (ver == 2) {
        uint32_t h[8];
        take(h, 32);
        arch = int(h[0]); l1 = int(h[1]); heads = int(h[3]); l2 = int(h[4]); l3 = int(h[5]); experts = int(h[6]); router = int(h[7]);
        if (h[2] != N_BUCKETS || l1 <= 0 || (heads != 1 && heads != 3) || l2 <= 0 || l2 > 64 || l3 <= 0 || l3 > 64 ||
            experts < 1 || experts > 16 || router < 0 || router > 2 || (experts == 1) != (router == ROUTER_NONE))
            return fail("unsupported architecture");
        size_t n_ft = size_t(N_BUCKETS) * FEATURES * l1, n_l2 = size_t(experts) * l2 * 2 * l1;
        if (quantized) { vec(q_ft_w, n_ft); vec(q_ft_b, l1); vec(q_l2_w, n_l2); }
        else { vec(ft_w, n_ft); vec(ft_b, l1); vec(f_l2_w, n_l2); }
        vec(l2_b, size_t(experts) * l2);
        vec(l3_w, size_t(experts) * l3 * l2); vec(l3_b, size_t(experts) * l3);
        vec(h_w, size_t(experts) * heads * l3); vec(h_b, size_t(experts) * heads);
        if (router == ROUTER_LEARNED) {
            if (quantized) vec(q_r_w, size_t(experts) * 2 * l1); else vec(f_r_w, size_t(experts) * 2 * l1);
            vec(r_s, size_t(experts) * N_SCALARS); vec(r_b, experts);
        }
    } else {
        return fail("unsupported version");
    }
    if (!ok) return fail("file too small");
    if (at + 32 != size) return fail("size mismatch");
    if (quantized && l1 > MAX_L1) return fail("L1 wider than MAX_L1");
    version = int(ver);
    path = name;
    finalize();
    return true;
}

void Net::finalize() {
    fast_out = quantized && version == 1 && l1 <= MAX_L1;
    for (int16_t w : q_out_w)
        if (std::abs(int(w)) * QA > 32767) { fast_out = false; break; }
}

int king_layout(const Net& net, Color persp, int ksq) {
    if (net.version != 2) return 0;
    int k = persp == BLACK ? ksq ^ 56 : ksq;
    bool mirror = (k & 7) >= 4;
    int f = mirror ? 7 - (k & 7) : (k & 7);
    return BUCKETS[(k >> 3) * 4 + f] * 2 + int(mirror);
}

int feature_index(const Net& net, Color persp, Piece p, int sq, int layout) {
    if (net.version != 2) return feature(persp, p, sq);
    int s = persp == BLACK ? sq ^ 56 : sq;
    if (layout & 1) s ^= 7;
    int rel = color_of(p) == persp ? 0 : 1;
    return (layout >> 1) * FEATURES + (rel * 6 + type_of(p)) * 64 + s;
}

void scalars(const Position& pos, float* sc) {
    float phase = 0, bal = 0;
    for (int c = 0; c < 2; ++c)
        for (int pt = PAWN; pt <= KING; ++pt) {
            int n = popcount(pos.pieces(Color(c), PieceType(pt)));
            phase += PHASE_W[pt] * n;
            bal += (c == pos.stm ? 1.0f : -1.0f) * MAT_W[pt] * n;
        }
    sc[0] = std::min(phase, 24.0f) / 24.0f;
    sc[1] = bal / 10.0f;
    sc[2] = float(pos.rule50) / 100.0f;
    sc[3] = pos.stm == BLACK ? 1.0f : 0.0f;
}

// ---------------------------------------------------------------------------------------------------------------
// Kernels

static int64_t screlu_dot_scalar(const int16_t* acc, const int16_t* w, int n) {
    int64_t sum = 0;
    for (int i = 0; i < n; ++i) {
        int64_t x = std::min(std::max(int(acc[i]), 0), QA);
        sum += x * x * w[i];
    }
    return sum;
}

static int32_t dot_u8i8_scalar(const uint8_t* x, const int8_t* w, int n) {
    int32_t s = 0;
    for (int i = 0; i < n; ++i) s += int32_t(x[i]) * int32_t(w[i]);
    return s;
}

#ifdef __AVX2__
// sum clamp(acc, 0, QA)^2 * w. Requires |w| * QA <= 32767 so that v*w fits int16 (Net::fast_out) and n % 16 == 0.
// madd(v*w, v) is at most 2 * 255 * 32767 per int32 lane, so the lanes are flushed to int64 every 64 steps.
static int64_t screlu_dot_avx2(const int16_t* acc, const int16_t* w, int n) {
    const __m256i zero = _mm256_setzero_si256(), qa = _mm256_set1_epi16(QA);
    __m256i s32 = zero, s64 = zero;
    auto flush = [&]() {
        s64 = _mm256_add_epi64(s64, _mm256_cvtepi32_epi64(_mm256_castsi256_si128(s32)));
        s64 = _mm256_add_epi64(s64, _mm256_cvtepi32_epi64(_mm256_extracti128_si256(s32, 1)));
        s32 = zero;
    };
    for (int i = 0, k = 1; i < n; i += 16, ++k) {
        __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + i));
        v = _mm256_min_epi16(_mm256_max_epi16(v, zero), qa);
        __m256i ww = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + i));
        s32 = _mm256_add_epi32(s32, _mm256_madd_epi16(_mm256_mullo_epi16(v, ww), v));
        if ((k & 63) == 0) flush();
    }
    flush();
    alignas(32) int64_t lanes[4];
    _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), s64);
    return lanes[0] + lanes[1] + lanes[2] + lanes[3];
}

// u8 (0..127) x i8 (|w| <= 127): maddubs pairs are at most 32258, so nothing saturates; n % 32 == 0.
static int32_t dot_u8i8_avx2(const uint8_t* x, const int8_t* w, int n) {
    const __m256i ones = _mm256_set1_epi16(1);
    __m256i acc = _mm256_setzero_si256();
    for (int i = 0; i < n; i += 32) {
        __m256i p = _mm256_maddubs_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + i)),
                                         _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + i)));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(p, ones));
    }
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
    return _mm_cvtsi128_si32(s);
}
#endif

static int32_t dot_u8i8(const uint8_t* x, const int8_t* w, int n, bool simd) {
#ifdef __AVX2__
    if (simd && n % 32 == 0) return dot_u8i8_avx2(x, w, n);
#else
    (void)simd;
#endif
    return dot_u8i8_scalar(x, w, n);
}

// v2 after L2: clamp -> L3 -> clamp -> head (float, identical code for every path), scaled by the router gate.
static void tail_v2(const Net& net, int e, const float* z2, float gate, float* out) {
    float h2[64], h3[64];
    for (int j = 0; j < net.l2; ++j) h2[j] = std::min(std::max(z2[j], 0.0f), 1.0f);
    for (int j = 0; j < net.l3; ++j) {
        const float* w = &net.l3_w[(size_t(e) * net.l3 + j) * net.l2];
        float s = net.l3_b[size_t(e) * net.l3 + j];
        for (int i = 0; i < net.l2; ++i) s += w[i] * h2[i];
        h3[j] = std::min(std::max(s, 0.0f), 1.0f);
    }
    for (int h = 0; h < net.heads; ++h) {
        const float* w = &net.h_w[(size_t(e) * net.heads + h) * net.l3];
        float s = net.h_b[size_t(e) * net.heads + h];
        for (int i = 0; i < net.l3; ++i) s += w[i] * h3[i];
        out[h] = s * gate;
    }
}

static int count_expert(const Net& net, const Position& pos) {
    return std::min((popcount(pos.pieces()) - 1) * net.experts / 32, net.experts - 1);
}

static int softmax_argmax(const float* logits, int k, float* gate) {
    float m = logits[0];
    for (int i = 1; i < k; ++i) m = std::max(m, logits[i]);
    float e[16], t = 0;
    int best = 0;
    for (int i = 0; i < k; ++i) { e[i] = std::exp(logits[i] - m); t += e[i]; if (logits[i] > logits[best]) best = i; }
    *gate = e[best] / t;
    return best;
}

// Quantised v2 activations: x_q = round(clamp(acc, 0, QA)^2 / 512) in 0..127, stm perspective first.
static void activations_v2(const Net& net, const Accumulator& acc, Color stm, uint8_t* x) {
    const int n = net.l1;
    const int16_t* persp[2] = {acc.v[stm], acc.v[~stm]};
    for (int v = 0; v < 2; ++v)
        for (int i = 0; i < n; ++i) {
            int a = std::min(std::max(int(persp[v][i]), 0), QA);
            x[v * n + i] = uint8_t((a * a + 256) >> 9);
        }
}

static int select_expert_q(const Net& net, const uint8_t* x, const Position& pos, float* gate, bool simd) {
    *gate = 1.0f;
    if (net.router == ROUTER_NONE) return 0;
    if (net.router == ROUTER_COUNT) return count_expert(net, pos);
    float sc[N_SCALARS], logits[16];
    scalars(pos, sc);
    for (int k = 0; k < net.experts; ++k) {
        float s = float(dot_u8i8(x, &net.q_r_w[size_t(k) * 2 * net.l1], 2 * net.l1, simd)) * DEQ + net.r_b[k];
        for (int j = 0; j < N_SCALARS; ++j) s += sc[j] * net.r_s[size_t(k) * N_SCALARS + j];
        logits[k] = s;
    }
    return softmax_argmax(logits, net.experts, gate);
}

void Net::forward(const Position& pos, float* out) const {
    if (quantized && version == 2) {
        auto acc = std::make_unique<Accumulator>();
        refresh(*this, pos, *acc);
        output(*this, *acc, pos, out, false);
        return;
    }
    const Color persp[2] = {pos.stm, ~pos.stm};
    if (!quantized && version == 2) {
        std::vector<float> x(size_t(2) * l1);
        for (int v = 0; v < 2; ++v) {
            std::vector<float> a(ft_b.begin(), ft_b.end());
            int layout = king_layout(*this, persp[v], pos.king_sq(persp[v]));
            Bitboard occ = pos.pieces();
            while (occ) {
                int s = pop_lsb(occ);
                const float* w = &ft_w[size_t(feature_index(*this, persp[v], pos.board[s], s, layout)) * l1];
                for (int i = 0; i < l1; ++i) a[i] += w[i];
            }
            for (int i = 0; i < l1; ++i) { float c = std::min(std::max(a[i], 0.0f), 1.0f); x[size_t(v) * l1 + i] = c * c; }
        }
        int e = 0;
        float gate = 1.0f;
        if (router == ROUTER_COUNT) e = count_expert(*this, pos);
        else if (router == ROUTER_LEARNED) {
            float sc[N_SCALARS], logits[16];
            scalars(pos, sc);
            for (int k = 0; k < experts; ++k) {
                float s = r_b[k];
                for (int i = 0; i < 2 * l1; ++i) s += x[i] * f_r_w[size_t(k) * 2 * l1 + i];
                for (int j = 0; j < N_SCALARS; ++j) s += sc[j] * r_s[size_t(k) * N_SCALARS + j];
                logits[k] = s;
            }
            e = softmax_argmax(logits, experts, &gate);
        }
        float z2[64];
        for (int j = 0; j < l2; ++j) {
            const float* w = &f_l2_w[(size_t(e) * l2 + j) * 2 * l1];
            float s = l2_b[size_t(e) * l2 + j];
            for (int i = 0; i < 2 * l1; ++i) s += x[i] * w[i];
            z2[j] = s;
        }
        tail_v2(*this, e, z2, gate, out);
        return;
    }
    if (!quantized) {
        std::vector<float> acc[2];
        for (int v = 0; v < 2; ++v) {
            acc[v].assign(ft_b.begin(), ft_b.end());
            Bitboard occ = pos.pieces();
            while (occ) {
                int s = pop_lsb(occ);
                const float* w = &ft_w[size_t(feature(persp[v], pos.board[s], s)) * l1];
                for (int i = 0; i < l1; ++i) acc[v][i] += w[i];
            }
        }
        for (int h = 0; h < heads; ++h) {
            float sum = out_b[h];
            for (int v = 0; v < 2; ++v)
                for (int i = 0; i < l1; ++i) {
                    float x = std::min(std::max(acc[v][i], 0.0f), 1.0f);
                    sum += x * x * out_w[size_t(h) * 2 * l1 + v * l1 + i];
                }
            out[h] = sum;
        }
        return;
    }
    // v1 quantised reference: int32 accumulators, independent of the int16 search path.
    std::vector<int32_t> acc[2];
    for (int v = 0; v < 2; ++v) {
        acc[v].assign(q_ft_b.begin(), q_ft_b.end());
        Bitboard occ = pos.pieces();
        while (occ) {
            int s = pop_lsb(occ);
            const int16_t* w = &q_ft_w[size_t(feature(persp[v], pos.board[s], s)) * l1];
            for (int i = 0; i < l1; ++i) acc[v][i] += w[i];
        }
    }
    for (int h = 0; h < heads; ++h) {
        int64_t sum = 0;
        for (int v = 0; v < 2; ++v)
            for (int i = 0; i < l1; ++i) {
                int64_t x = std::min(std::max(acc[v][i], 0), QA);
                sum += x * x * q_out_w[size_t(h) * 2 * l1 + v * l1 + i];
            }
        out[h] = float((double(sum) + double(q_out_b[h])) / (double(QA) * QA * QB));
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Incremental quantised path

static void refresh_persp(const Net& net, const Position& pos, Color c, int16_t* dst, Finny* finny) {
    const int n = net.l1;
    const int layout = king_layout(net, c, pos.king_sq(c));
    auto add = [&](int16_t* a, Piece p, int sq, int sign) {
        const int16_t* w = &net.q_ft_w[size_t(feature_index(net, c, p, sq, layout)) * n];
        if (sign > 0) for (int i = 0; i < n; ++i) a[i] = int16_t(a[i] + w[i]);
        else for (int i = 0; i < n; ++i) a[i] = int16_t(a[i] - w[i]);
    };
    if (finny && net.version == 2) {
        Finny::Entry& e = finny->e[c][layout];
        if (!e.valid) {
            std::memcpy(e.acc, net.q_ft_b.data(), sizeof(int16_t) * n);
            std::memset(e.bb, 0, sizeof(e.bb));
            e.valid = true;
        }
        for (int p = 0; p < 12; ++p) {
            Bitboard now = pos.pieces(Color(p / 6), PieceType(p % 6));
            Bitboard x = e.bb[p] ^ now;
            for (Bitboard b = x & now; b;) add(e.acc, Piece(p), pop_lsb(b), +1);
            for (Bitboard b = x & e.bb[p]; b;) add(e.acc, Piece(p), pop_lsb(b), -1);
            e.bb[p] = now;
        }
        std::memcpy(dst, e.acc, sizeof(int16_t) * n);
        return;
    }
    std::memcpy(dst, net.q_ft_b.data(), sizeof(int16_t) * n);
    Bitboard occ = pos.pieces();
    while (occ) {
        int s = pop_lsb(occ);
        add(dst, pos.board[s], s, +1);
    }
}

void refresh(const Net& net, const Position& pos, Accumulator& acc, Finny* finny) {
    for (int c = 0; c < 2; ++c) refresh_persp(net, pos, Color(c), acc.v[c], finny);
    acc.computed = true;
}

void update(const Net& net, const Position& parent, const Position& child, const Accumulator& from, Accumulator& to,
            Finny* finny) {
    const int n = net.l1;
    int add_pc[8], add_sq[8], sub_pc[8], sub_sq[8], na = 0, ns = 0;
    for (int c = 0; c < 2; ++c)
        for (int pt = PAWN; pt <= KING; ++pt) {
            Bitboard a = parent.pieces(Color(c), PieceType(pt)), b = child.pieces(Color(c), PieceType(pt));
            Bitboard x = a ^ b;
            if (!x) continue;
            Piece p = make_piece(Color(c), PieceType(pt));
            for (Bitboard r = x & a; r && ns < 8;) { sub_pc[ns] = p; sub_sq[ns++] = pop_lsb(r); }
            for (Bitboard r = x & b; r && na < 8;) { add_pc[na] = p; add_sq[na++] = pop_lsb(r); }
        }
    for (int c = 0; c < 2; ++c) {
        const int layout = king_layout(net, Color(c), child.king_sq(Color(c)));
        if (net.version == 2 && layout != king_layout(net, Color(c), parent.king_sq(Color(c)))) {
            refresh_persp(net, child, Color(c), to.v[c], finny);   // king changed bucket or side: features all move
            continue;
        }
        const int16_t* src = from.v[c];
        int16_t* dst = to.v[c];
        auto W = [&](int pc, int sq) { return &net.q_ft_w[size_t(feature_index(net, Color(c), Piece(pc), sq, layout)) * n]; };
        int ai = 0, si = 0;
        if (na && ns) {
            const int16_t *a = W(add_pc[0], add_sq[0]), *s = W(sub_pc[0], sub_sq[0]);
            for (int i = 0; i < n; ++i) dst[i] = int16_t(src[i] + a[i] - s[i]);
            ai = si = 1;
        } else {
            std::memcpy(dst, src, sizeof(int16_t) * n);
        }
        for (; ai < na; ++ai) { const int16_t* a = W(add_pc[ai], add_sq[ai]); for (int i = 0; i < n; ++i) dst[i] = int16_t(dst[i] + a[i]); }
        for (; si < ns; ++si) { const int16_t* s = W(sub_pc[si], sub_sq[si]); for (int i = 0; i < n; ++i) dst[i] = int16_t(dst[i] - s[i]); }
    }
    to.computed = true;
}

void output(const Net& net, const Accumulator& acc, const Position& pos, float* out, bool simd) {
    const int n = net.l1;
    const Color stm = pos.stm;
    if (net.version == 2) {
        alignas(64) uint8_t x[2 * MAX_L1];
        activations_v2(net, acc, stm, x);
        float gate;
        int e = select_expert_q(net, x, pos, &gate, simd);
        float z2[64];
        for (int j = 0; j < net.l2; ++j)
            z2[j] = float(dot_u8i8(x, &net.q_l2_w[(size_t(e) * net.l2 + j) * 2 * n], 2 * n, simd)) * DEQ +
                    net.l2_b[size_t(e) * net.l2 + j];
        tail_v2(net, e, z2, gate, out);
        return;
    }
    const int16_t* persp[2] = {acc.v[stm], acc.v[~stm]};
    for (int h = 0; h < net.heads; ++h) {
        int64_t sum = 0;
        for (int v = 0; v < 2; ++v) {
            const int16_t* w = &net.q_out_w[size_t(h) * 2 * n + size_t(v) * n];
#ifdef __AVX2__
            if (simd && net.fast_out && n % 16 == 0) { sum += screlu_dot_avx2(persp[v], w, n); continue; }
#endif
            sum += screlu_dot_scalar(persp[v], w, n);
        }
        out[h] = float((double(sum) + double(net.q_out_b[h])) / (double(QA) * QA * QB));
    }
}

int expert_of(const Net& net, const Accumulator& acc, const Position& pos) {
    if (net.version != 2) return -1;
    alignas(64) uint8_t x[2 * MAX_L1];
    activations_v2(net, acc, pos.stm, x);
    float gate;
    return select_expert_q(net, x, pos, &gate, true);
}

void wdl_probs(const float* z, int heads, double* p) {
    if (heads == 1) { p[0] = 1.0 / (1.0 + std::exp(-double(z[0]))); p[1] = 0; p[2] = 1 - p[0]; return; }
    if (heads == 4) z += 1;                       // dual head: [scalar, W, D, L]
    double m = std::max({double(z[0]), double(z[1]), double(z[2])});
    double e0 = std::exp(z[0] - m), e1 = std::exp(z[1] - m), e2 = std::exp(z[2] - m), t = e0 + e1 + e2;
    p[0] = e0 / t; p[1] = e1 / t; p[2] = e2 / t;
}

static bool parse_line(const std::string& line, std::string& fen, std::vector<float>& vals) {
    size_t a = line.find("\"fen\": \"");
    if (a == std::string::npos) return false;
    a += 8;
    size_t b = line.find('"', a);
    fen = line.substr(a, b - a);
    size_t o = line.find("\"out\": [", b);
    if (o == std::string::npos) return false;
    o += 8;
    size_t e = line.find(']', o);
    std::stringstream ss(line.substr(o, e - o));
    std::string tok;
    vals.clear();
    while (std::getline(ss, tok, ',')) vals.push_back(std::stof(tok));
    return true;
}

double verify(const Net& net, const std::string& ref_jsonl, size_t limit) {
    std::ifstream f(ref_jsonl);
    if (!f) { std::printf("cannot open %s\n", ref_jsonl.c_str()); return -1; }
    std::string line, fen;
    std::vector<float> ref;
    double max_err = 0, sum_err = 0;
    size_t n = 0;
    while (std::getline(f, line)) {
        if (limit && n >= limit) break;
        if (!parse_line(line, fen, ref) || int(ref.size()) != net.heads) { std::printf("bad reference line %zu\n", n); return -1; }
        Position pos;
        if (!pos.set_fen(fen)) { std::printf("bad fen %s\n", fen.c_str()); return -1; }
        float out[MAX_HEADS];
        net.forward(pos, out);
        for (int h = 0; h < net.heads; ++h) {
            double e = std::fabs(double(out[h]) - ref[h]);
            max_err = std::max(max_err, e);
            sum_err += e;
        }
        ++n;
    }
    std::printf("verifynet (%s): %zu positions, max abs error %.6g, mean abs error %.6g\n",
                net.quantized ? "quantized" : "float", n, max_err, n ? sum_err / (n * net.heads) : 0.0);
    return max_err;
}

}  // namespace NNUE
