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

bool Net::load(const std::string& path, std::string* err) {
    auto fail = [&](const char* m) { if (err) *err = m; return false; };
    std::ifstream f(path, std::ios::binary);
    if (!f) return fail("cannot open file");
    std::vector<char> data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (data.size() < 8 + 20 + 32) return fail("file too small");
    quantized = std::memcmp(data.data(), "NNUEWDL1", 8) == 0;
    bool flt = std::memcmp(data.data(), "NNUEFLT1", 8) == 0;
    if (!quantized && !flt) return fail("bad magic");
    uint32_t h[5];
    std::memcpy(h, data.data() + 8, 20);
    if (h[0] != 1) return fail("unsupported version");
    arch = int(h[1]); l1 = int(h[2]); heads = int(h[4]);
    if (h[3] != 1 || l1 <= 0 || (heads != 1 && heads != 3)) return fail("unsupported architecture");
    if (quantized && l1 > MAX_L1) return fail("L1 wider than MAX_L1");
    size_t n_ft = size_t(FEATURES) * l1, n_ow = size_t(heads) * 2 * l1;
    size_t body = quantized ? 2 * (n_ft + l1 + n_ow) + 4 * heads : 4 * (n_ft + l1 + n_ow + heads);
    if (data.size() != 28 + body + 32) return fail("size mismatch");
    const char* p = data.data() + 28;
    if (quantized) {
        q_ft_w.resize(n_ft); q_ft_b.resize(l1); q_out_w.resize(n_ow); q_out_b.resize(heads);
        std::memcpy(q_ft_w.data(), p, 2 * n_ft); p += 2 * n_ft;
        std::memcpy(q_ft_b.data(), p, 2 * l1); p += 2 * l1;
        std::memcpy(q_out_w.data(), p, 2 * n_ow); p += 2 * n_ow;
        std::memcpy(q_out_b.data(), p, 4 * heads);
    } else {
        ft_w.resize(n_ft); ft_b.resize(l1); out_w.resize(n_ow); out_b.resize(heads);
        std::memcpy(ft_w.data(), p, 4 * n_ft); p += 4 * n_ft;
        std::memcpy(ft_b.data(), p, 4 * l1); p += 4 * l1;
        std::memcpy(out_w.data(), p, 4 * n_ow); p += 4 * n_ow;
        std::memcpy(out_b.data(), p, 4 * heads);
    }
    this->path = path;
    finalize();
    return true;
}

void Net::finalize() {
    fast_out = quantized && l1 <= MAX_L1;
    for (int16_t w : q_out_w)
        if (std::abs(int(w)) * QA > 32767) { fast_out = false; break; }
}

void Net::forward(const Position& pos, float* out) const {
    const Color persp[2] = {pos.stm, ~pos.stm};
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

void refresh(const Net& net, const Position& pos, Accumulator& acc) {
    const int n = net.l1;
    for (int c = 0; c < 2; ++c) {
        int16_t* a = acc.v[c];
        std::memcpy(a, net.q_ft_b.data(), sizeof(int16_t) * n);
        Bitboard occ = pos.pieces();
        while (occ) {
            int s = pop_lsb(occ);
            const int16_t* w = &net.q_ft_w[size_t(feature(Color(c), pos.board[s], s)) * n];
            for (int i = 0; i < n; ++i) a[i] += w[i];
        }
    }
    acc.computed = true;
}

void update(const Net& net, const Position& parent, const Position& child, const Accumulator& from, Accumulator& to) {
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
        const int16_t* src = from.v[c];
        int16_t* dst = to.v[c];
        auto W = [&](int pc, int sq) { return &net.q_ft_w[size_t(feature(Color(c), Piece(pc), sq)) * n]; };
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

static int64_t screlu_dot_scalar(const int16_t* acc, const int16_t* w, int n) {
    int64_t sum = 0;
    for (int i = 0; i < n; ++i) {
        int64_t x = std::min(std::max(int(acc[i]), 0), QA);
        sum += x * x * w[i];
    }
    return sum;
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
#endif

void output(const Net& net, const Accumulator& acc, Color stm, float* out, bool simd) {
    const int n = net.l1;
    const int16_t* persp[2] = {acc.v[stm], acc.v[~stm]};
    for (int h = 0; h < net.heads; ++h) {
        int64_t sum = 0;
        for (int v = 0; v < 2; ++v) {
            const int16_t* w = &net.q_out_w[size_t(h) * 2 * n + size_t(v) * n];
#ifdef __AVX2__
            if (simd && net.fast_out && n % 16 == 0) { sum += screlu_dot_avx2(persp[v], w, n); continue; }
#else
            (void)simd;
#endif
            sum += screlu_dot_scalar(persp[v], w, n);
        }
        out[h] = float((double(sum) + double(net.q_out_b[h])) / (double(QA) * QA * QB));
    }
}

void wdl_probs(const float* z, int heads, double* p) {
    if (heads == 1) { p[0] = 1.0 / (1.0 + std::exp(-double(z[0]))); p[1] = 0; p[2] = 1 - p[0]; return; }
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
        float out[3];
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
