#include "nnue.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>

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
    return true;
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
