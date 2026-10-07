// Lc0 training data v6 (input format 1) -> master records (56 B, see docs/interfaces.md).
//
// Input (stdin): frames of  { u32 magic 'GAME', u32 n_records, u32 game_local_id, u32 reserved }  followed by
// n_records * 8356 bytes of V6TrainingData, i.e. one decompressed training.*.gz file per frame (one game).
// Output: a shard file (--out) of master records; statistics as JSON on stderr.
//
// Filters (B.7 gotchas): early plies (fewer than 8 plies of history), in-check positions, positions whose best
// move is a capture or promotion (decoded with the 1858-move policy table), extreme |root_q|, records marked for
// deletion, malformed positions. An optional tablebase override replaces the target by the exact outcome for
// <= 5 piece positions with a zero halfmove clock (WDL tables assume rule50 == 0; positions with a running clock
// are left alone).
//
// Orientation: Lc0 planes are stored from the side to move's view (us at the bottom; ranks mirrored when black
// is to move) and every byte is bit-reversed (ReverseBitsInBytes).
#include "movegen.h"
#include "tablebase.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#pragma pack(push, 1)
struct V6 {
    uint32_t version, input_format;
    float probabilities[1858];
    uint64_t planes[104];
    uint8_t castling_us_ooo, castling_us_oo, castling_them_ooo, castling_them_oo;
    uint8_t side_to_move_or_enpassant, rule50_count, invariance_info, dummy;
    float root_q, best_q, root_d, best_d, root_m, best_m, plies_left;
    float result_q, result_d, played_q, played_d, played_m;
    float orig_q, orig_d, orig_m;
    uint32_t visits;
    uint16_t played_idx, best_idx;
    float policy_kld, q_st;
};
struct Master {
    uint64_t occ;
    uint8_t pcs[16];
    uint8_t stm_cas, ep, r50;
    int16_t sf_cp;
    uint16_t sf_w, sf_d;
    int16_t lc_q;
    uint16_t lc_d;
    int16_t own_cp;
    uint8_t result, flags;
    uint16_t ply;
    uint64_t gid;
    uint8_t rsv[5];
};
struct Header {
    char magic[8];
    uint32_t version, record_size;
    uint64_t count;
    uint32_t shard_id, reserved;
};
#pragma pack(pop)
static_assert(sizeof(V6) == 8356, "V6 size");
static_assert(sizeof(Master) == 56, "Master size");
static_assert(sizeof(Header) == 32, "Header size");

// The 1858 policy moves: every queen-ray or knight move from each square (sorted by destination square),
// then the underpromotions (q/r/b suffix, three per from/to pair). Indexed in "us at the bottom" coordinates.
static std::vector<std::string> build_policy() {
    std::vector<std::string> out;
    auto name = [](int s) { return std::string{char('a' + (s & 7)), char('1' + (s >> 3))}; };
    for (int from = 0; from < 64; ++from) {
        Bitboard t = rook_attacks(from, 0) | bishop_attacks(from, 0) | KnightAttacks[from];
        while (t) { int to = pop_lsb(t); out.push_back(name(from) + name(to)); }
    }
    for (int f = 0; f < 8; ++f)
        for (int df = -1; df <= 1; ++df) {
            int tf = f + df;
            if (tf < 0 || tf > 7) continue;
            for (char p : {'q', 'r', 'b'}) out.push_back(name(48 + f) + name(56 + tf) + p);
        }
    return out;
}

static uint64_t reverse_bits_in_bytes(uint64_t x) {
    x = ((x >> 1) & 0x5555555555555555ULL) | ((x & 0x5555555555555555ULL) << 1);
    x = ((x >> 2) & 0x3333333333333333ULL) | ((x & 0x3333333333333333ULL) << 2);
    x = ((x >> 4) & 0x0F0F0F0F0F0F0F0FULL) | ((x & 0x0F0F0F0F0F0F0F0FULL) << 4);
    return x;
}

struct Stats {
    uint64_t seen = 0, kept = 0, games = 0, bad_version = 0, deleted = 0, early = 0, in_check = 0, best_cap_promo = 0,
             extreme_q = 0, malformed = 0, tb_override = 0, stride_skipped = 0, nan = 0;
};

int main(int argc, char** argv) {
    std::string out_path, syzygy;
    uint32_t session = 0, shard_id = 0, source_id = 0;
    int stride = 1;
    float max_abs_q = 0.98f;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
        if (a == "--out") out_path = next();
        else if (a == "--session") session = std::stoul(next());
        else if (a == "--shard-id") shard_id = std::stoul(next());
        else if (a == "--source-id") source_id = std::stoul(next());
        else if (a == "--stride") stride = std::max(1, std::stoi(next()));
        else if (a == "--max-abs-q") max_abs_q = std::stof(next());
        else if (a == "--syzygy") syzygy = next();
        else if (a == "--dump-policy") { Bitboards::init(); for (auto& m : build_policy()) std::printf("%s\n", m.c_str()); return 0; }
        else { std::fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
    }
    if (out_path.empty()) { std::fprintf(stderr, "usage: lc0_convert --out shard.bin [--session N --shard-id N --source-id N --stride K --syzygy path]\n"); return 2; }
    Bitboards::init();
    Zobrist::init();
    if (!syzygy.empty()) TB::init(syzygy);
    const std::vector<std::string> policy = build_policy();
    if (policy.size() != 1858) { std::fprintf(stderr, "policy table has %zu entries\n", policy.size()); return 3; }

    FILE* out = std::fopen(out_path.c_str(), "wb");
    if (!out) { std::perror("open out"); return 3; }
    Header hdr{};
    std::memcpy(hdr.magic, "NNUEMR01", 8);
    hdr.version = 1; hdr.record_size = 56; hdr.shard_id = shard_id;
    std::fwrite(&hdr, sizeof hdr, 1, out);

    Stats st;
    std::vector<V6> buf;
    uint32_t fh[4];
    while (std::fread(fh, sizeof fh, 1, stdin) == 1) {
        if (fh[0] != 0x47414D45) { std::fprintf(stderr, "bad frame magic\n"); return 4; }
        uint32_t n = fh[1], game_local = fh[2];
        buf.resize(n);
        if (std::fread(buf.data(), sizeof(V6), n, stdin) != n) { std::fprintf(stderr, "truncated frame\n"); return 4; }
        ++st.games;
        uint64_t gid = (uint64_t(session) << 32) | game_local;

        for (uint32_t idx = 0; idx < n; ++idx) {
            const V6& r = buf[idx];
            ++st.seen;
            if (r.version != 6 || r.input_format != 1) { ++st.bad_version; continue; }
            if (r.invariance_info & 0x40) { ++st.deleted; continue; }
            if (!std::isfinite(r.root_q) || !std::isfinite(r.root_d) || !std::isfinite(r.result_q)) { ++st.nan; continue; }
            if (idx < 8) { ++st.early; continue; }
            uint64_t pl[104];
            for (int i = 0; i < 104; ++i) pl[i] = reverse_bits_in_bytes(r.planes[i]);
            bool hist7 = false;   // history position 7 (the oldest) must exist: at least 8 plies were played
            for (int i = 91; i < 103; ++i) hist7 |= pl[i] != 0;
            if (!hist7) { ++st.early; continue; }
            if (std::fabs(r.root_q) > max_abs_q) { ++st.extreme_q; continue; }
            if (stride > 1 && (idx + gid) % stride != 0) { ++st.stride_skipped; continue; }

            // Best move decode (in us-at-bottom coordinates): skip captures and promotions.
            if (r.best_idx >= 1858) { ++st.malformed; continue; }
            const std::string& bm = policy[r.best_idx];
            int bfrom = (bm[0] - 'a') + 8 * (bm[1] - '1'), bto = (bm[2] - 'a') + 8 * (bm[3] - '1');
            Bitboard their_occ = 0;
            for (int i = 6; i < 12; ++i) their_occ |= pl[i];
            bool pawn_from = pl[0] & sq_bb(bfrom);
            bool capture = (their_occ & sq_bb(bto)) || (pawn_from && (bfrom & 7) != (bto & 7));
            bool promo = bm.size() == 5 || (pawn_from && (bfrom >> 3) == 6);
            if (capture || promo) { ++st.best_cap_promo; continue; }

            // Build the position in real coordinates.
            bool black = r.side_to_move_or_enpassant != 0;
            Color us = black ? BLACK : WHITE;
            Piece board[64];
            for (auto& p : board) p = NO_PIECE;
            bool dup = false;
            for (int k = 0; k < 12; ++k) {
                Bitboard b = pl[k];
                Color c = k < 6 ? us : ~us;
                while (b) {
                    int s = pop_lsb(b);
                    int real = black ? (s ^ 56) : s;
                    if (board[real] != NO_PIECE) dup = true;
                    board[real] = make_piece(c, PieceType(k % 6));
                }
            }
            if (dup) { ++st.malformed; continue; }
            std::string fen;
            for (int rk = 7; rk >= 0; --rk) {
                int empty = 0;
                for (int f = 0; f < 8; ++f) {
                    Piece p = board[rk * 8 + f];
                    if (p == NO_PIECE) { ++empty; continue; }
                    if (empty) { fen += char('0' + empty); empty = 0; }
                    fen += "PNBRQKpnbrqk"[p];
                }
                if (empty) fen += char('0' + empty);
                if (rk) fen += '/';
            }
            fen += black ? " b " : " w ";
            std::string cas;
            bool wk = black ? r.castling_them_oo : r.castling_us_oo, wq = black ? r.castling_them_ooo : r.castling_us_ooo;
            bool bk = black ? r.castling_us_oo : r.castling_them_oo, bq = black ? r.castling_us_ooo : r.castling_them_ooo;
            // Lc0 data also contains Chess960 games. Keep only rights that match the standard king/rook squares;
            // the others are dropped (castling is not a net input, and Stockfish gets a plain-chess FEN).
            wk = wk && board[E1] == W_KING && board[H1] == W_ROOK;
            wq = wq && board[E1] == W_KING && board[A1] == W_ROOK;
            bk = bk && board[60] == B_KING && board[63] == B_ROOK;
            bq = bq && board[60] == B_KING && board[56] == B_ROOK;
            if (wk) cas += 'K';
            if (wq) cas += 'Q';
            if (bk) cas += 'k';
            if (bq) cas += 'q';
            fen += (cas.empty() ? "-" : cas) + " - " + std::to_string(r.rule50_count) + " 1";
            Position pos;
            if (!pos.set_fen(fen)) { ++st.malformed; continue; }
            // The side not to move must not be in check (illegal position).
            if (pos.attacked_by(pos.stm, pos.king_sq(~pos.stm), pos.pieces())) { ++st.malformed; continue; }
            if (pos.in_check()) { ++st.in_check; continue; }

            Master m{};
            m.occ = pos.pieces();
            int k = 0;
            Bitboard occ = m.occ;
            while (occ) {
                int s = pop_lsb(occ);
                uint8_t code = uint8_t(pos.board[s]);
                m.pcs[k >> 1] |= (k & 1) ? uint8_t(code << 4) : code;
                ++k;
            }
            m.stm_cas = uint8_t((black ? 1 : 0) | (pos.castling << 1));
            m.ep = 0;
            m.r50 = uint8_t(std::min<int>(r.rule50_count, 255));
            m.sf_cp = -32768; m.sf_w = 0xFFFF; m.sf_d = 0xFFFF;
            m.lc_q = int16_t(std::lround(r.root_q * 10000.0f));
            m.lc_d = uint16_t(std::lround(std::min(std::max(r.root_d, 0.0f), 1.0f) * 10000.0f));
            m.own_cp = -32768;
            m.result = r.result_q > 0.5f ? 2 : (r.result_q < -0.5f ? 0 : 1);   // side to move's view
            m.flags = uint8_t((source_id & 0x7F) << 1);
            m.ply = uint16_t(std::min<uint32_t>(idx, 65535));
            m.gid = gid;

            if (TB::largest() > 0 && m.r50 == 0 && !pos.castling && popcount(pos.pieces()) <= TB::largest()) {
                int w = TB::probe_wdl(pos);
                if (w != TB::FAIL) {
                    m.result = w == TB::WIN ? 2 : w == TB::LOSS ? 0 : 1;   // cursed win / blessed loss are draws
                    m.flags |= 1;
                    ++st.tb_override;
                }
            }
            std::fwrite(&m, sizeof m, 1, out);
            ++st.kept;
        }
    }
    hdr.count = st.kept;
    std::fseek(out, 0, SEEK_SET);
    std::fwrite(&hdr, sizeof hdr, 1, out);
    std::fclose(out);
    std::fprintf(stderr,
        "{\"games\":%llu,\"seen\":%llu,\"kept\":%llu,\"bad_version\":%llu,\"deleted\":%llu,\"nan\":%llu,\"early\":%llu,"
        "\"extreme_q\":%llu,\"stride_skipped\":%llu,\"best_cap_promo\":%llu,\"in_check\":%llu,\"malformed\":%llu,\"tb_override\":%llu}\n",
        (unsigned long long)st.games, (unsigned long long)st.seen, (unsigned long long)st.kept,
        (unsigned long long)st.bad_version, (unsigned long long)st.deleted, (unsigned long long)st.nan,
        (unsigned long long)st.early, (unsigned long long)st.extreme_q, (unsigned long long)st.stride_skipped,
        (unsigned long long)st.best_cap_promo, (unsigned long long)st.in_check, (unsigned long long)st.malformed,
        (unsigned long long)st.tb_override);
    return 0;
}
