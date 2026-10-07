/*
 * (c) 2015 basil, all rights reserved,
 * Modifications Copyright (c) 2016-2019 by Jon Dart
 * Modifications Copyright (c) 2020-2026 by Andrew Grant
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#pragma once

/* Glue for the Bulwark engine (LERF squares, a1 = 0). Attack functions live in src/tablebase.cpp.
 * Pyrrhic defines white as 1; Bulwark defines WHITE as 0, handled in pyr_pawn_attacks. */
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
uint64_t pyr_pawn_attacks(unsigned sq, unsigned pyrrhic_color);
uint64_t pyr_knight_attacks(unsigned sq);
uint64_t pyr_bishop_attacks(unsigned sq, uint64_t occ);
uint64_t pyr_rook_attacks(unsigned sq, uint64_t occ);
uint64_t pyr_queen_attacks(unsigned sq, uint64_t occ);
uint64_t pyr_king_attacks(unsigned sq);
#ifdef __cplusplus
}
#endif

#define PYRRHIC_POPCOUNT(x)              (__builtin_popcountll(x))
#define PYRRHIC_LSB(x)                   (__builtin_ctzll(x))
#define PYRRHIC_POPLSB(x)                ((*(x)) &= (*(x)) - 1)

#define PYRRHIC_PAWN_ATTACKS(sq, c)      (pyr_pawn_attacks(sq, c))
#define PYRRHIC_KNIGHT_ATTACKS(sq)       (pyr_knight_attacks(sq))
#define PYRRHIC_BISHOP_ATTACKS(sq, occ)  (pyr_bishop_attacks(sq, occ))
#define PYRRHIC_ROOK_ATTACKS(sq, occ)    (pyr_rook_attacks(sq, occ))
#define PYRRHIC_QUEEN_ATTACKS(sq, occ)   (pyr_queen_attacks(sq, occ))
#define PYRRHIC_KING_ATTACKS(sq)         (pyr_king_attacks(sq))
