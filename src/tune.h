#pragma once
// Search constants tuned by SPSA (scripts/spsa.py). A normal build makes them compile-time constants (no speed cost);
// a tuning build (CMake -DBULWARK_TUNE=ON) makes them variables, exposed as UCI spin options and listed by the
// "tunables" command as "tunable <name> <value> <min> <max> <c_end>".
//
// X(name, default, min, max, c_end): c_end is the SPSA perturbation size at the end of a run (about (max - min) / 20,
// at least 0.5 for small integers).
#define BULWARK_TUNABLES(X)                                                                                       \
    X(LmrBase, 78, 0, 200, 8)           /* LMR table: LmrBase / 100 + ln(depth) ln(moves) / (LmrDiv / 100) */     \
    X(LmrDiv, 247, 150, 400, 12)                                                                                  \
    X(LmrHistDiv, 8192, 2048, 16384, 600) /* LMR: reduction -= (history + continuation history) / LmrHistDiv */ \
    X(RazorMargin, 250, 50, 600, 25)    /* razoring: eval + RazorMargin * depth < alpha */                        \
    X(RfpMargin, 80, 30, 200, 8)        /* reverse futility: eval - RfpMargin * (depth - improving) >= beta */    \
    X(RfpDepth, 8, 4, 12, 0.5)                                                                                    \
    X(NmpBase, 3, 2, 5, 0.5)            /* null move: R = NmpBase + depth / NmpDepthDiv + (eval - beta) / ... */  \
    X(NmpDepthDiv, 3, 2, 6, 0.5)                                                                                  \
    X(NmpEvalDiv, 200, 100, 400, 15)                                                                              \
    X(SingDepth, 7, 4, 10, 0.5)         /* singular extension: depth >= SingDepth */                              \
    X(SingMargin, 32, 8, 64, 3)         /* singular beta = tt score - SingMargin * depth / 16 */                  \
    X(LmpBase, 3, 1, 8, 0.5)            /* late move pruning: moves > (LmpBase + depth^2) * (1 + improving) */    \
    X(LmpDepth, 8, 4, 12, 0.5)                                                                                    \
    X(FutBase, 90, 20, 200, 8)          /* futility: eval + FutBase + FutMul * depth <= alpha */                  \
    X(FutMul, 90, 30, 200, 8)                                                                                     \
    X(FutDepth, 7, 4, 10, 0.5)                                                                                    \
    X(SeeQuiet, 25, 5, 60, 3)           /* SEE pruning: quiets below -SeeQuiet * depth^2, captures -SeeCapt * d */ \
    X(SeeCapt, 90, 30, 200, 8)                                                                                    \
    X(SeeDepth, 6, 3, 10, 0.5)                                                                                    \
    X(HistMul, 170, 80, 300, 12)        /* history bonus: min(HistMul * depth - HistSub, HistMax) */              \
    X(HistSub, 100, 0, 300, 15)                                                                                   \
    X(HistMax, 1700, 800, 3000, 120)                                                                              \
    X(AspDelta, 25, 10, 60, 3)          /* first aspiration window half-width */

namespace Tune {

#ifdef BULWARK_TUNE
#define BW_TUNE_DECL(n, d, lo, hi, c) inline int n = d;
#else
#define BW_TUNE_DECL(n, d, lo, hi, c) constexpr int n = d;
#endif
BULWARK_TUNABLES(BW_TUNE_DECL)
#undef BW_TUNE_DECL

#ifdef BULWARK_TUNE
struct Entry {
    const char* name;
    int* value;
    int def, lo, hi;
    double c_end;
};
#define BW_TUNE_ENTRY(n, d, lo, hi, c) Entry{#n, &n, d, lo, hi, double(c)},
inline Entry entries[] = {BULWARK_TUNABLES(BW_TUNE_ENTRY)};
#undef BW_TUNE_ENTRY
#endif

}  // namespace Tune
