"""
3_KarlWeierstrass_NEXT2_2a3c — POGODNI deo iz 1_KarlWeierstrass_v2.py
Aparat 2a: Brownovo kretanje  +  Test 3c: Mutual Information

Self-contained:
  - KORAK 1: ucitavanje 4624 izvlacenja i izgradnja f(t) = lex-indeks
  - KORAK 2a: Brown inkrementi (priprema)
  - KORAK 2a3c: MI nad centriranim Brown inkrementima,
                MI nad Brown-putanjom kao kontrola,
                shuffled max MI referenca

Output:
  3_KarlWeierstrass_NEXT2_2a3c.png
  3_KarlWeierstrass_NEXT2_2a3c.txt
"""

import csv
import math
import os
import time
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np


T0 = time.time()

CSV_DRAWS = "/data/loto7_4624_k43.csv"

HERE = os.path.dirname(os.path.abspath(__file__))
PNG_PATH = os.path.join(HERE, "3_KarlWeierstrass_NEXT2_2a3c.png")
TXT_PATH = os.path.join(HERE, "3_KarlWeierstrass_NEXT2_2a3c.txt")

N_MAX = 39
K_PICK = 7
TOTAL_COMBOS = math.comb(N_MAX, K_PICK)


# ─── helperi (samo oni potrebni za 2a + 2a3c) ────────────────────────
def read_loto_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < K_PICK:
                continue
            try:
                nums = tuple(sorted(int(x) for x in row[:K_PICK]))
            except ValueError:
                continue
            if len(nums) == K_PICK and len(set(nums)) == K_PICK:
                rows.append(nums)
    return rows


def lex_rank_1based(combo, n=N_MAX, k=K_PICK):
    """1-based lex indeks (poklapa se sa rednim brojem u kombinacije_39C7.csv)."""
    combo = tuple(sorted(combo))
    rank0 = 0
    prev = 0
    for i, value in enumerate(combo):
        remaining = k - i - 1
        for candidate in range(prev + 1, value):
            rank0 += math.comb(n - candidate, remaining)
        prev = value
    return rank0 + 1


def quantile_symbols(series, bins=16):
    """Pretvara niz u diskretne simbole preko kvantil-binova."""
    x = np.asarray(series, dtype=float)
    edges = np.quantile(x, np.linspace(0.0, 1.0, bins + 1))
    edges = np.unique(edges)
    if len(edges) <= 2:
        edges = np.linspace(float(x.min()), float(x.max()), bins + 1)
    return np.digitize(x, edges[1:-1], right=False)


def discrete_mutual_information(sym_a, sym_b):
    """Diskretna mutual information u bitovima."""
    a = np.asarray(sym_a, dtype=int)
    b = np.asarray(sym_b, dtype=int)
    n = len(a)
    size_a = int(a.max()) + 1
    size_b = int(b.max()) + 1
    table = np.zeros((size_a, size_b), dtype=float)
    np.add.at(table, (a, b), 1.0)
    pxy = table / n
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    expected = px @ py
    mask = pxy > 0
    return float(np.sum(pxy[mask] * np.log2(pxy[mask] / expected[mask])))


def mutual_information_lags(series, max_lag=60, bins=16):
    """MI(series[t], series[t+lag]) za lagove 1..max_lag."""
    symbols = quantile_symbols(series, bins=bins)
    vals = []
    for lag in range(1, max_lag + 1):
        vals.append(discrete_mutual_information(symbols[:-lag], symbols[lag:]))
    return np.asarray(vals, dtype=float)


# ─── KORAK 1: f(t) = lex-indeks ──────────────────────────────────────
draws = read_loto_csv(CSV_DRAWS)
N = len(draws)
lex_idx = np.array([lex_rank_1based(c) for c in draws], dtype=np.float64)

print()
print("3_KarlWeierstrass_NEXT2_2a3c — KORAK 1: formiranje krive f(t)")
print(f"  CSV:                  {CSV_DRAWS}")
print(f"  Ucitano izvlacenja:    {N}")
print(f"  C(39,7):              {TOTAL_COMBOS:,}")
print()

with open(TXT_PATH, "w", encoding="utf-8") as f:
    f.write("3_KarlWeierstrass_NEXT2_2a3c — Brownovo kretanje + Mutual Information (POGODNO)\n")
    f.write("=" * 60 + "\n\n")
    f.write("KORAK 1: Weierstrass-ova funkcija nad svih izvucenih kombinacija\n\n")
    f.write(f"  CSV izvucenih:        {CSV_DRAWS}\n")
    f.write(f"  Ucitano izvlacenja:    {N}\n")
    f.write(f"  C(39,7):              {TOTAL_COMBOS:,}\n")
    f.write("  f(t) = lex-indeks izvucene kombinacije u skupu svih 39C7\n\n")


# ─── KORAK 2a: priprema Brown inkremenata (samo ono sto 2a3c koristi) ─
incr = np.diff(lex_idx)
brown_incr_centered = incr - incr.mean()
brown_path = np.cumsum(brown_incr_centered)


# ─── KORAK 2a3c: Mutual Information nad Brown inkrementima ───────────
T0_2A3C = time.time()

mi_max_lag = 60
mi_bins = 16
mi_lags = np.arange(1, mi_max_lag + 1)
mi_brown_incr = mutual_information_lags(brown_incr_centered, mi_max_lag, mi_bins)
mi_brown_path = mutual_information_lags(brown_path, mi_max_lag, mi_bins)

max_mi_incr = float(mi_brown_incr.max())
max_mi_lag = int(mi_lags[int(np.argmax(mi_brown_incr))])
top_mi_idx = np.argsort(mi_brown_incr)[-10:][::-1]
top_mi_pairs = [(int(mi_lags[i]), float(mi_brown_incr[i])) for i in top_mi_idx]

rng_2a3c = np.random.default_rng(44)
mi_shuffle_runs = 200
shuffle_max_mi = []
for _ in range(mi_shuffle_runs):
    shuffled = rng_2a3c.permutation(brown_incr_centered)
    shuffled_mi = mutual_information_lags(shuffled, mi_max_lag, mi_bins)
    shuffle_max_mi.append(float(shuffled_mi.max()))
shuffle_max_mi = np.asarray(shuffle_max_mi, dtype=float)
shuffle_mi_mean = float(shuffle_max_mi.mean())
shuffle_mi_std = float(shuffle_max_mi.std(ddof=1))
shuffle_mi_p = float(np.mean(shuffle_max_mi >= max_mi_incr))
shuffle_mi_z = (max_mi_incr - shuffle_mi_mean) / (shuffle_mi_std + 1e-12)

if shuffle_mi_p <= 0.05:
    mi_note = "postoji MI signal iznad shuffled Brown reference"
else:
    mi_note = "nema jak MI signal iznad shuffled Brown reference"

print()
print("KORAK 2a3c: Aparat 2a Brownovo kretanje + Test 3c Mutual Information")
print(f"  max MI(dX) lag 1..{mi_max_lag}: {max_mi_incr:.6f} bits  (lag={max_mi_lag})")
print(f"  shuffled max MI: mean={shuffle_mi_mean:.6f} std={shuffle_mi_std:.6f} "
      f"z={shuffle_mi_z:.2f} p={shuffle_mi_p:.4f}")
print(f"  ⇒ {mi_note}")
print()

fig2a3c, ax2a3c = plt.subplots(1, 3, figsize=(16, 5))
fig2a3c.suptitle("KORAK 2a3c: Brownovo kretanje + Mutual Information test  (POGODNO)",
                 fontsize=13, fontweight="bold")

ax2a3c[0].plot(mi_lags, mi_brown_incr, "o-", markersize=3, color="darkorange")
ax2a3c[0].set_title("MI centriranih Brown inkremenata")
ax2a3c[0].set_xlabel("lag")
ax2a3c[0].set_ylabel("MI [bits]")
ax2a3c[0].grid(True, alpha=0.25)

ax2a3c[1].plot(mi_lags, mi_brown_path, "o-", markersize=3, color="steelblue")
ax2a3c[1].set_title("Kontrola: MI Brown-putanje")
ax2a3c[1].set_xlabel("lag")
ax2a3c[1].set_ylabel("MI [bits]")
ax2a3c[1].grid(True, alpha=0.25)

ax2a3c[2].hist(shuffle_max_mi, bins=24, color="lightgray", edgecolor="white")
ax2a3c[2].axvline(max_mi_incr, color="crimson", linewidth=2,
                  label=f"observed={max_mi_incr:.4f}")
ax2a3c[2].axvline(shuffle_mi_mean, color="black", linestyle="--",
                  label=f"shuffle mean={shuffle_mi_mean:.4f}")
ax2a3c[2].set_title("Shuffled max MI referenca")
ax2a3c[2].set_xlabel("max MI [bits]")
ax2a3c[2].set_ylabel("broj")
ax2a3c[2].legend(fontsize=8)

for a in ax2a3c:
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)

fig2a3c.tight_layout()
fig2a3c.savefig(PNG_PATH, dpi=150, bbox_inches="tight")
plt.show()

with open(TXT_PATH, "a", encoding="utf-8") as f:
    f.write("\n")
    f.write("=" * 60 + "\n")
    f.write("KORAK 2a3c: Aparat 2a Brownovo kretanje + Test 3c Mutual Information\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"  PNG:                  {PNG_PATH}\n\n")
    f.write("Mutual Information nad centriranim Brown inkrementima:\n")
    f.write(f"  max lag               = {mi_max_lag}\n")
    f.write(f"  bins                  = {mi_bins}\n")
    f.write(f"  max MI                = {max_mi_incr:.8f} bits\n")
    f.write(f"  max MI lag            = {max_mi_lag}\n\n")
    f.write("Shuffled max MI referenca:\n")
    f.write(f"  runs                  = {mi_shuffle_runs}\n")
    f.write(f"  mean                  = {shuffle_mi_mean:.8f}\n")
    f.write(f"  std                   = {shuffle_mi_std:.8f}\n")
    f.write(f"  z                     = {shuffle_mi_z:.8f}\n")
    f.write(f"  p(shuffled >= obs)    = {shuffle_mi_p:.8f}\n")
    f.write(f"  interpret.            = {mi_note}\n\n")
    f.write("Top 10 MI lagova:\n")
    f.write(f"  {'lag':<8}{'MI [bits]':>16}\n")
    for lag, val in top_mi_pairs:
        f.write(f"  {lag:<8}{val:>16,.8f}\n")
    f.write("\n")

    elapsed_2a3c = time.time() - T0_2A3C
    f.write(f"Vreme KORAKA 2a3c: {timedelta(seconds=int(elapsed_2a3c))} ({elapsed_2a3c:.1f} s)\n")
    f.write(f"Ukupno vreme:       {timedelta(seconds=int(time.time()-T0))} ({time.time()-T0:.1f} s)\n")

print(f"PNG saved → {PNG_PATH}")
print(f"TXT saved → {TXT_PATH}")
print(f"Vreme KORAKA 2a3c: {timedelta(seconds=int(time.time()-T0_2A3C))} "
      f"({time.time()-T0_2A3C:.1f} s)")
print(f"Ukupno vreme:      {timedelta(seconds=int(time.time()-T0))} "
      f"({time.time()-T0:.1f} s)")
print()
print("KRAJ 3_KarlWeierstrass_NEXT2_2a3c.")
print()
"""
3_KarlWeierstrass_NEXT2_2a3c — KORAK 1: formiranje krive f(t)
  CSV:                  /data/loto7_4624_k43.csv
  Ucitano izvlacenja:   4624
  C(39,7):              15,380,937


KORAK 2a3c: Aparat 2a Brownovo kretanje + Test 3c Mutual Information
  max MI(dX) lag 1..60: 0.263697 bits  (lag=1)
  shuffled max MI: mean=0.044002 std=0.001862 z=117.97 p=0.0000
  ⇒ postoji MI signal iznad shuffled Brown reference

PNG saved → /3_KarlWeierstrass_NEXT2_2a3c.png
TXT saved → /3_KarlWeierstrass_NEXT2_2a3c.txt
Vreme KORAKA 2a3c: 0:00:19 (19.1 s)
Ukupno vreme:      0:00:19 (19.1 s)

KRAJ 3_KarlWeierstrass_NEXT2_2a3c.
"""





###############   PREDIKCIJA 2  ###############################

"""
NEXT2 (2a3c, MI) — uslovna distribucija dX_t+1 | dX_t (kvantil binovi) → očekivani inkrement.
"""


def lex_unrank_1based(rank, n=N_MAX, k=K_PICK):
    """Vracanje 1-based lex indeksa u Loto 7/39 kombinaciju."""
    rank0 = int(rank) - 1
    combo = []
    prev = 0
    for i in range(k):
        remaining = k - i - 1
        for candidate in range(prev + 1, n + 1):
            count = math.comb(n - candidate, remaining)
            if rank0 >= count:
                rank0 -= count
            else:
                combo.append(candidate)
                prev = candidate
                break
    return tuple(combo)


T0_PRED2 = time.time()

# MI je najjaci na lag=1, zato gradimo/formiram uslovnu distribuciju:
# sledeci centrirani dX | trenutni kvantil-bin centriranog dX.
mi_pred_bins = mi_bins
symbols = quantile_symbols(brown_incr_centered, bins=mi_pred_bins)
last_symbol = int(symbols[-1])
transition_mask = symbols[:-1] == last_symbol
conditional_next_centered = brown_incr_centered[1:][transition_mask]

if len(conditional_next_centered) == 0:
    conditional_next_centered = brown_incr_centered
    pred2_note = "nema istorijskih prelaza iz zadnjeg bina; koristi se globalna distribucija"
else:
    pred2_note = "koristi se uslovna distribucija sledeceg dX iz istog kvantil-bina"

last_lex = float(lex_idx[-1])
last_incr = float(incr[-1])
mean_incr = float(incr.mean())
pred_centered_incr = float(np.mean(conditional_next_centered))
pred_incr = mean_incr + pred_centered_incr
pred_lex_float = last_lex + pred_incr
pred_lex = int(np.clip(round(pred_lex_float), 1, TOTAL_COMBOS))
pred_combo = lex_unrank_1based(pred_lex)

quantile_grid = [0.10, 0.25, 0.50, 0.75, 0.90]
candidate_rows = []
seen_lex = set()
for q in quantile_grid:
    cand_centered = float(np.quantile(conditional_next_centered, q))
    cand_incr = mean_incr + cand_centered
    cand_lex = int(np.clip(round(last_lex + cand_incr), 1, TOTAL_COMBOS))
    if cand_lex in seen_lex:
        continue
    seen_lex.add(cand_lex)
    candidate_rows.append((q, cand_incr, cand_lex, lex_unrank_1based(cand_lex)))

print()
print("PREDIKCIJA 2 — NEXT2 / 2a3c / MI / uslovna distribucija inkremenata")
print(f"  MI max lag             = {max_mi_lag}")
print(f"  MI bins                = {mi_pred_bins}")
print(f"  zadnji bin             = {last_symbol}")
print(f"  istorijskih prelaza    = {len(conditional_next_centered)}")
print(f"  zadnji lex             = {int(last_lex):,}")
print(f"  zadnji inkrement       = {last_incr:,.2f}")
print(f"  pred. inkrement        = {pred_incr:,.2f}")
print(f"  pred. lex              = {pred_lex:,}")
print(f"  pred. kombinacija      = {pred_combo}")
print(f"  napomena               = {pred2_note}")
print("  kvantil kandidati:")
for q, cand_incr, cand_lex, combo in candidate_rows:
    print(f"    q={q:>4.2f}  dX={cand_incr:>14,.2f}  lex={cand_lex:>10,}  combo={combo}")
print()

with open(TXT_PATH, "a", encoding="utf-8") as f:
    f.write("\n")
    f.write("=" * 60 + "\n")
    f.write("PREDIKCIJA 2: NEXT2 / 2a3c / MI / uslovna distribucija inkremenata\n")
    f.write("=" * 60 + "\n\n")
    f.write("Model:\n")
    f.write("  MI je najjaci na lag=1.\n")
    f.write("  Trenutni centrirani dX se stavlja u kvantil-bin.\n")
    f.write("  Predikcija koristi istorijske sledece dX vrednosti iz istog bina.\n\n")
    f.write("Parametri:\n")
    f.write(f"  max MI lag             = {max_mi_lag}\n")
    f.write(f"  max MI                = {max_mi_incr:.8f} bits\n")
    f.write(f"  bins                  = {mi_pred_bins}\n")
    f.write(f"  zadnji bin             = {last_symbol}\n")
    f.write(f"  broj prelaza           = {len(conditional_next_centered)}\n")
    f.write(f"  mean(dX)               = {mean_incr:,.8f}\n")
    f.write(f"  zadnji lex             = {int(last_lex):,}\n")
    f.write(f"  zadnji inkrement       = {last_incr:,.8f}\n")
    f.write(f"  pred. centrirani dX    = {pred_centered_incr:,.8f}\n")
    f.write(f"  pred. inkrement        = {pred_incr:,.8f}\n")
    f.write(f"  napomena               = {pred2_note}\n\n")
    f.write("Glavna prognoza:\n")
    f.write(f"  pred. lex float        = {pred_lex_float:,.8f}\n")
    f.write(f"  pred. lex              = {pred_lex:,}\n")
    f.write(f"  pred. kombinacija      = {pred_combo}\n\n")
    f.write("Kvantil kandidati iz uslovne distribucije:\n")
    f.write(f"  {'q':>8}{'dX':>18}{'lex':>14}  kombinacija\n")
    for q, cand_incr, cand_lex, combo in candidate_rows:
        f.write(f"  {q:>8.2f}{cand_incr:>18,.8f}{cand_lex:>14,}  {combo}\n")
    f.write("\n")
    elapsed_pred2 = time.time() - T0_PRED2
    f.write(f"Vreme PREDIKCIJE 2: {timedelta(seconds=int(elapsed_pred2))} ({elapsed_pred2:.1f} s)\n")

print(f"TXT updated → {TXT_PATH}")
print(f"Vreme PREDIKCIJE 2: {timedelta(seconds=int(time.time()-T0_PRED2))} "
      f"({time.time()-T0_PRED2:.1f} s)")
print()

"""
Predikcija zasnovana na MI/kvantil-bin uslovnoj distribuciji.

PREDIKCIJA 2: iz poslednjeg kvantil-bina inkrementa uzimam istorijske sledeće inkremente iz istog stanja, računam očekivani sledeći inkrement i nekoliko kvantil-kandidata.

koristi MI signal na lag=1
nalazi kvantil-bin poslednjeg centriranog inkrementa
iz istorije uzima sledeće inkremente iz istog bina
računa glavnu prognozu kao očekivani sledeći inkrement
dodaje kvantil-kandidate q=0.10, 0.25, 0.50, 0.75, 0.90
svaki lex vraća u Loto 7/39 kombinaciju
upisuje rezultate u 3_KarlWeierstrass_NEXT2_2a3c.txt
"""



"""
3_KarlWeierstrass_NEXT2_2a3c — KORAK 1: formiranje krive f(t)
  CSV:                  /data/loto7_4624_k43.csv
  Ucitano izvlacenja:   4624
  C(39,7):              15,380,937


KORAK 2a3c: Aparat 2a Brownovo kretanje + Test 3c Mutual Information
  max MI(dX) lag 1..60: 0.263697 bits  (lag=1)
  shuffled max MI: mean=0.044002 std=0.001862 z=117.97 p=0.0000
  ⇒ postoji MI signal iznad shuffled Brown reference

PNG saved → /3_KarlWeierstrass_NEXT2_2a3c.png
TXT saved → /3_KarlWeierstrass_NEXT2_2a3c.txt
Vreme KORAKA 2a3c: 0:00:03 (4.0 s)
Ukupno vreme:      0:00:03 (4.0 s)

KRAJ 3_KarlWeierstrass_NEXT2_2a3c.


PREDIKCIJA 2 — NEXT2 / 2a3c / MI / uslovna distribucija inkremenata
  MI max lag             = 1
  MI bins                = 16
  zadnji bin             = 5
  istorijskih prelaza    = 288
  zadnji lex             = 513,114
  zadnji inkrement       = -2,143,496.00
  pred. inkrement        = 1,679,399.20
  pred. lex              = 2,192,513
  pred. kombinacija      = (1, x, 12, y, 16, x, 39)
  napomena               = koristi se uslovna distribucija sledeceg dX iz istog kvantil-bina
  kvantil kandidati:
    q=0.10  dX= -6,463,217.30  lex=         1  combo=(1, 2, 3, 4, 5, 6, 7)
    q=0.50  dX=  1,409,167.50  lex= 1,922,282  combo=(1, x, 11, y, 22, z, 32)
    q=0.75  dX=  6,284,021.50  lex= 6,797,136  combo=(3, x, 23, y, 29, z, 37)
    q=0.90  dX=  9,223,870.40  lex= 9,736,984  combo=(5, x, 16, y, 29, z, 37)

TXT updated → /3_KarlWeierstrass_NEXT2_2a3c.txt
Vreme PREDIKCIJE 2: 0:00:00 (0.0 s)
"""
