"""Décision de géométrie -> carte de dose. On construit et on visualise."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SIZE = 15
START = (7, 0)
GOAL = (7, 14)
SOURCES = [(7, 7, 60.0), (5, 8, 25.0), (9, 6, 25.0)]   # (ligne, col, µSv/h)
OBSTACLES = {(2, 5), (3, 5), (11, 9), (12, 9)}

def build_dose_map():
    dose = np.zeros((SIZE, SIZE), dtype=np.float32)
    for i in range(SIZE):
        for j in range(SIZE):
            total = 0.0
            for (si, sj, inten) in SOURCES:
                d2 = (i - si) ** 2 + (j - sj) ** 2
                total += inten / (d2 + 1.0)          # loi inverse carré, +1 anti division par zéro
            dose[i, j] = total
    for o in OBSTACLES:
        dose[o] = 0.0
    return dose

dose = build_dose_map()

# --- quelques vérifications chiffrées ---
print(f"Dose max sur la carte : {dose.max():.1f} µSv/h  (sur/à côté de la source centrale)")
print(f"Dose min (hors murs)  : {dose[dose>0].min():.2f} µSv/h  (dans un coin froid)")
print(f"Dose sur le chemin direct ligne 7 : {[round(float(dose[7,j]),1) for j in range(SIZE)]}")
print(f"  -> somme sur la ligne droite    : {dose[7,:].sum():.1f}")
# un contournement par le haut (ligne 1) coûte quoi ?
print(f"Dose sur la ligne 1 (contournement haut) : {dose[1,:].sum():.1f}")
print(f"Dose sur la ligne 13 (contournement bas) : {dose[13,:].sum():.1f}")

# --- visualisation ---
fig, ax = plt.subplots(figsize=(7.5, 6.8))
im = ax.imshow(dose, cmap="inferno", origin="upper")
cbar = fig.colorbar(im, ax=ax, label="Débit de dose (µSv/h)")
for o in OBSTACLES:
    ax.add_patch(plt.Rectangle((o[1]-0.5, o[0]-0.5), 1, 1, facecolor="#5F5E5A", edgecolor="k", lw=0.5))
for (si, sj, inten) in SOURCES:
    ax.plot(sj, si, "*", color="#FFE23D", markersize=22, markeredgecolor="k", markeredgewidth=1.2)
    ax.annotate(f"{inten:g}", (sj, si), textcoords="offset points", xytext=(9, 9),
                color="white", fontsize=10, fontweight="bold")
ax.plot(START[1], START[0], "s", color="#4EA1FF", markersize=15, markeredgecolor="k", label="Départ")
ax.plot(GOAL[1], GOAL[0], "^", color="#3BD16F", markersize=17, markeredgecolor="k", label="Sortie")
ax.axhline(7, color="cyan", lw=1, ls=":", alpha=0.6)  # la ligne de traversée directe
ax.set_xticks(range(SIZE)); ax.set_yticks(range(SIZE))
ax.set_xticklabels(range(SIZE), fontsize=7); ax.set_yticklabels(range(SIZE), fontsize=7)
ax.set_title("Carte de dose — géométrie conçue (grille 15×15)", fontsize=12)
ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
fig.tight_layout()
fig.savefig("carte_dose.png", dpi=140)
print("\nFigure : carte_dose.png")
