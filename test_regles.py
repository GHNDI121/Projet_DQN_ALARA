"""On teste chaque règle du MDP séparément. Chaque bloc = une garantie."""
import numpy as np
from env_alara import RadiationEvacuationEnv

env = RadiationEvacuationEnv()
N = env.size

def titre(t): print(f"\n--- {t} ---")

# 1. reset remet bien au départ
titre("1. reset() place l'agent au départ (7,0), dose remise à zéro")
obs, info = env.reset()
print(f"  pos = {info['pos']}  (attendu (7,0))")
print(f"  cumulative_dose = {info['cumulative_dose']}  (attendu 0.0)")
print(f"  obs = {obs}  -> forme {obs.shape}, valeurs dans [0,1]")
assert info['pos'] == (7,0) and info['cumulative_dose'] == 0.0 and obs.shape == (2,)

# 2. un déplacement normal met à jour la position
titre("2. step(droite) depuis (7,0) -> (7,1)")
env.reset()
obs, r, term, trunc, info = env.step(3)  # droite
print(f"  pos = {info['pos']}  (attendu (7,1))")
print(f"  reward = {r:.3f}  (négatif : dose + coût du pas)")
assert info['pos'] == (7,1)

# 3. les 4 actions vont dans le bon sens
titre("3. les 4 actions déplacent dans la bonne direction (depuis le centre)")
for a, attendu in [(0,(6,7)), (1,(8,7)), (2,(7,6)), (3,(7,8))]:
    e = RadiationEvacuationEnv(); e.reset(); e.pos = (7,7)
    _,_,_,_,info = e.step(a)
    print(f"  action {a} ({e.ACTION_NAMES[a]:>6}) : (7,7) -> {info['pos']}  (attendu {attendu})")
    assert info['pos'] == attendu

# 4. un mur bloque : l'agent reste sur place mais paie le pas
titre("4. collision avec un mur : position inchangée, pas quand même compté")
e = RadiationEvacuationEnv(); e.reset()
e.pos = (2, 4)                 # juste à gauche du mur (2,5)
t_avant = e.t
_,r,_,_,info = e.step(3)        # tente d'aller à droite -> mur en (2,5)
print(f"  depuis (2,4), action droite vers le mur (2,5)")
print(f"  pos = {info['pos']}  (attendu (2,4) : bloqué)")
print(f"  t : {t_avant} -> {e.t}  (le pas est compté malgré le blocage)")
assert info['pos'] == (2,4) and e.t == t_avant + 1

# 5. sortir de la grille est aussi bloqué
titre("5. bord de grille : action vers l'extérieur = reste sur place")
e = RadiationEvacuationEnv(); e.reset()   # départ (7,0), bord gauche
_,_,_,_,info = e.step(2)        # gauche -> hors grille
print(f"  depuis (7,0), action gauche (hors grille) : pos = {info['pos']}  (attendu (7,0))")
assert info['pos'] == (7,0)

# 6. atteindre la sortie : terminated=True et gros bonus
titre("6. arrivée à la sortie : terminated=True, reward inclut +100")
e = RadiationEvacuationEnv(); e.reset()
e.pos = (7, 13)                 # juste avant la sortie (7,14)
obs,r,term,trunc,info = e.step(3)   # droite -> sortie
print(f"  (7,13) -> {info['pos']}  | terminated = {term}  | reward = {r:.2f}")
assert info['pos'] == (7,14) and term is True and r > 90

# 7. dose cumulée s'accumule correctement
titre("7. la dose cumulée = somme des doses de chaque case visitée")
e = RadiationEvacuationEnv(); e.reset()
somme = 0.0
for a in [3,3,3]:               # trois pas vers la droite
    _,_,_,_,info = e.step(a)
    somme += info['dose_step']
print(f"  après 3 pas : cumulative_dose = {info['cumulative_dose']:.3f}")
print(f"  somme manuelle des dose_step  = {somme:.3f}  (doivent être égales)")
assert abs(info['cumulative_dose'] - somme) < 1e-5

# 8. troncature : au-delà de max_steps, truncated=True
titre("8. max_steps : l'épisode est tronqué (truncated=True) sans terminated")
e = RadiationEvacuationEnv(max_steps=5); e.reset()
for k in range(5):
    _,_,term,trunc,_ = e.step(0)   # monte en boucle, n'atteint jamais la sortie
print(f"  après 5 pas (max=5) : terminated={term}, truncated={trunc}")
assert trunc is True and term is False

print("\n========================================")
print("TOUS LES TESTS PASSENT — le MDP se comporte comme conçu.")
print("========================================")
