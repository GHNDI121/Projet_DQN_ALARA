"""
Garde-fou d'atteignabilité (BFS).
Vérifie qu'un chemin départ -> sortie existe malgré les murs.
Si un jour quelqu'un déplace un mur et enferme la sortie, ce test échoue
immédiatement au lieu de laisser l'entraînement tourner dans le vide.
"""
from collections import deque
from env_alara import RadiationEvacuationEnv


def chemin_existe(env):
    """BFS pur : renvoie (existe, longueur_min_en_pas)."""
    start, goal = env.start, env.goal
    if start in env.obstacles or goal in env.obstacles:
        return False, None
    vus = {start}
    file = deque([(start, 0)])
    while file:
        (i, j), d = file.popleft()
        if (i, j) == goal:
            return True, d
        for di, dj in env.ACTIONS.values():
            nxt = (i + di, j + dj)
            if env._valid(nxt) and nxt not in vus:
                vus.add(nxt)
                file.append((nxt, d + 1))
    return False, None


def compte_cases_accessibles(env):
    """Combien de cases sont atteignables depuis le départ (diagnostic)."""
    vus = {env.start}
    file = deque([env.start])
    while file:
        i, j = file.popleft()
        for di, dj in env.ACTIONS.values():
            nxt = (i + di, j + dj)
            if env._valid(nxt) and nxt not in vus:
                vus.add(nxt)
                file.append(nxt)
    return len(vus)


if __name__ == "__main__":
    env = RadiationEvacuationEnv()
    libres = env.size * env.size - len(env.obstacles)

    print("--- Garde-fou d'atteignabilité ---")
    existe, dmin = chemin_existe(env)
    print(f"  Chemin départ {env.start} -> sortie {env.goal} : "
          f"{'OUI' if existe else 'NON'}")
    assert existe, "ÉCHEC : aucun chemin vers la sortie — un mur bloque tout !"
    print(f"  Longueur du plus court chemin : {dmin} pas "
          f"(borne physique : le meilleur temps possible)")

    accessibles = compte_cases_accessibles(env)
    print(f"  Cases accessibles depuis le départ : {accessibles} / {libres} libres")
    assert accessibles == libres, \
        f"ATTENTION : {libres - accessibles} case(s) enfermée(s), inatteignables."
    print("  Aucune zone enfermée : toute la grille est explorable.")

    # Test de robustesse : on vérifie que le garde-fou SAIT détecter un blocage.
    print("\n--- Vérification que le garde-fou détecte bien un blocage ---")
    piege = RadiationEvacuationEnv()
    piege.obstacles = {(7, 13), (6, 14), (8, 14)}   # on emmure la sortie (7,14)
    existe2, _ = chemin_existe(piege)
    print(f"  Sortie volontairement emmurée -> chemin existe ? {existe2} (attendu : False)")
    assert existe2 is False, "Le garde-fou aurait dû détecter le blocage."
    print("  Le garde-fou réagit correctement.")

    print("\n========================================")
    print("ATTEIGNABILITÉ OK — la sortie est toujours joignable.")
    print("========================================")
