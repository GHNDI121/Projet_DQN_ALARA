# Navigation d'un agent en zone irradiée par Reinforcement Learning (principe ALARA)

> Projet de Master 2 — Intelligence Artificielle · Cours de Reinforcement Learning

Un agent apprend, par essais-erreurs, à évacuer une zone contenant des sources
radioactives en suivant le chemin qui **minimise la dose de radiation reçue** tout en
atteignant la sortie dans un temps raisonnable — l'application directe du principe
**ALARA** (*As Low As Reasonably Achievable*). L'algorithme central sera un
**Deep Q-Network (DQN)**, comparé à une politique aléatoire et à un Q-learning tabulaire,
avec monitoring Weights & Biases et une démo interactive optionnelle.

---

## Table des matières

1. [Idée du projet](#1-idée-du-projet)
2. [Origine et simulation des données](#2-origine-et-simulation-des-données)
3. [Le problème formalisé (MDP)](#3-le-problème-formalisé-mdp)
4. [La géométrie de l'environnement](#4-la-géométrie-de-lenvironnement)
5. [Structure du dépôt](#5-structure-du-dépôt)
6. [Les cinq étapes du projet](#6-les-cinq-étapes-du-projet)
7. [Répartition des tâches](#7-répartition-des-tâches)
8. [Installation](#8-installation)
9. [Références](#9-références)

---

## 1. Idée du projet

Dans une évacuation d'urgence (accident en laboratoire, incident en installation
nucléaire), le chemin le plus court n'est pas toujours le plus sûr : filer tout droit
peut traverser une zone très irradiée, tandis qu'un détour réduit la dose au prix de
quelques secondes. On simule ce dilemme dans une grille 2D et on entraîne un agent à
trouver le bon compromis — sans jamais lui montrer la solution.

En Reinforcement Learning, **il n'y a aucun jeu de données à télécharger** : les données
d'entraînement (les transitions état → action → récompense → nouvel état) sont générées
par l'agent lui-même en interagissant avec l'environnement simulé.

---

## 2. Origine et simulation des données

Spécificité à comprendre : **nous ne simulons pas un jeu de données, nous simulons un
environnement qui produit les données par interaction.**

En apprentissage supervisé, on part d'un fichier d'exemples déjà étiquetés. Ici, un tel
fichier n'existe pas : il n'y a aucune base de « bons chemins d'évacuation en zone
irradiée » à télécharger. Les données sont **générées à la volée** par l'agent.

L'environnement est un **simulateur déterministe à base physique**, de type *grid world*,
conforme à l'interface Gymnasium :

1. **Modèle physique de la dose** : loi de l'inverse du carré de la distance,
   `dose(p) = Σₖ Iₖ / (dₖ² + 1)`, sommée sur les sources.
2. **Dynamique déterministe** : une action mène toujours au même état suivant (le seul
   hasard vient de l'exploration ε-greedy de l'agent, pas de l'environnement).
3. **Signal de récompense calculé** : `r = −λ·(dose/dose_max) − 1` (+100 à la sortie).

À chaque pas, l'interaction produit une **transition** `(état, action, récompense, état
suivant, terminé)` — la brique de donnée élémentaire du RL. Sur un entraînement complet,
l'agent génère des centaines de milliers de transitions, sans aucun fichier externe.

---

## 3. Le problème formalisé (MDP)

| Composante | Définition |
|---|---|
| **État** | Position de l'agent sur la grille 15×15. Vecteur `(x/(N-1), y/(N-1))` pour le DQN ; entier `state_id` pour le Q-learning tabulaire. |
| **Actions** | 4 déplacements : `0`=haut, `1`=bas, `2`=gauche, `3`=droite. |
| **Récompense** | `r = −λ·(dose_du_pas / dose_max) − 1`, plus `+100` à la sortie. |
| **Transition** | Nouvelle position après l'action (déterministe ; un mur laisse l'agent sur place mais le pas est compté). |

Le paramètre **λ** est le curseur ALARA : petit, l'agent privilégie la vitesse ; grand,
il fait de larges détours pour éviter la dose. Faire varier λ et observer le changement
de comportement sera l'analyse centrale du projet.

---

## 4. La géométrie de l'environnement

- Grille **15×15** (225 états).
- Départ **(7,0)** → sortie **(7,14)** : traversée par le milieu. Choix délibéré —
  un départ/sortie en coins opposés rendrait l'évitement des sources gratuit (tous les
  chemins auraient la même longueur) et ferait disparaître le compromis.
- 3 sources : **(7,7)=60**, **(5,8)=25**, **(9,6)=25** µSv/h. La source centrale barre le
  chemin direct ; les deux autres, décalées, créent une asymétrie haut/bas.
- Murs : **(2,5),(3,5)** et **(11,9),(12,9)** — à l'écart, pour du relief sans bloquer.

---

## 5. Structure du dépôt (cible)

```
projet-alara-rl/
├── env_alara.py            # Environnement + MDP (étapes 1-2)
├── dqn.py                  # QNetwork, ReplayBuffer, DQNAgent (étape 3)
├── train.py                # Entraînement + sauvegarde du meilleur modèle
├── monitoring.py           # Journalisation Weights & Biases (étape 4)
├── baselines.py            # Politique aléatoire + Q-learning tabulaire (comparaisons)
├── evaluate.py             # Banc de comparaison des 3 agents
├── carte_dose.py           # Figure de la carte de dose
├── app_gradio.py / app_streamlit.py   # Démo interactive (étape 5, optionnelle)
├── test_*.py               # Tests unitaires par brique
├── requirements.txt        # Dépendances
└── README.md               # Ce fichier
```

Principe : **un fichier = une responsabilité**.
**INTERFACE.md** fige le contrat de l'environnement (signatures de `reset`/`step`, forme
de l'observation) — à respecter par tous pour travailler en parallèle.

---

## 6. Les cinq étapes du projet

1. **Get the data** — l'environnement de simulation génère les transitions. *(`env_alara.py`)*
2. **Build the MDP** — le problème formalisé comme classe Python. *(`env_alara.py`)*
3. **Class DQN** — réseau de neurones + replay buffer + fonction d'entraînement. *(`dqn.py`, `train.py`)*
4. **Monitoring** — courbes Weights & Biases : value loss, mean reward, dose (radiation),
   success rate. *(`monitoring.py`)*
5. **Déploiement** *(optionnel)* — démo interactive Gradio / Streamlit.

**Comparaisons imposées** : DQN vs politique aléatoire vs Q-learning tabulaire.

---

## 7. Répartition des tâches (groupe de 4)

| Membre | Rôle | Périmètre |
|---|---|---|
| **Membre 1** | Environnement & données | `env_alara.py`, carte de dose, tests, garde-fou. Livre l'interface en premier. |
| **Membre 2** | Agent DQN | `dqn.py` : réseau, replay buffer, target network, entraînement. |
| **Membre 3** | Baselines & évaluation | `baselines.py`, `evaluate.py` : aléatoire, Q-learning, métriques, étude λ. |
| **Membre 4** | Monitoring & intégration | `monitoring.py`, démo, coordination Git, rapport. |

Calendrier indicatif : **3 semaines** (S1 environnement + squelettes, S2 entraînement +
monitoring + comparaisons, S3 analyse + rapport + soutenance).

---

## 8. Installation

```bash
pip install -r requirements.txt
```

Dépendances prévues : `gymnasium`, `numpy`, `matplotlib`, `torch`, `wandb`, et
`gradio` / `streamlit` (pour la démo optionnelle).

---

## 9. Références

- Sutton & Barto, *Reinforcement Learning: An Introduction* (2018) — MDP, Q-learning.
- Mnih et al. (2015), *Human-level control through deep reinforcement learning*, Nature — DQN.
- CIPR Publication 103 (2007) ; AIEA GSR Part 3 (2014) — radioprotection & principe ALARA.
- Outils : Python, Gymnasium, PyTorch, Weights & Biases.
