# TP 2 — Explorer les espaces de couleurs

Cours IMN119 — Acquisition des médias numériques. Travail individuel, 100 points.

[Ouvrir le TP2 dans Google Colab](https://colab.research.google.com/github/kyleross-dev/imn119-tp2/blob/main/TP2.ipynb)

Un seul fichier est à ouvrir : `TP2.ipynb`, identique pour Colab, VS Code et PyCharm. Les deux cellules de préparation installent les outils et leurs bibliothèques depuis la version GitHub indiquée dans le notebook, puis les importent dans le noyau actif. Git n’est pas nécessaire. L’installation et le premier chargement des images d’essai nécessitent Internet.

- Dans Colab, enregistrez une copie dans Drive et utilisez une session Python 3 sur CPU.
- Dans VS Code, ouvrez le notebook avec les extensions Python et Jupyter, puis sélectionnez un noyau Python 3.10 ou plus récent.
- Dans PyCharm avec prise en charge Jupyter, ouvrez le notebook et sélectionnez un noyau Python local, version 3.10 ou plus récente.

En local, acceptez l’installation d’`ipykernel` proposée par l’IDE, ou installez-le dans l’environnement choisi avec `python -m pip install ipykernel`. Le notebook décrit la préparation, l’import d’images personnelles par sélecteur ou par chemin, les essais sans widgets et la remise locale. Aucun fichier `.py` supplémentaire n’est à récupérer.

Pour la remise locale, enregistrez le notebook avec ses sorties et récupérez l’archive au chemin affiché. Dans Colab, les fichiers sont téléchargés.

Le paquet `imn119-tp2` fournit les interfaces RVB, CIELAB et HSV, les essais et l’export des réponses. Les fonctions de conversion à programmer restent dans le notebook. Aucun corrigé n’est inclus.

Les consignes, les crédits des images et la procédure de remise figurent dans le notebook.

## Outils, version 2.2.0

L’éditeur RVB utilise directement la fonction fournie dans le notebook :

```python
interface_rvb = outils.creer_editeur(images, "RVB", conversion=appliquer_rgb)
```

Le paramètre `conversion` est nécessaire pour les trois espaces : `appliquer_rgb` pour RVB, `rgb_to_lab` pour CIELAB et `rgb_to_hsv` pour HSV. Le module ne contient plus de seconde implémentation du réglage RVB. L’affichage sans widgets reçoit la même fonction avec `conversion=appliquer_rgb`.

La question 1 suit la démarche « prédire, tester, expliquer » : ses réponses sont des phrases organisées dans `q01`. L’export les conserve dans `reponses.json` et utilise le schéma `tp2-couleurs-2.2`. Les calculs numériques des réglages restent à la question 7. Le barème total reste de 100 points, dont 10 pour Q1.

Un **curseur de réglage** est une commande que l’on fait glisser pour changer une valeur. Les trois éditeurs proposent un bouton **Réinitialiser**, qui rétablit leurs réglages neutres sans changer l’image. Les figures sont fermées après affichage pour éviter leur accumulation pendant les essais.

Les images d’essai sont conservées dans `.cache-tp2` et réutilisées sans réseau après un premier chargement réussi. Un cache endommagé est téléchargé à nouveau. Les images personnelles transparentes sont composées sur fond blanc, en respectant leur orientation EXIF.

Les messages de vérification expliquent les erreurs de calcul courantes. L’export accepte toujours une remise partielle, mais signale une identification invalide ou des valeurs impossibles à enregistrer, notamment NaN et l’infini. Il ne calcule aucune note.

## Vérifier les outils localement

Depuis ce dossier, avec les dépendances installées dans le Python sélectionné :

```text
python -m unittest discover -s tests -v
```

Ces tests couvrent le cache hors ligne et sa réparation, les transparences, la réinitialisation des éditeurs, la fermeture des figures et l’export des réponses. Aucun corrigé des conversions à programmer n’est fourni.
