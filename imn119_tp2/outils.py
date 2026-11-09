"""Outils fournis pour le TP2 IMN119.

Les fonctions de conversion sont passées explicitement par le notebook.
Ce module ne contient pas les solutions de rgb_to_lab ou rgb_to_hsv.
"""

import io
import json
from pathlib import Path
import re
from urllib.request import Request, urlopen
from urllib.error import URLError
import zipfile

import ipywidgets as widgets
from IPython import get_ipython
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from skimage.color import hsv2rgb, lab2rgb


VERSION_TP = "tp2-couleurs-2.2"
BAREME = {'q01': 10, 'q02': 30, 'q03': 30, 'q04': 5, 'q05': 5, 'q06': 5, 'q07': 15}


def preparer():
    """Configure les figures dans Jupyter, VS Code, PyCharm et Colab."""
    shell = get_ipython()
    if shell is not None:
        shell.run_line_magic("matplotlib", "inline")
    try:
        from google.colab import output
        output.enable_custom_widget_manager()
    except ImportError:
        pass
    plt.rcParams.update({"figure.figsize": (11, 4), "axes.grid": False})
    print("Outils du TP2 prêts.")


def _lire_image(contenu, taille_maximale):
    """Respecte l’orientation EXIF et compose les transparences sur fond blanc."""
    with Image.open(io.BytesIO(bytes(contenu))) as im:
        im = ImageOps.exif_transpose(im).convert("RGBA")
        fond = Image.new("RGBA", im.size, "white")
        im = Image.alpha_composite(fond, im).convert("RGB")
        im.thumbnail((taille_maximale, taille_maximale))
        return np.asarray(im, dtype=float) / 255.0


def charger_images(dossier_cache=".cache-tp2"):
    """Charge les images d’essai; une copie locale évite de les retélécharger."""
    URL_IMAGES = {
        "Roue chromatique": "https://upload.wikimedia.org/wikipedia/commons/9/98/Color_circle_%28RGB%29.png",
        "Spectre": "https://upload.wikimedia.org/wikipedia/commons/4/44/Linear_visible_spectrum_ciecam02.png",
    }

    images = {}
    cache = Path(dossier_cache) if dossier_cache is not None else None
    for titre, url in URL_IMAGES.items():
        fichier_cache = cache / url.rsplit("/", 1)[-1] if cache is not None else None
        if fichier_cache is not None and fichier_cache.is_file():
            try:
                images[titre] = _lire_image(fichier_cache.read_bytes(), 700)
            except (OSError, ValueError):
                pass  # Un cache incomplet ou endommagé est téléchargé à nouveau.
            else:
                print(f"Image chargée : {titre} (copie locale)")
                continue
        requete = Request(url, headers={"User-Agent": "Mozilla/5.0 IMN119-TP2/1.0"})
        try:
            with urlopen(requete, timeout=30) as reponse:
                contenu = reponse.read()
            images[titre] = _lire_image(contenu, 700)
        except (URLError, OSError, ValueError) as erreur:
            raise RuntimeError(
                f"Impossible de charger l’image « {titre} ». Vérifiez votre connexion Internet, "
                "puis relancez cette cellule. Vous pouvez aussi importer une image personnelle "
                "avec outils.importer_images dans un dictionnaire images = {}."
            ) from erreur
        if fichier_cache is not None:
            try:
                fichier_cache.parent.mkdir(parents=True, exist_ok=True)
                fichier_cache.write_bytes(contenu)
            except OSError:
                pass  # Le chargement fonctionne aussi dans un dossier non modifiable.
        print(f"Image chargée : {titre}")
    return images

def _ajouter_images(images, fichiers):
    """Charge des couples (nom, octets), communs aux différents environnements."""
    for fichier, contenu in fichiers:
        try:
            images["Personnelle : " + fichier] = _lire_image(contenu, 1000)
            print("Image ajoutée :", fichier)
        except (OSError, ValueError):
            print("Image non reconnue :", fichier)


def importer_images(images, chemins=None):
    """Ajoute des images par chemins locaux ou par un sélecteur Colab/Jupyter."""
    if chemins is not None:
        if isinstance(chemins, (str, Path)):
            chemins = [chemins]
        for chemin in chemins:
            fichier = Path(chemin).expanduser()
            try:
                contenu = fichier.read_bytes()
            except OSError as erreur:
                print("Fichier inaccessible :", fichier, "—", erreur)
                continue
            _ajouter_images(images, [(fichier.name, contenu)])
        return
    try:
        from google.colab import files
    except ImportError:
        selecteur = widgets.FileUpload(accept=".png,.jpg,.jpeg", multiple=True,
                                      description="Choisir les images")

        def ajouter_selection(changement):
            selection = changement["new"]
            if isinstance(selection, dict):  # ipywidgets 7
                fichiers = [(nom, fichier["content"]) for nom, fichier in selection.items()]
            else:  # ipywidgets 8
                fichiers = [(fichier["name"], fichier["content"]) for fichier in selection]
            _ajouter_images(images, fichiers)

        selecteur.observe(ajouter_selection, names="value")
        display(selecteur)
        print("Si le sélecteur ne s’affiche pas, renseignez chemins_images dans le notebook.")
        return selecteur
    else:
        _ajouter_images(images, files.upload().items())


def verifier_tableau(resultat, original):
    assert isinstance(resultat, np.ndarray), "Renvoyez un tableau NumPy."
    assert resultat.shape == original.shape, "La forme doit rester (hauteur, largeur, 3)."
    assert np.issubdtype(resultat.dtype, np.floating), "Renvoyez des flottants."
    assert np.isfinite(resultat).all(), "Le résultat contient NaN ou une valeur infinie."

def traiter_image(tableau, espace, reglages, conversion=None):
    """Applique la fonction fournie par le notebook, puis prépare l’affichage."""
    p, q, r = reglages
    if not callable(conversion):
        raise ValueError("Passez la fonction du notebook avec conversion=appliquer_rgb, rgb_to_lab ou rgb_to_hsv.")
    if espace == "RVB":
        resultat = conversion(tableau.copy(), rouge=p, vert=q, bleu=r)
        verifier_tableau(resultat, tableau)
        if not np.all((resultat >= 0) & (resultat <= 1)):
            raise ValueError("La fonction appliquer_rgb doit borner le résultat dans [0, 1] avec np.clip.")
        canaux = [np.zeros_like(resultat) for _ in range(3)]
        for i in range(3):
            canaux[i][..., i] = resultat[..., i]
        return resultat, canaux

    if espace == "CIELAB":
        canaux = np.array(conversion(tableau.copy()), copy=True)
        verifier_tableau(canaux, tableau)
        canaux[..., 0] = np.clip(canaux[..., 0] + p, 0, 100)
        canaux[..., 1] = np.clip(canaux[..., 1] + q, -128, 127)
        canaux[..., 2] = np.clip(canaux[..., 2] + r, -128, 127)
        resultat = lab2rgb(canaux)
        gris = [canaux[..., 0] / 100, (canaux[..., 1] + 128) / 255,
                (canaux[..., 2] + 128) / 255]
    elif espace == "HSV":
        canaux = np.array(conversion(tableau.copy()), copy=True)
        verifier_tableau(canaux, tableau)
        assert np.all((canaux >= -1e-9) & (canaux <= 1 + 1e-9)), "HSV doit être normalisé entre 0 et 1."
        canaux[..., 0] = (canaux[..., 0] + p / 360.0) % 1.0
        canaux[..., 1] = np.clip(canaux[..., 1] * q, 0, 1)
        canaux[..., 2] = np.clip(canaux[..., 2] * r, 0, 1)
        resultat = hsv2rgb(canaux)
        gris = [canaux[..., i] for i in range(3)]
    else:
        raise ValueError("Espace inconnu.")
    return np.clip(resultat, 0, 1), [np.clip(c, 0, 1) for c in gris]

def afficher_comparaison(images, espace, image="Roue chromatique", reglages=None, conversion=None):
    """Affichage statique, utilisable même sans widgets."""
    if reglages is None:
        reglages = (0, 0, 0) if espace == "CIELAB" else ((0, 1, 1) if espace == "HSV" else (1, 1, 1))
    tableau = images[image]
    try:
        resultat, canaux = traiter_image(tableau, espace, reglages, conversion)
    except NotImplementedError:
        print("Complétez d’abord la fonction de conversion pour", espace)
        return
    noms = {"RVB": ["Rouge", "Vert", "Bleu"],
            "CIELAB": ["L*", "a*", "b*"], "HSV": ["H (teinte)", "S (saturation)", "V (valeur)"]}[espace]
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    axes[0, 0].imshow(tableau); axes[0, 0].set_title("Image originale")
    axes[0, 1].imshow(resultat); axes[0, 1].set_title("Image traitée — " + espace)
    axes[0, 2].text(0.05, 0.7, "Réglages :\n" + "\n".join(
        f"{n} : {v:g}" for n, v in zip(noms, reglages)), transform=axes[0, 2].transAxes)
    for ax, canal, titre in zip(axes[1], canaux, noms):
        ax.imshow(canal, cmap="gray", vmin=0, vmax=1)
        ax.set_title("Canal " + titre)
    for ax in axes.flat:
        ax.axis("off")
    fig.tight_layout(); plt.show()
    plt.close(fig)

def creer_editeur(images, espace, conversion=None):
    """Crée l’interface avec les widgets standards d’ipywidgets.

    conversion : appliquer_rgb, rgb_to_lab ou rgb_to_hsv, définie dans le notebook.
    Relancez l’appel après avoir redéfini cette fonction dans le notebook.
    """
    if not images:
        raise ValueError("Chargez ou importez au moins une image avant de lancer l’éditeur.")
    if not callable(conversion):
        raise ValueError("Passez la fonction du notebook avec le paramètre conversion.")
    specifications = {
        "RVB": [("Rouge ×", 0, 2, .01, 1), ("Vert ×", 0, 2, .01, 1), ("Bleu ×", 0, 2, .01, 1)],
        "CIELAB": [("L* +", -50, 50, 1, 0), ("a* +", -50, 50, 1, 0), ("b* +", -50, 50, 1, 0)],
        "HSV": [("H + (degrés)", -180, 180, 1, 0), ("S ×", 0, 2, .01, 1), ("V ×", 0, 2, .01, 1)]}
    specs = specifications[espace]
    choix = widgets.Dropdown(options=list(images), description="Image :")
    curseurs = [widgets.FloatSlider(description=t, min=lo, max=hi, step=pas, value=v,
                continuous_update=False, style={"description_width": "110px"},
                layout=widgets.Layout(width="420px", max_width="100%")) for t, lo, hi, pas, v in specs]
    sortie = widgets.Output()
    reinitialiser = widgets.Button(description="Réinitialiser", icon="undo")
    mise_a_jour_suspendue = False

    def actualiser(changement=None):
        if mise_a_jour_suspendue:
            return
        with sortie:
            sortie.clear_output(wait=True)
            try:
                afficher_comparaison(images, espace, choix.value,
                                     tuple(c.value for c in curseurs), conversion)
            except Exception as erreur:
                print(type(erreur).__name__ + " : " + str(erreur))
                print("Vérifiez la fonction, puis relancez la cellule de l’éditeur.")

    def remettre_a_zero(bouton):
        nonlocal mise_a_jour_suspendue
        mise_a_jour_suspendue = True
        try:
            for curseur, specification in zip(curseurs, specs):
                curseur.value = specification[-1]
        finally:
            mise_a_jour_suspendue = False
        actualiser()

    choix.observe(actualiser, names="value")
    for curseur in curseurs:
        curseur.observe(actualiser, names="value")
    reinitialiser.on_click(remettre_a_zero)
    display(widgets.VBox([choix, *curseurs, reinitialiser, sortie]))
    actualiser()
    return {"image": choix, "curseurs": curseurs, "reinitialiser": reinitialiser}


def executer_essai(nom_essai, fonction):
    try:
        fonction()
    except NotImplementedError:
        print(nom_essai + " : fonction à compléter.")
    except AssertionError as erreur:
        print(nom_essai + " : vérification échouée — " + str(erreur))
    except (TypeError, ValueError, IndexError, FloatingPointError, ZeroDivisionError) as erreur:
        print(nom_essai + " : " + type(erreur).__name__ + " — " + str(erreur))
        print("Vérifiez la forme du tableau, les types numériques et les divisions, puis relancez les essais.")
    else:
        print(nom_essai + " : essais fournis réussis (vérification non exhaustive).")

def _essai_lab(conversion):
    couleurs = np.array([[[0., 0., 0.], [1., 1., 1.], [1., 0., 0.]],
                         [[0., 1., 0.], [0., 0., 1.], [0.5, 0.5, 0.5]]])
    copie = couleurs.copy()
    resultat = conversion(couleurs)
    verifier_tableau(resultat, couleurs)
    assert np.array_equal(couleurs, copie), "Ne modifiez pas l’entrée."
    attendu = np.array([[[0., 0., 0.], [100., 0.0053, -0.0104], [53.2329, 80.1093, 67.2201]],
                        [[87.7370, -86.1846, 83.1812], [32.3026, 79.1967, -107.8637],
                         [53.3890, 0.0031, -0.0062]]])
    assert np.allclose(resultat, attendu, atol=0.08, rtol=0), "Vérifiez la linéarisation, XYZ et le blanc D65."
    sombre = np.array([[[0.01, 0.01, 0.01]]])
    assert np.allclose(conversion(sombre), [[[0.6991, 0.0001, -0.0002]]], atol=0.01), "Vérifiez les branches pour les faibles valeurs."
    print("CIELAB pour noir, blanc, rouge / vert, bleu, gris :")
    print(np.round(resultat, 3))

def tester_lab(conversion):
    executer_essai('CIELAB', lambda: _essai_lab(conversion))

def _essai_hsv(conversion):
    couleurs = np.array([[[1., 0., 0.], [1., 1., 0.], [0., 1., 0.]],
                         [[0., 1., 1.], [0., 0., 1.], [1., 0., 1.]],
                         [[0., 0., 0.], [1., 1., 1.], [0.5, 0.5, 0.5]]])
    copie = couleurs.copy()
    with np.errstate(divide="raise", invalid="raise"):
        resultat = conversion(couleurs)
    verifier_tableau(resultat, couleurs)
    assert np.array_equal(couleurs, copie), "Ne modifiez pas l’entrée."
    attendu = np.array([[[0., 1., 1.], [1/6, 1., 1.], [2/6, 1., 1.]],
                        [[3/6, 1., 1.], [4/6, 1., 1.], [5/6, 1., 1.]],
                        [[0., 0., 0.], [0., 0., 1.], [0., 0., 0.5]]])
    assert np.allclose(resultat, attendu, atol=1e-7), "Vérifiez les teintes, les unités et les pixels gris."
    cas = conversion(np.array([[[0.8, 0.4, 0.2]]]))
    assert np.allclose(cas, [[[0.0530739, 0.75, 0.8]]], atol=1e-6), "Appliquez la formule par arccos de l’algorithme B."
    print("HSV pour six couleurs et trois niveaux de gris :")
    print(np.round(resultat, 4))

def tester_hsv(conversion):
    executer_essai('HSV', lambda: _essai_hsv(conversion))

def creer_remise(nom, cip, reponses):
    if (not isinstance(nom, str) or not nom.strip() or not isinstance(cip, str)
            or not re.fullmatch(r"[a-z0-9]{1,40}", cip)):
        print("Complétez votre nom et votre CIP en minuscules, sans espace, dans REPONSE_IDENTITE.")
        return None

    marqueurs = tuple(f"REPONSE_Q{i:02d}" for i in range(1, 8))
    sources = {}
    shell = get_ipython()
    if shell is None or not hasattr(shell, "history_manager"):
        print("Créez la remise depuis le notebook après avoir exécuté les cellules de réponse.")
        return None
    for source in shell.history_manager.input_hist_raw:
        premiere = source.strip().splitlines()[0] if source.strip() else ""
        for marqueur in marqueurs:
            if premiere == "# " + marqueur:
                sources[marqueur] = source
    absentes = [m for m in marqueurs if m not in sources]
    if absentes:
        print("Exécutez les cellules de réponse en conservant leur marqueur :", ", ".join(absentes))
        return None

    def vers_json(v):
        if isinstance(v, np.ndarray):
            return vers_json(v.tolist())
        if isinstance(v, np.generic):
            return v.item()
        if isinstance(v, dict):
            return {k: vers_json(x) for k, x in v.items()}
        if isinstance(v, (tuple, list)):
            return [vers_json(x) for x in v]
        return v

    reponses = dict(reponses)
    reponses.update({"q02": {"fonction": "rgb_to_lab", "fichier": "conversions.py"},
                     "q03": {"fonction": "rgb_to_hsv", "fichier": "conversions.py"}})
    donnees = {"schema_version": VERSION_TP, "nom": nom,
               "cip": cip, "bareme": BAREME, "reponses": vers_json(reponses)}
    entete = "import numpy as np\n\n"
    toutes_reponses = entete + "\n\n".join(sources[m] for m in marqueurs)
    conversions = entete + "\n\n".join(sources[m] for m in ("REPONSE_Q02", "REPONSE_Q03"))
    try:
        texte_json = json.dumps(donnees, ensure_ascii=False, indent=2, allow_nan=False)
    except (TypeError, ValueError):
        print("Archive non créée : une réponse contient un type non exportable ou une valeur NaN/infinie.")
        print("Utilisez des nombres finis, des booléens, des textes, des listes ou None pour une réponse manquante.")
        return None
    contenu = {
        "reponses.json": texte_json,
        "conversions.py": conversions,
        "reponses.py": toutes_reponses,
    }
    chemin_archive = Path(f"TP2_{cip}.zip")
    with zipfile.ZipFile(chemin_archive, "w", zipfile.ZIP_DEFLATED) as z:
        for fichier, texte in contenu.items():
            z.writestr(fichier, texte.encode("utf-8"))
    print(f"Archive prête : {chemin_archive}. Téléchargez aussi le notebook exécuté.")
    print("L’export conserve vos réponses; il ne vérifie pas leur exactitude.")
    return chemin_archive

def telecharger(archive):
    if archive is None:
        print("Corrigez les messages affichés lors de la création de la remise, puis relancez l’export.")
        return
    try:
        from google.colab import files
    except ImportError:
        from IPython.display import FileLink
        print("Archive enregistrée :", Path(archive).resolve())
        display(FileLink(str(archive)))
    else:
        files.download(str(archive))

