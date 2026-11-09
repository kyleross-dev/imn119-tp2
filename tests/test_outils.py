import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.error import URLError
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from imn119_tp2 import outils


class OutilsTests(unittest.TestCase):
    def setUp(self):
        self.temporaire = tempfile.TemporaryDirectory()
        self.dossier = Path(self.temporaire.name)
        self.sortie = io.StringIO()
        self.capture = contextlib.redirect_stdout(self.sortie)
        self.capture.__enter__()

    def tearDown(self):
        plt.close('all')
        self.capture.__exit__(None, None, None)
        self.temporaire.cleanup()

    def image_png(self, couleur='red'):
        tampon = io.BytesIO()
        Image.new('RGBA', (12, 8), couleur).save(tampon, format='PNG')
        return tampon.getvalue()

    def test_image_transparente_sur_fond_blanc(self):
        images = {}
        outils._ajouter_images(images, [('transparence.png', self.image_png((255, 0, 0, 0)))])
        self.assertTrue(np.all(images['Personnelle : transparence.png'] == 1))

    def test_cache_images_utilisable_sans_reseau(self):
        with patch.object(outils, 'urlopen', side_effect=lambda *a, **k: io.BytesIO(self.image_png())) as reseau:
            premieres = outils.charger_images(self.dossier)
        self.assertEqual(reseau.call_count, 2)
        with patch.object(outils, 'urlopen', side_effect=URLError('hors ligne')) as reseau:
            suivantes = outils.charger_images(self.dossier)
        reseau.assert_not_called()
        for nom in premieres:
            np.testing.assert_array_equal(premieres[nom], suivantes[nom])

    def test_cache_corrompu_telecharge_a_nouveau(self):
        with patch.object(outils, 'urlopen', side_effect=lambda *a, **k: io.BytesIO(self.image_png())):
            outils.charger_images(self.dossier)
        prochain = next(self.dossier.iterdir())
        prochain.write_bytes(b'image invalide')
        with patch.object(outils, 'urlopen', side_effect=lambda *a, **k: io.BytesIO(self.image_png())) as reseau:
            images = outils.charger_images(self.dossier)
        self.assertEqual(reseau.call_count, 1)
        self.assertEqual(len(images), 2)

    def test_echec_reseau_explique_la_cellule_a_relancer(self):
        with patch.object(outils, 'urlopen', side_effect=URLError('hors ligne')):
            with self.assertRaisesRegex(RuntimeError, 'connexion Internet'):
                outils.charger_images(dossier_cache=None)

    def test_chargement_sans_droit_ecriture_cache(self):
        with patch.object(outils, 'urlopen', side_effect=lambda *a, **k: io.BytesIO(self.image_png())), \
                patch.object(Path, 'write_bytes', side_effect=PermissionError('lecture seule')):
            self.assertEqual(len(outils.charger_images(self.dossier)), 2)

    def test_reinitialisation_des_trois_espaces(self):
        for espace, neutres in [('RVB', (1., 1., 1.)), ('CIELAB', (0., 0., 0.)), ('HSV', (0., 1., 1.))]:
            with self.subTest(espace=espace), patch.object(outils, 'afficher_comparaison') as afficher, \
                    patch.object(outils, 'display'):
                editeur = outils.creer_editeur({'essai': np.zeros((2, 2, 3))}, espace)
                for curseur in editeur['curseurs']:
                    curseur.value = curseur.max
                avant = afficher.call_count
                editeur['reinitialiser'].click()
                self.assertEqual(tuple(c.value for c in editeur['curseurs']), neutres)
                self.assertEqual(afficher.call_count, avant + 1)
                self.assertEqual(editeur['image'].value, 'essai')

    def test_editeur_sans_image(self):
        with self.assertRaisesRegex(ValueError, 'au moins une image'):
            outils.creer_editeur({}, 'RVB')

    def test_figures_ne_s_accumulent_pas(self):
        avant = plt.get_fignums()
        with patch.object(plt, 'show'):
            for _ in range(3):
                outils.afficher_comparaison({'Roue chromatique': np.zeros((2, 2, 3))}, 'RVB')
        self.assertEqual(plt.get_fignums(), avant)

    def historique(self):
        sources = [f'# REPONSE_Q{i:02d}\nvaleur = None' for i in range(1, 8)]
        sources.append('# REPONSE_Q01\nvaleur = "dernière version"')
        return SimpleNamespace(history_manager=SimpleNamespace(input_hist_raw=sources))

    def test_remise_partielle_et_derniere_execution(self):
        precedent = Path.cwd()
        try:
            os.chdir(self.dossier)
            with patch.object(outils, 'get_ipython', return_value=self.historique()):
                archive = outils.creer_remise('Essai', 'test00', {'q01': {'neutre': None}})
            with zipfile.ZipFile(archive) as z:
                donnees = json.loads(z.read('reponses.json'))
                self.assertIsNone(donnees['reponses']['q01']['neutre'])
                reponses = z.read('reponses.py').decode('utf-8')
                self.assertIn('dernière version', reponses)
                self.assertEqual(reponses.count('# REPONSE_Q01'), 1)
        finally:
            os.chdir(precedent)

    def test_valeur_non_finie_ne_cree_pas_archive(self):
        precedent = Path.cwd()
        try:
            os.chdir(self.dossier)
            with patch.object(outils, 'get_ipython', return_value=self.historique()):
                self.assertIsNone(outils.creer_remise('Essai', 'test00', {'q01': float('nan')}))
            self.assertFalse(list(self.dossier.glob('*.zip')))
            self.assertIn('NaN', self.sortie.getvalue())
        finally:
            os.chdir(precedent)

    def test_identification_invalide_sans_traceback(self):
        self.assertIsNone(outils.creer_remise(123, 'test00', {}))
        self.assertIsNone(outils.creer_remise('Essai', 123, {}))

    def test_essai_erreur_de_calcul_expliquee(self):
        outils.executer_essai('CIELAB', lambda: 1 / 0)
        self.assertIn('ZeroDivisionError', self.sortie.getvalue())
        self.assertIn('divisions', self.sortie.getvalue())


if __name__ == '__main__':
    unittest.main()
