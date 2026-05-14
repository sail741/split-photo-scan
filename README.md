# SplitPhotoScan - Extracteur de photos

Application qui extrait automatiquement les photos individuelles depuis des scans A4 contenant plusieurs photos.

## Fonctionnalités

- Détection automatique des photos sur un scan, même si elles sont en diagonal
- Redressement des photos via transformation de perspective
- Recadrage intelligent pour supprimer les marges blanches
- Traitement par lot d'un dossier entier de scans
- Interface graphique simple (sélection de dossier + barre de progression)
- Export en JPG ou PNG

## Utilisation

### Interface graphique (Windows)

Télécharger `SplitPhotoScan.exe` depuis la [page des releases](https://github.com/sail741/split-photo-scan/releases) et le lancer. Aucune installation requise.

### Ligne de commande

```bash
pip install -r requirements.txt

# Un seul scan
python extract_photos.py scan.jpg

# Un dossier entier
python extract_photos.py ./mes_scans/ -o photos_extraites/

# Mode debug (visualise les zones détectées)
python extract_photos.py scan.jpg --debug
```

---

Réalisé avec l'assistance de Claude Code.
