# Diagnostic de la sécheresse - Bassin de la Sauer

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://geolab2026a09741389.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-En_Développement-green)]()

**Application interactive permettant de cartographier et de diagnostiquer la vulnérabilité à la sécheresse sur le bassin versant de la Sauer (France) via une analyse multicritère spatiale.**

---

## Présentation

Ce projet est une application web développée avec **Streamlit** qui implémente la méthode **AHP (Analytic Hierarchy Process)** de Saaty combinée à de la logique floue (Fuzzy Logic). Elle permet aux gestionnaires et hydrologues de :

1.  **Visualiser** le risque de sécheresse sur différentes périodes clés (Mai 2022, Juin 2022, Octobre 2023).
2.  **Moduler** l'importance des facteurs environnementaux via une matrice de décision interactive.
3.  **Comparer** spatialement l'évolution de la sécheresse entre deux dates.
4.  **Exporter** les résultats pour des rapports (PNG) ou pour une intégration SIG (GeoTIFF).

## Fonctionnalités clés

*   **Analyse multicritère dynamique** : Éditeur de matrice de Saaty intégré. Modifiez les poids des critères et voyez le résultat en temps réel avec calcul instantané du *Ratio de Cohérence (RC)*.
*   **Cartographie interactive** :
    *   Rendu en 5 classes (très faible à très fort).
    *   Rendu continu (0-1) pour voir les gradients.
*   **Comparateur temporel** : Mode "Swipe" (avant/après) pour analyser les changements entre deux mois.
*   **Statistiques et dashboard** : Graphiques de répartition (Camemberts/Barres) pour quantifier les surfaces touchées par niveau de risque.
*   **Export de Données** :
    *   **GeoTIFF** : Données brutes géoréférencées.
    *   **PNG** : Cartes mises en page avec légende, flèche nord et échelle.

## Méthodologie et indicateurs

Le calcul de vulnérabilité repose sur la combinaison spatiale de **7 indicateurs majeurs** :

| Catégorie | Indicateur | Description |
| :--- | :--- | :--- |
| **Climat** | **Précipitations** | Cumul mensuel. |
| | **ETP** | Évapotranspiration Potentielle. |
| **Végétation** | **NDDI** | Normalized Difference Drought Index. |
| | **OCS** | Occupation du Sol. |
| **Sol / Terrain** | **Texture Sol** | Capacité de rétention en eau. |
| | **Pente** | Influence sur le ruissellement vs l'infiltration. |
| | **Distance à l'eau** | Proximité au réseau hydrographique. |

## Installation locale

Pour faire tourner l'application sur votre machine :

1.  **Cloner le dépôt :**
    ```bash
    git clone https://github.com/votre-user/nom-du-repo.git
    cd nom-du-repo
    ```

2.  **Créer un environnement virtuel (recommandé) :**
    ```bash
    python -m venv venv
    # Windows :
    venv\Scripts\activate
    # Mac/Linux :
    source venv/bin/activate
    ```

3.  **Installer les dépendances :**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Lancer l'application :**
    ```bash
    streamlit run app.py
    ```

> **Note sur les données :** L'application télécharge automatiquement les couches raster (GeoTIFF) depuis un serveur distant (Seafile Unistra) lors du premier lancement. Le démarrage initial peut prendre quelques minutes selon votre connexion.

## Architecture technique

*   **Interface :** [Streamlit](https://streamlit.io/)
*   **Traitement géospatial :** `Rasterio`, `Geopandas`
*   **Calculs matriciels :** `Numpy`, `Pandas`
*   **Visualisation :** `Matplotlib` (cartes), `Altair` (stats), `Streamlit Image Comparison`(comparaison)
*   **Données :** Hébergement distant pour contourner la limite de taille GitHub.

## Auteurs

Projet réalisé dans le cadre du **Master GEOLAB**.

---
*N'hésitez pas à ouvrir une Issue pour toute suggestion d'amélioration !*
