import streamlit as st
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.features import rasterize
import os, requests, re, io
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
from matplotlib.colors import ListedColormap, BoundaryNorm
import contextily as ctx 
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from streamlit_image_comparison import image_comparison

# --- 1. CONFIGURATION PAGE & NOUVEAU STYLE PRO ---
st.set_page_config(layout="wide", page_title="Sauer - OAD sécheresse", page_icon="💧")

st.markdown("""
<style>
/* Fond général */
.main { background-color:#f8fafc; }
h1 { font-size:2.2rem; color: #0f172a; font-weight: 800; }
h2 { font-size:1.5rem; color: #334155; }
.block-container { padding-top:1rem; }

/* STYLE SIDEBAR MODERNE */
section[data-testid="stSidebar"] {
    background-color: #000000;
    border-right: 1px solid #e2e8f0;
    box-shadow: 4px 0 15px rgba(0,0,0,0.03);
}
section[data-testid="stSidebar"] h2 {
    color: #3290c7 !important;
    font-size: 1.0rem !important;
    text-transform: uppercase;
    font-weight: 800 !important;
    letter-spacing: 0.05em;
    border-bottom: 2px solid #bfdbfe;
    padding-bottom: 0.5rem;
    margin-bottom: 1rem !important;
}
section[data-testid="stSidebar"] .stRadio label, 
section[data-testid="stSidebar"] .stCheckbox span,
section[data-testid="stSidebar"] .stSelectbox label {
    color: #84aae0 !important;
    font-weight: 500;
}
div[data-testid="stSidebarUserContent"] {
    padding-left: 0.5rem;
    padding-right: 0.5rem;
}

/* NOUVEAU STYLE POUR LA BOÎTE MÉTRIQUE (CR) */
.metric-box {
    background-color: white !important;
    border-radius: 8px;
    padding: 0.5rem 1.5rem;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px -1px rgba(0, 0, 0, 0.1);
    border: 1px solid #e2e8f0;
    border-left: 4px solid #2563eb;
    text-align: left;
    margin-bottom: 1rem;
}

.metric-label {
    color: #64748b !important;
    font-size: 0.875rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.25rem;
}

.metric-value {
    color: #0f172a !important;
    font-size: 2.5rem;
    font-weight: 700;
    line-height: 1;
}
</style>
""", unsafe_allow_html=True)

# --- 2. FONCTIONS LOGIQUE FLOU (IDENTIQUE NOTEBOOK) ---
def fuzzy_smf(arr, a, b):
    """Fonction S-Shape croissante (Bloc 7 du notebook)"""
    x = arr.astype(np.float32)
    mu = np.ones_like(x)
    mu[x <= a] = 0.0
    mid = (a + b) / 2.0
    denom = (b - a) if (b - a) != 0 else 1e-12
    mask1 = (x > a) & (x <= mid)
    mu[mask1] = 2.0 * ((x[mask1] - a) / denom)**2
    mask2 = (x > mid) & (x < b)
    mu[mask2] = 1.0 - 2.0 * ((x[mask2] - b) / denom)**2
    mu[np.isnan(x)] = np.nan
    return mu

# --- 3. CONFIGURATION DES DONNÉES ---
FIXED_FILES = {
    "climat": "https://seafile.unistra.fr/f/5f080dd56cb6413a8756/?dl=1",
    "dist_eau": "https://seafile.unistra.fr/f/c7fa36532cee4eb1a679/?dl=1",
    "pente": "https://seafile.unistra.fr/f/fff9931fdfc844f0ba06/?dl=1",
    "sol": "https://seafile.unistra.fr/f/fbdd45ab063e4cbb9390/?dl=1",
    "mask_bati": "https://seafile.unistra.fr/f/bcc6dded1f41440aaa22/?dl=1"
}

PERIODS_CONFIG = {
    "Mai 2022 (Période sèche)": {
        "slug": "mai_22",
        "etp": "https://seafile.unistra.fr/f/4593ba33e7284936b3fc/?dl=1",
        "nddi": "https://seafile.unistra.fr/f/60cf50014fe74d319a0f/?dl=1",
        "OCS": "https://seafile.unistra.fr/f/8933db4f9b8742f68391/?dl=1",
        "precip": "https://seafile.unistra.fr/f/8fab59fed5114ee68330/?dl=1"
    },
    "Juin 2022 (Période sèche à fortes pluies)": {
        "slug": "juin_22",
        "etp": "https://seafile.unistra.fr/f/bbf9c9d3c9c24ecc8d8f/?dl=1",
        "nddi": "https://seafile.unistra.fr/f/c12d0077d1d747a2b35b/?dl=1",
        "OCS": "https://seafile.unistra.fr/f/20f1807848754f91870b/?dl=1",
        "precip": "https://seafile.unistra.fr/f/628cd9f2016b46d4b1f1/?dl=1"
    },
    "Octobre 2023 (Période normale)": {
        "slug": "oct_23",
        "etp": "https://seafile.unistra.fr/f/11036c976e7d452e9931/?dl=1",
        "nddi": "https://seafile.unistra.fr/f/5369b66bffb34f4d8e59/?dl=1",
        "OCS": "https://seafile.unistra.fr/f/e1b98c1698cf4f11be58/?dl=1",
        "precip": "https://seafile.unistra.fr/f/f8763d11a6634062b215/?dl=1"
    }
}

TEXTURE_SCORE = {"S":1, "Sa":0.9, "SS":1, "LSa":0.35, "Sal":0.75, "Sl":0.7, "L":0, "Ls":0.1, "La":0.25, "LAS":0.30, "Als":0.75, "Al":0.40, "A":0.85}

# --- 4. FONCTIONS TECHNIQUES ---
def smart_download(url, filename):
    if not os.path.exists("data"): os.makedirs("data")
    path = os.path.join("data", filename)
    if os.path.exists(path) and os.path.getsize(path) < 1000: os.remove(path)
    if not os.path.exists(path):
        r = requests.get(url, stream=True)
        if r.status_code == 200:
            with open(path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024*1024): f.write(chunk)
    return path

def align_strict(path, ref_prof, resampling=Resampling.bilinear):
    with rasterio.open(path) as src:
        dst = np.full((ref_prof['height'], ref_prof['width']), np.nan, dtype=np.float32)
        reproject(rasterio.band(src, 1), dst, src_transform=src.transform, src_crs=src.crs,
                  dst_transform=ref_prof['transform'], dst_crs=ref_prof['crs'],
                  src_nodata=src.nodata, dst_nodata=np.nan, resampling=resampling)
    return dst

@st.cache_resource
def get_soil_index_strict(_ref_prof):
    path = smart_download(FIXED_FILES["sol"], "sols.gpkg")
    gdf = gpd.read_file(path, layer="dcoup")
    def get_val(row):
        t1, p1 = str(row.get("TEXT_1")), float(row.get("TEXT_1_P") or 0)
        t2, p2 = str(row.get("TEXT_2")), float(row.get("TEXT_2_P") or 0)
        s1, s2 = TEXTURE_SCORE.get(t1, 0.5), TEXTURE_SCORE.get(t2, 0.5)
        tex = (s1*p1 + s2*p2)/(p1+p2) if (p1+p2)>0 else 0.5
        def d_cm(s):
            if not s or str(s).lower() == "nan": return 100.0
            if str(s).startswith(">"): return 220.0
            m = re.match(r"(\d+)\s*-\s*(\d+)\s*cm", str(s))
            return (float(m.group(1)) + float(m.group(2))) / 2.0 if m else 100.0
        d1, dp1 = d_cm(row.get("CL_PROF_1")), float(row.get("CL_PRO_1_P") or 0)
        d2, dp2 = d_cm(row.get("CL_PROF_2")), float(row.get("CL_PRO_2_P") or 0)
        depth = (d1*dp1 + d2*dp2)/(dp1+dp2) if (dp1+dp2)>0 else 100.0
        depth_r = 1.0 - np.clip((depth - 30)/170, 0, 1)
        return 0.5 * tex + 0.5 * depth_r
    gdf["val"] = gdf.apply(get_val, axis=1)
    gdf = gdf.to_crs(_ref_prof['crs'])
    shapes = [(g, float(v)) for g, v in zip(gdf.geometry, gdf.val) if g.is_valid]
    return rasterize(shapes, out_shape=(_ref_prof['height'], _ref_prof['width']), transform=_ref_prof['transform'], fill=np.nan)

# --- 4. CHARGEMENT ---
@st.dialog("Guide et définitions", width="large")
def show_help():
    h_style = "color: #3290c7; font-size: 1.0rem; text-transform: uppercase; font-weight: 800; letter-spacing: 0.05em; border-bottom: 2px solid #bfdbfe; padding-bottom: 0.5rem; margin-bottom: 1rem; margin-top: 1rem;"
    st.markdown(f"""
    <div style="{h_style}">1. Indicateurs</div>
    
    - **Précipitations** : Quantité de pluie cumulée sur le mois analysé. Une faible valeur augmente mathématiquement le score de risque .
    - **Evapotranspiration potentielle (ETP)** : Représente la perte d'eau par le sol et la végétation. Plus elle est élevée, plus la sécheresse s'accentue.
    - **Normalized Difference Drought Index (NDDI)** : Un indice NDDI élevé signale une sécheresse, cela se produit typiquement lorsque l’eau interne des plantes (NDWI) diminue plus rapidement que la végétation ne change (NDVI), ce qui indique un stress hydrique.
    - **Texture du sol** : Capacité de rétention. Les sols sableux drainent trop vite, tandis que les argiles retiennent mieux l'eau.
    - **Pente** : Favorise le ruissellement ce qui induit une perte d'eau par écoulement latéral au détriment de l'infiltration.
    - **Occupation du sol (OCS)** : Vulnérabilité selon le couvert (forêt vs cultures).
    - **Distance à l'eau** : Mesure l'éloignement horizontal au réseau hydrographique. Plus un pixel est éloigné, moins il bénéficie de la régulation hydrique des rivières.

    <div style="{h_style}">2. Périodes</div>
    
    - **Mai 2022** : Un des mois de sécheresse les plus important de 2022.
    - **Juin 2022** : Un mois de sécheresse également important mais accompagné de beaucoup de précipitation.
    - **Oct 2023** : Période dite normale car elle ne se situe ni en période humide ni en période sèche.

    <div style="{h_style}">3. Modes d'analyse</div>
    
    - **Standard** : Diagnostic complet d'une date unique (Mai 2022 / Juin 2022 / Oct 2023).
    - **Comparaison** : Outil comparatif interactif pour observer les évolutions entre les périodes.

    <div style="{h_style}">4. Rendu</div>
    
    - **Classes** : Simplification en 5 niveaux (très faible à très fort).
    - **Continu** : Valeur brute de l'indice (0-1) montrant les gradients.
    """, unsafe_allow_html=True)

@st.dialog("Bienvenue sur le diagnostic de la sécheresse pour le bassin de la Sauer", width="large")
def show_welcome():
    h_style = "color: #3290c7; font-size: 1.0rem; text-transform: uppercase; font-weight: 800; letter-spacing: 0.05em; border-bottom: 2px solid #bfdbfe; padding-bottom: 0.5rem; margin-bottom: 1rem; margin-top: 1rem;"
    st.markdown(f"""
    Cet outil interactif permet de diagnostiquer la vulnérabilité à la sécheresse sur le bassin de la Sauer via une analyse multicritère spatiale.
    Vous êtes libre de réaliser vos propres interprétations en sélectionnant les indicateurs pertinents et en définissant leur importance.

    <div style="{h_style}">Méthodologie AHP / Saaty</div>
    
    Le calcul de risque repose sur l'**Analytic Hierarchy Process (AHP)** :
    - **Choix des indicateurs** : Activez ou désactivez les données Climat (précipitations, etp) / Sol (texture, pente, dist. eau) / Végétation (nddi, ocs) selon votre analyse.
    - **Matrice de Saaty** : Pondérez les indicateurs entre eux. **Attention à la cohérence** ! Si vos comparaisons sont contradictoires, le *Ratio de Cohérence (RC)* augmentera, signalant qu'il faut réajuster la matrice.

    <div style="{h_style}">Fonctionnalités et rendu</div>
    
    - **Visualisation** : Choix entre un affichage simplifié en 5 classes (très faible à très fort) ou une valeur continue (0-1).
    - **Statistiques** : Analysez la répartition du risque pour chaque carte générée via les graphiques dédiés.
    - **Comparaison** : Mode "Comparaison temporelle" pour visualiser l'évolution du risque entre deux périodes.
    - **Export** : Téléchargez le résultat final au format PNG avec la mise en page, ou au format GeoTIFF pour vos rapports ou vos traitements SIG.
    
    Retrouvez plus d'infos dans le Guide (bouton en haut à droite).
    """, unsafe_allow_html=True)

if "welcome_shown" not in st.session_state:
    show_welcome()
    st.session_state.welcome_shown = True

c1, c2 = st.columns([14, 1])
with c1:
    st.title("Diagnostic de la sécheresse - Bassin de la Sauer")
with c2:
    st.write("")
    st.write("")
    if st.button("🛈 Guide"):
        show_help()

with st.sidebar:
    st.header("Mode et Période")
    mode_ana = st.radio("Mode d'analyse :", ["Analyse standard", "Comparaison temporelle"])
    
    selected_period = None
    period_1_key = None
    period_2_key = None
    p_slug = None # Default
    
    # Préparation des labels et sous-titres (captions) pour le style
    p_keys = list(PERIODS_CONFIG.keys())
    p_labels = [k.split(" (")[0] for k in p_keys]
    p_captions = [k.split(" (")[1].replace(")", "") for k in p_keys]
    
    if mode_ana == "Analyse standard":
        # Utilisation de captions pour afficher la description en petit
        sel_idx = st.radio(
            "Sélectionner la période :", 
            range(len(p_keys)), 
            format_func=lambda i: p_labels[i], 
            captions=p_captions
        )
        selected_period = p_keys[sel_idx]
        period_cfg = PERIODS_CONFIG[selected_period]
        p_slug = period_cfg["slug"]
    else:
        st.markdown("---")
        # Passage en Radio pour bénéficier du style "captions" (texte petit)
        idx1 = st.radio(
            "Période A (Gauche)", 
            range(len(p_keys)), 
            index=0,
            format_func=lambda i: p_labels[i],
            captions=p_captions,
            key="p1_radio"
        )
        period_1_key = p_keys[idx1]

        st.markdown("---") 
        idx2 = st.radio(
            "Période B (Droite)", 
            range(len(p_keys)), 
            index=1,
            format_func=lambda i: p_labels[i],
            captions=p_captions,
            key="p2_radio"
        )
        period_2_key = p_keys[idx2]
        
        # On charge par defaut la config de la periode 1 pour les initialisations globales
        period_cfg = PERIODS_CONFIG[period_1_key]
        p_slug = period_cfg["slug"]

    st.header("Indicateurs")
    use_precip = st.checkbox("1. Précipitations", value=True)
    use_etp = st.checkbox("2. ETP", value=True)
    use_nddi = st.checkbox("3. NDDI", value=True)
    use_soil = st.checkbox("4. Texture sol", value=True)
    use_slope = st.checkbox("5. Pente", value=True)
    use_ocs = st.checkbox("6. OCS", value=True)
    use_dist = st.checkbox("7. Distance à l'eau", value=True)

    st.header("Visualisation")
    viz_mode = st.radio("Type de rendu :", ["Classes", "Continu (0-1)"])

pente_path = smart_download(FIXED_FILES["pente"], "pente.tif")
with rasterio.open(pente_path) as ref:
    ref_prof = ref.profile.copy()
    ref_prof.update(dtype='float32', nodata=np.nan)
    extent = [ref.bounds.left, ref.bounds.right, ref.bounds.bottom, ref.bounds.top]

@st.cache_data(show_spinner="Chargement des données...", persist=True)
def load_and_process_layers(period_cfg, p_slug, _ref_prof, use_precip, use_etp, use_nddi, use_soil, use_slope, use_ocs, use_dist):
    layers = {}
    labels = []

    if use_precip:
        arr = align_strict(smart_download(period_cfg["precip"], f"{p_slug}_pre.tif"), _ref_prof)
        layers["Précipitations"] = 1.0 - np.clip((arr - 9.451)/(161.829 - 9.451), 0.0, 1.0)
        labels.append("Précipitations")

    if use_etp:
        arr = align_strict(smart_download(period_cfg["etp"], f"{p_slug}_etp.tif"), _ref_prof)
        layers["ETP"] = np.clip((arr - 7.536)/(164.315 - 7.536), 0.0, 1.0)
        labels.append("ETP")

    if use_nddi:
        arr = align_strict(smart_download(period_cfg["nddi"], f"{p_slug}_nddi.tif"), _ref_prof)
        layers["NDDI"] = fuzzy_smf(arr, 0.0, 1.0)
        labels.append("NDDI")

    if use_soil:
        arr = get_soil_index_strict(_ref_prof)
        layers["Texture Sol"] = np.clip((arr - 0.173)/(0.887 - 0.173), 0.0, 1.0)
        labels.append("Texture Sol")

    if use_slope:
        p_path = smart_download(FIXED_FILES["pente"], "pente.tif")
        arr = align_strict(p_path, _ref_prof)
        layers["Pente"] = fuzzy_smf(arr, 0.964, 29.146)
        labels.append("Pente")

    if use_ocs:
        arr = align_strict(smart_download(period_cfg["OCS"], f"{p_slug}_ocs.tif"), _ref_prof, Resampling.nearest)
        ocs_f = np.zeros_like(arr)
        for k, v in {0:0, 1:0.25, 2:0.5, 3:0.75, 4:1}.items(): ocs_f[np.isclose(arr, k, atol=0.1)] = v
        layers["OCS"] = ocs_f
        labels.append("OCS")

    if use_dist:
        arr = align_strict(smart_download(FIXED_FILES["dist_eau"], "dist.tif"), _ref_prof)
        layers["Distance eau"] = np.clip((arr - 0.0)/(800.391 - 0.0), 0.0, 1.0)
        labels.append("Distance eau")
        
    return layers, labels

data_layers, active_labels = load_and_process_layers(period_cfg, p_slug, ref_prof, use_precip, use_etp, use_nddi, use_soil, use_slope, use_ocs, use_dist)

if len(active_labels) < 2:
    st.error("⚠️ Sélectionnez au moins 2 indicateurs.")
    st.stop()

# --- 6. MATRICE AHP ---
if 'master_matrix' not in st.session_state:
    all_sorted_labels = ["Précipitations", "ETP", "NDDI", "Texture Sol", "Pente", "OCS", "Distance eau"]
    m = np.ones((7, 7))
    m[0,1]=2; m[0,2]=3; m[0,3]=3; m[0,4]=6; m[0,5]=6; m[0,6]=7
    m[1,2]=1; m[1,3]=1.5; m[1,4]=5; m[1,5]=5; m[1,6]=6
    m[2,3]=1; m[2,4]=5; m[2,5]=3; m[2,6]=3
    m[3,4]=2; m[3,5]=3; m[3,6]=4
    m[4,5]=2; m[4,6]=3
    m[5,6]=2
    for i in range(7):
        for j in range(i+1, 7): m[j,i] = 1.0 / m[i,j]
    st.session_state.master_matrix = pd.DataFrame(m, index=all_sorted_labels, columns=all_sorted_labels)

current_matrix = st.session_state.master_matrix.loc[active_labels, active_labels]

def update_matrix():
    edits = st.session_state["matrix_editor"]["edited_rows"]
    for row_idx, changes in edits.items():
        row_name = active_labels[row_idx]
        for col_name, val in changes.items():
            st.session_state.master_matrix.at[row_name, col_name] = float(val)
            st.session_state.master_matrix.at[col_name, row_name] = 1.0 / float(val)

def reset_matrix_ones():
    labels = ["Précipitations", "ETP", "NDDI", "Texture Sol", "Pente", "OCS", "Distance eau"]
    st.session_state.master_matrix = pd.DataFrame(np.ones((7, 7)), index=labels, columns=labels)

def reset_matrix_expert():
    labels = ["Précipitations", "ETP", "NDDI", "Texture Sol", "Pente", "OCS", "Distance eau"]
    m = np.ones((7, 7))
    m[0,1]=2; m[0,2]=3; m[0,3]=3; m[0,4]=6; m[0,5]=6; m[0,6]=7
    m[1,2]=1; m[1,3]=1.5; m[1,4]=5; m[1,5]=5; m[1,6]=6
    m[2,3]=1; m[2,4]=5; m[2,5]=3; m[2,6]=3
    m[3,4]=2; m[3,5]=3; m[3,6]=4
    m[4,5]=2; m[4,6]=3
    m[5,6]=2
    for i in range(7):
        for j in range(i+1, 7): m[j,i] = 1.0 / m[i,j]
    st.session_state.master_matrix = pd.DataFrame(m, index=labels, columns=labels)

with st.expander("✏️ ÉDITEUR DE MATRICE SAATY", expanded=False):
    # On utilise des colonnes avec des ratios pour rapprocher les boutons à gauche
    # [1, 1.2, 5] signifie que les boutons prendront une petite partie de la largeur
    c1, c2, _ = st.columns([1, 3, 5]) 
    c1.button("Égaliser (tout à 1)", on_click=reset_matrix_ones, help="Met tous les rapports à 1")
    c2.button("⮌ Rétablir poids experts", on_click=reset_matrix_expert, help="Rétablit la matrice initiale du modèle")
    st.data_editor(current_matrix, use_container_width=True, key="matrix_editor", on_change=update_matrix)

mat_v = current_matrix.values.astype(float)
eigv, eigvec = np.linalg.eig(mat_v)
idx = np.argmax(np.real(eigv))
weights = np.abs(np.real(eigvec[:, idx]))
weights /= weights.sum()
cr = ((np.real(eigv[idx]) - len(active_labels)) / (len(active_labels) - 1)) / {2:0, 3:0.58, 4:0.9, 5:1.12, 6:1.24, 7:1.32}.get(len(active_labels), 1)

# --- HELPER: CALCUL AHP ---
def calculate_ahp_map(p_slug_target, _ref_prof, _weights, _active_labels):
    # Charge les layers pour cette periode specifique
    p_cfg = PERIODS_CONFIG[next(k for k, v in PERIODS_CONFIG.items() if v["slug"] == p_slug_target)]
    # Note: On re-utilise load_and_process_layers qui est cachee (st.cache_data)
    layers, _ = load_and_process_layers(p_cfg, p_slug_target, _ref_prof, use_precip, use_etp, use_nddi, use_soil, use_slope, use_ocs, use_dist)
    
    # Calcul Masque
    c_mask = np.zeros_like(layers[_active_labels[0]], dtype=bool)
    for name in _active_labels: c_mask |= np.isnan(layers[name])
    b_arr = align_strict(smart_download(FIXED_FILES["mask_bati"], "bati.tif"), _ref_prof, Resampling.nearest)
    c_mask |= (b_arr > 0)

    # Somme ponderee
    res = np.zeros_like(c_mask, dtype=np.float32)
    for idx, name in enumerate(_active_labels): res += layers[name] * _weights[idx]
    res[c_mask] = np.nan
    return res

# --- HELPER: CALCUL DES LIMITES DE ZOOM (unifié) ---
def calculate_zoom_bounds(mask, _extent, shape):
    if not np.any(mask): 
        return _extent[:2], _extent[2:]

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Conversion en coordonnées géographiques
    # On suppose que l'extent correspond exactement au shape
    res_x = (_extent[1] - _extent[0]) / shape[1]
    res_y = (_extent[3] - _extent[2]) / shape[0]

    # Coordonnées des pixels extrêmes
    x_min = _extent[0] + cmin * res_x
    x_max = _extent[0] + (cmax + 1) * res_x
    y_min = _extent[3] - (rmax + 1) * res_y # Attention axe Y image vs géo
    y_max = _extent[3] - rmin * res_y

    w = x_max - x_min
    h = y_max - y_min

    # Forçage ratio 1.5 (format rectangulaire paysage)
    target_ratio = 1.5
    current_ratio = w / h
    if current_ratio < target_ratio:
        target_w = h * target_ratio
        delta_w = target_w - w
        x_min -= delta_w / 2
        x_max += delta_w / 2
        w = target_w  # Update width
    
    # Ajout Marge (Zoom Out) - 1%
    buffer_ratio = 0.08
    x_min -= w * buffer_ratio
    x_max += w * buffer_ratio
    y_min -= h * buffer_ratio 
    y_max += h * buffer_ratio

    return (x_min, x_max), (y_min, y_max)

def fig_to_array(fig):
    """Convertit une figure Matplotlib en tableau RGB pour image_comparison"""
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    try:
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    except:
        # Fallback pour versions récentes de Matplotlib
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        # Convert RGBA to RGB if needed, but tostring_rgb usually returns 3 channels
        # If buffer_rgba, it is 4 channels.
        if len(buf) == w * h * 4:
           return buf.reshape(h, w, 4)[:,:,:3]
           
    return buf.reshape(h, w, 3)

# --- HELPER: GENERATION FIGURE (pour affichage classique/export) ---
def generate_map_figure(ahp_data, title, viz_type, _extent, xlim=None, ylim=None):
    # Optimisation pour le Slider : Ratio 1.5 et suppression des marges
    # Cela évite les bandes blanches autour de la carte
    fig, ax = plt.subplots(figsize=(12, 8), dpi=100) 
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    fig.patch.set_alpha(0) 
    ax.axis('off') 
    if title: ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    # Gestion du Zoom
    if xlim is not None and ylim is not None:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    else:
        # Mode auto (standard)
        mask_data = ~np.isnan(ahp_data)
        if np.any(mask_data):
            xl, yl = calculate_zoom_bounds(mask_data, _extent, ahp_data.shape)
            ax.set_xlim(xl)
            ax.set_ylim(yl)

    if viz_type == "Classes":
        cmap = ListedColormap(["#1a9850", "#91cf60", "#fee08b", "#fc8d59", "#d73027"])
        norm = BoundaryNorm([0, 0.2, 0.4, 0.6, 0.8, 1.0], 5)
    else:
        cmap = "RdYlGn_r"
        norm = plt.Normalize(vmin=0, vmax=1)

    im = ax.imshow(ahp_data, cmap=cmap, norm=norm, extent=_extent, alpha=1, zorder=2)
    try: ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, crs=ref_prof['crs'].to_string(), zorder=1, alpha=1.0)
    except: pass
    
    # Echelle (simplifiee)
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    scale_len = 5000; sx = x1 - (x1-x0)*0.06 - scale_len; sy = y0 + (y1-y0)*0.08
    ax.add_patch(plt.Rectangle((sx-300, sy-350), scale_len+600, 900, facecolor="white", edgecolor="none", zorder=2))
    ax.plot([sx, sx+scale_len], [sy, sy], color="black", lw=3, zorder=3)
    ax.text(sx + scale_len/2, sy+400, "5 km", ha="center", va="bottom", fontsize=10, zorder=3)
    
    # Nord
    xn_arrow = x0 + (x1-x0)*0.9; yn_arrow = y1 - (y1-y0)*0.1
    ax.text(xn_arrow, yn_arrow, 'N', ha='center', va='bottom', fontsize=16, fontweight='bold', zorder=4)
    ax.arrow(xn_arrow, yn_arrow - 2000, 0, 1000, head_width=500, fc='black', lw=1, zorder=4)

    # Legende
    ax_ins = inset_axes(ax, width="4%", height="40%", loc='lower left', borderpad=4)
    if viz_type == "Classes":
        cb = fig.colorbar(im, cax=ax_ins, ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
        cb.ax.set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1'], fontsize=9)
    else:
        cb = fig.colorbar(im, cax=ax_ins)
    cb.outline.set_visible(False)

    # Titre de la légende à gauche et vertical
    cb.ax.yaxis.set_label_position('left')
    cb.set_label("Indice de vulnérabilité à la sécheresse", rotation=90, labelpad=15, fontsize=10, fontweight='bold')

    return fig

# --- 7. LOGIQUE D'ANALYSE ---
if mode_ana == "Comparaison temporelle":
    
    # Calculs
    ahp_full_1 = calculate_ahp_map(PERIODS_CONFIG[period_1_key]["slug"], ref_prof, weights, active_labels)
    ahp_full_2 = calculate_ahp_map(PERIODS_CONFIG[period_2_key]["slug"], ref_prof, weights, active_labels)
    
    with st.spinner("Génération de la vue comparative..."):
        # Calcul limite commune (union des données valides)
        mask_union = (~np.isnan(ahp_full_1)) | (~np.isnan(ahp_full_2))
        xlim_common, ylim_common = calculate_zoom_bounds(mask_union, extent, ahp_full_1.shape)
        
        # Génération des 2 figures via Matplotlib pour avoir tous les habillages (Légende, Fond, etc)
        f1 = generate_map_figure(ahp_full_1, "", viz_mode, extent, xlim=xlim_common, ylim=ylim_common)
        img1 = fig_to_array(f1)
        plt.close(f1)

        f2 = generate_map_figure(ahp_full_2, "", viz_mode, extent, xlim=xlim_common, ylim=ylim_common)
        img2 = fig_to_array(f2)
        plt.close(f2)
    
    # Layout : Carte + Stats (Camemberts)
    col_viz, col_stats = st.columns([2, 1])
    
    with col_viz:
        st.markdown("#### Comparaison temporelle")
        image_comparison(
            img1=img1,
            img2=img2,
            label1=period_1_key,
            label2=period_2_key,
            width=1000, 
            starting_position=50,
            show_labels=True,
            make_responsive=True,
            in_memory=True
        )

    with col_stats:
        st.markdown("#### Répartition")
        
        bins = [0, 0.2, 0.4, 0.6, 0.8, 1.01]
        labels_txt = ["Très faible", "Faible", "Modéré", "Fort", "Très fort"]
        colors_stat = ["#1a9850", "#91cf60", "#fee08b", "#fc8d59", "#d73027"]
        
        def display_pie(arr, title):
            valid = arr[~np.isnan(arr)]
            if len(valid) == 0: return
            counts, _ = np.histogram(valid, bins=bins)
            percents = (counts / counts.sum()) * 100
            df = pd.DataFrame({"Niveau": labels_txt, "Pourcentage": percents, "Couleur": colors_stat})
            
            c = alt.Chart(df).mark_arc(innerRadius=40).encode(
                theta=alt.Theta("Pourcentage", stack=True),
                color=alt.Color("Niveau", scale=alt.Scale(domain=labels_txt, range=colors_stat), legend=None),
                order=alt.Order("Niveau", sort="ascending"),
                tooltip=["Niveau", alt.Tooltip("Pourcentage", format=".1f")]
            ).properties(title=title, height=290)
            
            st.altair_chart(c, use_container_width=True)

        display_pie(ahp_full_1, period_1_key)
        st.markdown("---")
        display_pie(ahp_full_2, period_2_key)

else:
    # --- MODE STANDARD ---
    ahp_final = calculate_ahp_map(p_slug, ref_prof, weights, active_labels)

    # Affichage
    col_map, col_info = st.columns([2, 1])

    with col_map:
        st.subheader(f"Carte : {selected_period}")
        fig = generate_map_figure(ahp_final, "", viz_mode, extent)
        
        # Capture PNG pour export
        buf_png = io.BytesIO()
        fig.savefig(buf_png, format="png", dpi=150, bbox_inches='tight', transparent=True)
        
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        st.pyplot(fig, clear_figure=True)

        # Stats Standard
        valid_pixels = ahp_final[~np.isnan(ahp_final)]
        if len(valid_pixels) > 0:
            with st.expander("Répartition statistique", expanded=False):
                bins = [0, 0.2, 0.4, 0.6, 0.8, 1.01]
                labels_txt = ["Très faible", "Faible", "Modéré", "Fort", "Très fort"]
                colors_stat = ["#1a9850", "#91cf60", "#fee08b", "#fc8d59", "#d73027"]
                counts, _ = np.histogram(valid_pixels, bins=bins)
                percents = (counts / counts.sum()) * 100
                
                df_stats = pd.DataFrame({"Niveau": labels_txt, "Pourcentage": percents, "Couleur": colors_stat})
                chart = alt.Chart(df_stats).mark_bar().encode(
                    x=alt.X('Niveau', sort=None, axis=alt.Axis(labelAngle=0, titleFontWeight="bold")),
                    y=alt.Y('Pourcentage', scale=alt.Scale(domain=[0, 100]), axis=alt.Axis(titleFontWeight="bold")),
                    color=alt.Color('Niveau', scale=alt.Scale(domain=labels_txt, range=colors_stat), legend=None),
                    tooltip=['Niveau', alt.Tooltip('Pourcentage', format='.1f')]
                ).properties(height=300)
                st.altair_chart(chart, use_container_width=True)

    with col_info:
        st.subheader("Diagnostics AHP")
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Ratio de cohérence (RC)</div>
            <div class="metric-value">{cr:.3f}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("Poids des indicateurs")
        for i, name in enumerate(active_labels):
            st.write(f"**{name}** : `{weights[i]*100:.1f}%`")
            st.progress(float(weights[i]))
            
    # Export standard
    st.write("---")
    st.subheader("💾 Export")
    st.write("")

    c_dl1, c_dl2, _ = st.columns([1, 1, 5])
    
    # Export Tiff
    buf = io.BytesIO()
    with rasterio.open(buf, 'w', driver='GTiff', height=ref_prof['height'], width=ref_prof['width'], count=1, dtype='float32', crs=ref_prof['crs'], transform=ref_prof['transform'], nodata=-9999, compress='lzw') as dst:
        dst.write(np.where(np.isnan(ahp_final), -9999, ahp_final).astype('float32'), 1)
    
    with c_dl1:
        st.download_button(label="Télécharger GeoTIFF", data=buf.getvalue(), file_name=f"sauer_{p_slug}.tif", mime="image/tiff")
        
    # Export PNG
    if 'buf_png' in locals() and buf_png:
        with c_dl2:
            st.download_button(label="Télécharger Image (PNG)", data=buf_png.getvalue(), file_name=f"sauer_{p_slug}.png", mime="image/png")