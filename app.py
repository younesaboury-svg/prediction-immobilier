"""
============================================================
GateOne.immo — Estimateur de prix (app Streamlit autonome)
============================================================
Version FUSIONNÉE : les modèles ML (XGBoost / LightGBM) sont
chargés et exécutés DIRECTEMENT dans cette app Streamlit.
Aucune API séparée à héberger — un seul déploiement suffit.

Lancement local :
    pip install -r requirements.txt
    streamlit run app.py

Déploiement : Streamlit Community Cloud, en pointant sur ce
fichier comme point d'entrée, avec models/ commité dans le repo.
============================================================
"""
import json
import os
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Configuration ────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

VALID_CATEGORIES = [
    "appartements",
    "bureaux",
    "magasins",
    "maisons",
    "riads",
    "terrains",
    "villas",
    "maisons_dhotes",
]

MIN_ANNONCES_QUARTIER_DISPLAY = 5  # doit correspondre au seuil utilisé dans compute_dashboard_stats.py

st.set_page_config(
    page_title="GateOne.immo — Estimateur de prix",
    page_icon="🏠",
    layout="wide",
)

# ── Thème visuel ────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, sans-serif;
    }

    /* Cache le header gris par défaut de Streamlit pour un look plus net */
    header[data-testid="stHeader"] {
        background: transparent;
    }

    /* Bandeau de marque */
    .gateone-hero {
        background: #1C1815;
        border-radius: 12px;
        padding: 1.8rem 2.2rem;
        margin-bottom: 1.6rem;
    }
    .gateone-hero .eyebrow {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #C1602E;
        margin-bottom: 0.4rem;
    }
    .gateone-hero h1 {
        font-family: 'Fraunces', serif;
        font-weight: 600;
        font-size: 1.85rem;
        color: #FFFFFF;
        margin: 0 0 0.35rem 0;
        line-height: 1.2;
    }
    .gateone-hero p {
        font-family: 'Inter', sans-serif;
        font-size: 0.92rem;
        color: rgba(255, 255, 255, 0.68);
        margin: 0;
        max-width: 620px;
    }

    /* Sous-titres de section, façon "label" éditorial */
    .section-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #C1602E;
        border-bottom: 2px solid #EADFCB;
        padding-bottom: 0.5rem;
        margin: 1.6rem 0 1.1rem 0;
    }

    /* Carte méta (n annonces, R², erreur) */
    .meta-strip {
        display: flex;
        gap: 1.6rem;
        flex-wrap: wrap;
        background: #F0E9DD;
        border: 1px solid #E4D7BE;
        border-radius: 10px;
        padding: 0.85rem 1.2rem;
        margin-bottom: 0.4rem;
    }
    .meta-strip .meta-item {
        font-family: 'Inter', sans-serif;
        font-size: 0.88rem;
        color: #5C5248;
    }
    .meta-strip .meta-item b {
        font-family: 'JetBrains Mono', monospace;
        color: #1C1815;
        font-weight: 700;
    }

    /* Carte résultat — mise en page en grille, façon fiche d'évaluation */
    .result-card {
        background: #FFFFFF;
        border: 1px solid #E4D7BE;
        border-radius: 12px;
        padding: 0;
        margin: 1rem 0 1.4rem 0;
        overflow: hidden;
    }
    .result-card .result-header {
        background: #1C1815;
        padding: 1.1rem 1.6rem;
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 0.5rem;
    }
    .result-card .result-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: rgba(255, 255, 255, 0.55);
    }
    .result-card .result-price {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        font-size: 1.9rem;
        color: #FFFFFF;
        line-height: 1;
    }
    .result-card .result-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
    }
    .result-card .result-cell {
        padding: 0.95rem 1.2rem;
        border-right: 1px solid #EFE8DA;
    }
    .result-card .result-cell:last-child { border-right: none; }
    .result-card .result-cell .cell-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #8A7F6E;
        margin-bottom: 0.25rem;
    }
    .result-card .result-cell .cell-value {
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
        font-weight: 600;
        color: #1C1815;
    }
    .result-card .result-footnote {
        font-family: 'Inter', sans-serif;
        font-size: 0.8rem;
        color: #8A7F6E;
        padding: 0.7rem 1.6rem;
        border-top: 1px solid #EFE8DA;
        background: #FAF7F2;
    }

    /* Bouton principal */
    .stButton > button[kind="primary"] {
        background: #1C1815;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.65rem 1.2rem;
        transition: background 0.15s ease;
    }
    .stButton > button[kind="primary"]:hover {
        background: #C1602E;
    }

    .disclaimer-strip {
        font-family: 'Inter', sans-serif;
        font-size: 0.82rem;
        color: #8A7F6E;
        border-top: 1px solid #EFE8DA;
        padding-top: 1rem;
        margin-top: 1.4rem;
    }

    /* Cartes KPI du dashboard */
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E4D7BE;
        border-left: 3px solid #C1602E;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        height: 100%;
    }
    .kpi-card .kpi-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #8A7F6E;
        margin-bottom: 0.35rem;
    }
    .kpi-card .kpi-value {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        font-size: 1.5rem;
        color: #1C1815;
        line-height: 1.15;
    }
    .kpi-card .kpi-sub {
        font-family: 'Inter', sans-serif;
        font-size: 0.78rem;
        color: #8A7F6E;
        margin-top: 0.3rem;
    }

    .stTabs [data-baseweb="tab"] {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
    }

    /* Limite la largeur du contenu en mode wide, pour garder une lecture confortable */
    .block-container {
        max-width: 1080px;
        padding-top: 2.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


class CategoryNotFoundError(Exception):
    pass


# ── Chargement des modèles (mis en cache par Streamlit) ──────
@st.cache_resource(show_spinner=False)
def load_registry() -> dict:
    path = os.path.join(MODEL_DIR, "registry.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource(show_spinner=False)
def load_dashboard_stats() -> Optional[dict]:
    path = os.path.join(MODEL_DIR, "dashboard_stats.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource(show_spinner=False)
def load_artefacts(categorie: str) -> dict:
    if categorie not in VALID_CATEGORIES:
        raise CategoryNotFoundError(f"Catégorie inconnue: {categorie}")
    path = os.path.join(MODEL_DIR, f"model_{categorie}.pkl")
    if not os.path.exists(path):
        raise CategoryNotFoundError(f"Modèle introuvable pour la catégorie: {categorie}")
    return joblib.load(path)


def list_categories() -> list:
    registry = load_registry()
    return [
        {
            "categorie": k,
            "n_rows_trained": v["stats"]["n_rows_trained"],
            "best_model": v["best_model"],
            "r2_final": v["metrics"]["final"][v["best_model"]]["r2"],
            "mdape_final": v["metrics"]["final"][v["best_model"]]["mdape"],
        }
        for k, v in registry.items()
    ]


def get_required_fields(categorie: str) -> dict:
    art = load_artefacts(categorie)
    return {
        "num_features": art["num_features"],
        "cat_features": art["cat_features"],
        "cat_values": art["stats"]["cat_values"],
        "equip_features": art["equip_features"],
        "has_localisation": art["has_localisation"],
        "localisations": art["stats"]["localisations"] if art["has_localisation"] else [],
        "num_ranges": art["stats"]["num_ranges"],
    }


def _build_feature_row(art: dict, payload: dict) -> pd.DataFrame:
    """Construit une ligne de features prête à être encodée, à partir d'un payload
    (dict) saisi via le formulaire. Applique des valeurs par défaut neutres pour
    les champs manquants."""
    row = {}

    for c in art["num_features"]:
        val = payload.get(c, 0)
        row[c] = float(val) if val is not None else 0.0

    for c in art["cat_features"]:
        val = payload.get(c)
        row[c] = val if val not in (None, "") else "Inconnu"

    equipements = payload.get("equipements") or {}
    for ef in art["equip_features"]:
        v = equipements.get(ef, payload.get(ef, 0))
        row[ef] = int(bool(v))

    if art["has_localisation"]:
        loc = payload.get("localisation")
        row["localisation"] = loc if loc not in (None, "") else "Inconnue"

    return pd.DataFrame([row])[art["feature_cols"]]


def predict_price(categorie: str, payload: dict, use_model: Optional[str] = None) -> dict:
    art = load_artefacts(categorie)

    model_choice = use_model or art["best_model"]
    if model_choice not in ("xgb", "lgb"):
        raise ValueError("use_model doit être 'xgb' ou 'lgb'")

    X_pred = _build_feature_row(art, payload)

    if art["cat_features"]:
        X_pred[art["cat_features"]] = art["ord_enc"].transform(X_pred[art["cat_features"]])

    if art["has_localisation"] and art["target_enc"] is not None:
        X_pred = art["target_enc"].transform(X_pred)

    model = art["xgb_model"] if model_choice == "xgb" else art["lgb_model"]
    pred_log = model.predict(X_pred)[0]
    price = max(float(np.expm1(pred_log)), 0.0)

    mdape = art["metrics"]["final"][model_choice]["mdape"] / 100.0
    low = price * (1 - mdape)
    high = price * (1 + mdape)

    return {
        "categorie": categorie,
        "prix_estime_dh": round(price),
        "prix_estime_min_dh": round(low),
        "prix_estime_max_dh": round(high),
        "modele_utilise": model_choice,
        "marge_erreur_pct": round(art["metrics"]["final"][model_choice]["mdape"], 2),
        "r2_modele": round(art["metrics"]["final"][model_choice]["r2"], 4),
        "prix_median_dataset_dh": art["stats"]["price_median"],
    }


def humanize(label: str) -> str:
    return label.replace("equip_", "").replace("_", " ").strip().capitalize()


def fmt_dh(value: float) -> str:
    return f"{value:,.0f}".replace(",", " ")


def render_dashboard(categories_info: list) -> None:
    """Affiche le tableau de bord du marché : volumes, prix au m² par
    catégorie et par quartier, à partir des statistiques précalculées
    (models/dashboard_stats.json), sans recharger les fichiers Excel."""
    stats = load_dashboard_stats()

    if stats is None:
        st.warning(
            "Les statistiques de marché (`models/dashboard_stats.json`) sont "
            "introuvables. Générez-les avec `compute_dashboard_stats.py` puis "
            "ajoutez le fichier au dossier `models/` du repo déployé."
        )
        return

    par_categorie = stats["par_categorie"]
    categorie_labels = {c["categorie"]: c["categorie"].replace("_", " ").capitalize() for c in categories_info}

    # ── KPI globaux ──────────────────────────────────────────────
    n_categories = len(par_categorie)
    prix_m2_medians = [v["prix_m2"]["median"] for v in par_categorie.values()]
    prix_m2_global_median = sorted(prix_m2_medians)[len(prix_m2_medians) // 2]

    kpi_cols = st.columns(3)
    with kpi_cols[0]:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Total annonces</div>
                <div class="kpi-value">{stats['total_annonces']:,}</div>
                <div class="kpi-sub">toutes catégories confondues</div>
            </div>
            """.replace(",", " "),
            unsafe_allow_html=True,
        )
    with kpi_cols[1]:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Catégories suivies</div>
                <div class="kpi-value">{n_categories}</div>
                <div class="kpi-sub">types de biens à Marrakech</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi_cols[2]:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Prix/m² médian (toutes cat.)</div>
                <div class="kpi-value">{fmt_dh(prix_m2_global_median)} DH</div>
                <div class="kpi-sub">médiane des médianes par catégorie</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 0.6rem'></div>", unsafe_allow_html=True)

    # ── Répartition du marché par catégorie ──────────────────────
    st.markdown('<div class="section-label">Répartition des annonces par catégorie</div>', unsafe_allow_html=True)

    repartition = sorted(stats["repartition_marche"], key=lambda r: r["n_annonces"], reverse=True)
    labels = [categorie_labels.get(r["categorie"], r["categorie"]) for r in repartition]
    values = [r["n_annonces"] for r in repartition]

    fig_repartition = go.Figure(
        data=[
            go.Bar(
                x=values,
                y=labels,
                orientation="h",
                marker_color="#C1602E",
                text=[f"{v:,}".replace(",", " ") for v in values],
                textposition="outside",
            )
        ]
    )
    fig_repartition.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#1C1815", size=13),
        xaxis=dict(title="Nombre d'annonces", gridcolor="#EFE8DA"),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig_repartition, width="stretch", config={"displayModeBar": False})

    # ── Sélecteur de catégorie pour le détail ────────────────────
    st.markdown('<div class="section-label">Détail par catégorie</div>', unsafe_allow_html=True)

    dash_categorie = st.selectbox(
        "Catégorie à explorer",
        options=list(categorie_labels.keys()),
        format_func=lambda c: categorie_labels[c],
        key="dash_categorie",
    )

    cat_stats = par_categorie.get(dash_categorie)
    if not cat_stats:
        st.info("Pas de statistiques disponibles pour cette catégorie.")
        return

    kpi_cols2 = st.columns(3)
    with kpi_cols2[0]:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Annonces</div>
                <div class="kpi-value">{cat_stats['n_total_annonces']:,}</div>
            </div>
            """.replace(",", " "),
            unsafe_allow_html=True,
        )
    with kpi_cols2[1]:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Prix/m² médian</div>
                <div class="kpi-value">{fmt_dh(cat_stats['prix_m2']['median'])} DH</div>
                <div class="kpi-sub">moyenne : {fmt_dh(cat_stats['prix_m2']['moyen'])} DH</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi_cols2[2]:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Prix médian</div>
                <div class="kpi-value">{fmt_dh(cat_stats['prix']['median'])} DH</div>
                <div class="kpi-sub">moyenne : {fmt_dh(cat_stats['prix']['moyen'])} DH</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    quartiers = cat_stats.get("quartiers", [])
    if not quartiers:
        st.info("Pas assez de données par quartier pour cette catégorie.")
        return

    st.markdown("<div style='height: 0.6rem'></div>", unsafe_allow_html=True)

    # ── Tableau des quartiers (triable, format outil pro) ────────
    st.markdown('<div class="section-label">Quartiers — détail par zone</div>', unsafe_allow_html=True)

    quartiers_sorted = sorted(quartiers, key=lambda q: q["n_annonces"], reverse=True)
    df_quartiers = pd.DataFrame([
        {
            "Quartier": q["quartier"],
            "Annonces": q["n_annonces"],
            "Prix/m² médian (DH)": q["prix_m2_median"],
            "Prix médian (DH)": q["prix_median"],
        }
        for q in quartiers_sorted
    ])
    st.dataframe(
        df_quartiers,
        width="stretch",
        hide_index=True,
        column_config={
            "Annonces": st.column_config.NumberColumn(format="%d"),
            "Prix/m² médian (DH)": st.column_config.NumberColumn(format="%d DH"),
            "Prix médian (DH)": st.column_config.NumberColumn(format="%d DH"),
        },
    )

    st.markdown("<div style='height: 0.6rem'></div>", unsafe_allow_html=True)

    # ── Graphique comparatif prix/m² par quartier (top 15 par volume) ──
    st.markdown('<div class="section-label">Prix au m² par quartier (top 15 par volume)</div>', unsafe_allow_html=True)
    top15 = sorted(quartiers, key=lambda q: q["n_annonces"], reverse=True)[:15]
    top15_sorted = sorted(top15, key=lambda q: q["prix_m2_median"])

    fig_quartiers = go.Figure(
        data=[
            go.Bar(
                x=[q["prix_m2_median"] for q in top15_sorted],
                y=[q["quartier"] for q in top15_sorted],
                orientation="h",
                marker_color="#3B4FA0",
                text=[f"{fmt_dh(q['prix_m2_median'])} DH" for q in top15_sorted],
                textposition="outside",
            )
        ]
    )
    fig_quartiers.update_layout(
        height=460,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#1C1815", size=13),
        xaxis=dict(title="Prix médian au m² (DH)", gridcolor="#EFE8DA"),
    )
    st.plotly_chart(fig_quartiers, width="stretch", config={"displayModeBar": False})

    st.caption(
        f"Quartiers retenus uniquement s'ils comptent au moins {MIN_ANNONCES_QUARTIER_DISPLAY} "
        "annonces, pour la fiabilité statistique des moyennes affichées."
    )


# ── Interface ──────────────────────────────────────────────────
st.markdown(
    """
    <div class="gateone-hero">
        <div class="eyebrow">GateOne.immo · Marrakech</div>
        <h1>Estimateur de prix immobilier</h1>
        <p>Estimez le prix d'un bien et explorez les tendances du marché
        immobilier à Marrakech, à partir de modèles XGBoost / LightGBM
        entraînés sur des milliers d'annonces réelles.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    categories_info = list_categories()
except FileNotFoundError:
    st.error(
        "Dossier `models/` introuvable. Assurez-vous que le dossier `models/` "
        "(contenant les fichiers `.pkl` et `registry.json`) est bien présent à "
        "côté de `app.py` dans le repo déployé."
    )
    st.stop()

tab_estimate, tab_dashboard = st.tabs(["🏷️ Estimer un prix", "📊 Dashboard du marché"])

# ════════════════════════════════════════════════════════════════
# ONGLET 1 — ESTIMATEUR
# ════════════════════════════════════════════════════════════════
with tab_estimate:
    categorie_labels = {c["categorie"]: c["categorie"].replace("_", " ").capitalize() for c in categories_info}

    categorie_choice = st.selectbox(
        "Type de bien",
        options=list(categorie_labels.keys()),
        format_func=lambda c: categorie_labels[c],
        key="estim_categorie",
    )

    cat_meta = next(c for c in categories_info if c["categorie"] == categorie_choice)
    st.markdown(
        f"""
        <div class="meta-strip">
            <div class="meta-item">Annonces analysées <b>{cat_meta['n_rows_trained']}</b></div>
            <div class="meta-item">Précision (R²) <b>{cat_meta['r2_final']:.2f}</b></div>
            <div class="meta-item">Erreur médiane <b>≈ {cat_meta['mdape_final']:.1f}%</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    fields = get_required_fields(categorie_choice)

    st.markdown('<div class="section-label">Caractéristiques du bien</div>', unsafe_allow_html=True)

    payload = {}

    num_features = fields.get("num_features", [])
    num_ranges = fields.get("num_ranges", {})
    if num_features:
        cols = st.columns(2)
        for i, field in enumerate(num_features):
            rng = num_ranges.get(field, {})
            default_val = rng.get("median", 0.0)
            with cols[i % 2]:
                payload[field] = st.number_input(
                    humanize(field),
                    min_value=0.0,
                    value=float(default_val),
                    step=1.0,
                    key=f"num_{field}",
                )

    cat_features = fields.get("cat_features", [])
    cat_values = fields.get("cat_values", {})
    if cat_features:
        cols = st.columns(2)
        for i, field in enumerate(cat_features):
            options = cat_values.get(field, [])
            with cols[i % 2]:
                if options:
                    payload[field] = st.selectbox(humanize(field), options=options, key=f"cat_{field}")
                else:
                    payload[field] = st.text_input(humanize(field), key=f"cat_{field}")

    if fields.get("has_localisation"):
        localisations = fields.get("localisations", [])
        payload["localisation"] = st.selectbox(
            "Localisation / quartier",
            options=localisations,
            key="localisation",
        )

    equip_features = fields.get("equip_features", [])
    if equip_features:
        st.markdown('<div class="section-label">Équipements</div>', unsafe_allow_html=True)
        equipements = {}
        n_cols = 3
        cols = st.columns(n_cols)
        for i, eq in enumerate(equip_features):
            with cols[i % n_cols]:
                equipements[eq] = 1 if st.checkbox(humanize(eq), key=f"eq_{eq}") else 0
        payload["equipements"] = equipements

    st.markdown('<div class="section-label">Modèle de prédiction</div>', unsafe_allow_html=True)
    use_model = st.radio(
        "Modèle à utiliser",
        options=["Automatique (recommandé)", "xgb", "lgb"],
        horizontal=True,
        label_visibility="collapsed",
        key="estim_use_model",
    )

    if st.button("Estimer le prix", type="primary", width="stretch"):
        with st.spinner("Calcul de l'estimation..."):
            try:
                model_arg = use_model if use_model != "Automatique (recommandé)" else None
                result = predict_price(categorie_choice, payload, use_model=model_arg)
            except Exception as e:
                st.error(f"Erreur lors du calcul : {e}")
            else:
                prix_fmt = f"{result['prix_estime_dh']:,}".replace(",", " ")
                low_fmt = f"{result['prix_estime_min_dh']:,}".replace(",", " ")
                high_fmt = f"{result['prix_estime_max_dh']:,}".replace(",", " ")
                median_fmt = f"{result['prix_median_dataset_dh']:,.0f}".replace(",", " ")

                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-header">
                            <span class="result-label">Prix estimé</span>
                            <span class="result-price">{prix_fmt} DH</span>
                        </div>
                        <div class="result-grid">
                            <div class="result-cell">
                                <div class="cell-label">Fourchette basse</div>
                                <div class="cell-value">{low_fmt} DH</div>
                            </div>
                            <div class="result-cell">
                                <div class="cell-label">Fourchette haute</div>
                                <div class="cell-value">{high_fmt} DH</div>
                            </div>
                            <div class="result-cell">
                                <div class="cell-label">Modèle</div>
                                <div class="cell-value">{result['modele_utilise'].upper()} · R² {result['r2_modele']}</div>
                            </div>
                            <div class="result-cell">
                                <div class="cell-label">Marge d'erreur</div>
                                <div class="cell-value">≈ {result['marge_erreur_pct']}%</div>
                            </div>
                        </div>
                        <div class="result-footnote">
                            Prix médian observé pour cette catégorie sur l'ensemble du marché analysé : {median_fmt} DH
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                with st.expander("Voir les données utilisées pour le calcul"):
                    st.json(payload)


# ════════════════════════════════════════════════════════════════
# ONGLET 2 — DASHBOARD DU MARCHÉ
# ════════════════════════════════════════════════════════════════
with tab_dashboard:
    render_dashboard(categories_info)
