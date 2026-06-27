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
from functools import lru_cache
from typing import Optional

import joblib
import numpy as np
import pandas as pd
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

st.set_page_config(
    page_title="GateOne.immo — Estimateur de prix",
    page_icon="🏠",
    layout="centered",
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


# ── Interface ──────────────────────────────────────────────────
st.title("🏠 GateOne.immo — Estimateur de prix")
st.caption("Estimation de prix immobiliers à Marrakech, propulsée par des modèles XGBoost / LightGBM.")

try:
    categories_info = list_categories()
except FileNotFoundError:
    st.error(
        "Dossier `models/` introuvable. Assurez-vous que le dossier `models/` "
        "(contenant les fichiers `.pkl` et `registry.json`) est bien présent à "
        "côté de `app.py` dans le repo déployé."
    )
    st.stop()

categorie_labels = {c["categorie"]: c["categorie"].replace("_", " ").capitalize() for c in categories_info}
categorie_choice = st.selectbox(
    "Type de bien",
    options=list(categorie_labels.keys()),
    format_func=lambda c: categorie_labels[c],
)

cat_meta = next(c for c in categories_info if c["categorie"] == categorie_choice)
st.caption(
    f"📊 Modèle entraîné sur **{cat_meta['n_rows_trained']}** annonces · "
    f"R² = **{cat_meta['r2_final']:.2f}** · "
    f"erreur médiane ≈ **{cat_meta['mdape_final']:.1f}%**"
)

fields = get_required_fields(categorie_choice)

st.divider()
st.subheader("Caractéristiques du bien")

payload = {}

num_features = fields.get("num_features", [])
num_ranges = fields.get("num_ranges", {})
if num_features:
    cols = st.columns(2)
    for i, field in enumerate(num_features):
        rng = num_ranges.get(field, {})
        default_val = rng.get("median", 0.0)
        min_val = rng.get("min", 0.0)
        max_val = rng.get("max", default_val * 3 if default_val else 1000.0)
        with cols[i % 2]:
            payload[field] = st.number_input(
                humanize(field),
                min_value=0.0,
                value=float(default_val),
                step=1.0,
                help=f"Plage observée dans les données : {min_val:g} – {max_val:g}",
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
    st.subheader("Équipements")
    equipements = {}
    n_cols = 3
    cols = st.columns(n_cols)
    for i, eq in enumerate(equip_features):
        with cols[i % n_cols]:
            equipements[eq] = 1 if st.checkbox(humanize(eq), key=f"eq_{eq}") else 0
    payload["equipements"] = equipements

st.divider()
use_model = st.radio(
    "Modèle à utiliser",
    options=["Automatique (recommandé)", "xgb", "lgb"],
    horizontal=True,
    help="XGBoost (xgb) et LightGBM (lgb) sont les deux algorithmes entraînés pour cette catégorie.",
)

if st.button("💰 Estimer le prix", type="primary", use_container_width=True):
    with st.spinner("Calcul de l'estimation..."):
        try:
            model_arg = use_model if use_model != "Automatique (recommandé)" else None
            result = predict_price(categorie_choice, payload, use_model=model_arg)
        except Exception as e:
            st.error(f"Erreur lors du calcul : {e}")
        else:
            st.success("Estimation calculée avec succès")

            col1, col2, col3 = st.columns(3)
            col1.metric("Prix estimé", f"{result['prix_estime_dh']:,} DH".replace(",", " "))
            col2.metric("Fourchette basse", f"{result['prix_estime_min_dh']:,} DH".replace(",", " "))
            col3.metric("Fourchette haute", f"{result['prix_estime_max_dh']:,} DH".replace(",", " "))

            st.caption(
                f"Modèle utilisé : **{result['modele_utilise']}** · "
                f"R² du modèle : **{result['r2_modele']}** · "
                f"marge d'erreur médiane : **{result['marge_erreur_pct']}%** · "
                f"prix médian observé pour cette catégorie : "
                f"**{result['prix_median_dataset_dh']:,.0f} DH**".replace(",", " ")
            )

            with st.expander("Voir les données utilisées pour le calcul"):
                st.json(payload)

st.divider()
st.caption(
    "⚠️ Ces estimations sont fournies à titre indicatif et ne remplacent pas "
    "une évaluation professionnelle ou une expertise immobilière."
)
