# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 📏 Analyse d'exactitude et d'incertitude
# Streamlit App pour le traitement de données métrologiques avec visualisation

# %%
import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# %% [markdown]
# ## 🧮 Fonctions d'analyse

# %%
def analyse_par_niveau(df, gamma=0.95, lambd=0.05):
    grouped = df.groupby("Valeur certifiée Vc")
    results = {}

    for niveau, group in grouped:
        repetitions = group[["répétition 1", "répétition 2", "répétition 3"]].values.flatten()
        valeurs = repetitions[~np.isnan(repetitions)]

        n = len(valeurs)
        if n < 2:
            continue

        m = np.mean(valeurs)
        s = np.std(valeurs, ddof=1)
        se = s / np.sqrt(n)

        t_critique = stats.t.ppf(1 - (1 - gamma) / 2, df=n - 1)
        k = stats.t.ppf(1 - lambd, df=n - 1)

        l100 = 100 * ((m - t_critique * se) - m) / m
        u100 = 100 * ((m + t_critique * se) - m) / m
        L100 = 100 * ((m - k * s) - m) / m
        U100 = 100 * ((m + k * s) - m) / m

        erreur_relative = 100 * (m - niveau) / niveau

        results[niveau] = {
            "l100": l100,
            "u100": u100,
            "L100": L100,
            "U100": U100,
            "Erreur relative": erreur_relative,
            "n": n
        }
    return results

# %%
def plot_exactitude(resultats, nom_element="Élément"):
    niveaux = list(resultats.keys())
    L100 = [resultats[n]["L100"] for n in niveaux]
    U100 = [resultats[n]["U100"] for n in niveaux]
    erreur_relative = [resultats[n]["Erreur relative"] for n in niveaux]

    plt.figure(figsize=(10, 5))
    plt.plot(niveaux, L100, 'o-', color='orange', label="Limite basse tolérance (%)")
    plt.plot(niveaux, U100, 'o-', color='orange', label="Limite haute tolérance (%)")
    plt.plot(niveaux, erreur_relative, 'o-', color='black', label="Erreur relative (%)")
    plt.plot(niveaux, [-15]*len(niveaux), 'o--', color='gray', label="Acceptabilité min (%)")
    plt.plot(niveaux, [15]*len(niveaux), 'o--', color='gray', label="Acceptabilité max (%)")
    plt.xlabel("Valeur certifiée")
    plt.ylabel("Erreur relative (%)")
    plt.title(f"Plan d'exactitude - {nom_element}")
    plt.grid(True)
    plt.legend()
    st.pyplot(plt.gcf())
    plt.close()

# %%
def plot_incertitude(resultats, nom_element="Élément"):
    niveaux = list(resultats.keys())
    l100 = [resultats[n]["l100"] for n in niveaux]
    u100 = [resultats[n]["u100"] for n in niveaux]
    erreur_relative = [resultats[n]["Erreur relative"] for n in niveaux]

    plt.figure(figsize=(10, 5))
    plt.plot(niveaux, l100, 'o-', color='green', label="Limite basse incertitude (%)")
    plt.plot(niveaux, u100, 'o-', color='green', label="Limite haute incertitude (%)")
    plt.plot(niveaux, erreur_relative, 'o-', color='black', label="Erreur relative (%)")
    plt.xlabel("Valeur certifiée")
    plt.ylabel("Incertitude (%)")
    plt.title(f"Plan d'incertitude - {nom_element}")
    plt.grid(True)
    plt.legend()
    st.pyplot(plt.gcf())
    plt.close()

# %% [markdown]
# ## 🧪 Interface utilisateur Streamlit

# %%
st.set_page_config(page_title="Analyse métrologique", layout="wide")
st.title("📏 Analyse d'exactitude et d'incertitude")
st.markdown("Analyse automatisée des mesures avec répétitions pour des niveaux certifiés.")

uploaded_file = st.file_uploader("📁 Charger un fichier CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📋 Aperçu des données")
    st.write(df.head())

    gamma = st.slider("Gamma (niveau de confiance)", 0.80, 0.99, 0.95, 0.01)
    lambd = st.slider("Lambda (niveau de couverture)", 0.01, 0.20, 0.05, 0.01)
    nom_element = st.text_input("Nom de l'élément analysé", value="Élément")

    if st.button("🧮 Lancer l'analyse"):
        resultats = analyse_par_niveau(df, gamma, lambd)
        st.success("✅ Analyse terminée")

        st.subheader("📈 Graphique d'exactitude")
        plot_exactitude(resultats, nom_element)

        st.subheader("📉 Graphique d'incertitude")
        plot_incertitude(resultats, nom_element)

        st.subheader("📊 Résultats détaillés")
        st.dataframe(pd.DataFrame(resultats).T)

else:
    st.info("Veuillez importer un fichier CSV avec les colonnes : `Valeur certifiée Vc`, `répétition 1`, `répétition 2`, `répétition 3`.")
