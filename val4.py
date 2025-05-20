
import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import io

# --- Configuration de la page ---
st.set_page_config(page_title="Analyse Exactitude & Incertitude", layout="wide")

# --- Titre principal ---
st.title("🔬 Analyse d'Exactitude & d'Incertitude")
st.markdown("Cette application permet de téléverser plusieurs fichiers CSV correspondant à différents éléments, puis d’analyser l’exactitude et l’incertitude de leurs mesures.")
# --- Explication contextuelle ---
with st.expander("ℹ️ Qu'est-ce que la validation analytique ?", expanded=True):
    st.markdown("""
    La **validation analytique** permet de s'assurer qu'une méthode de mesure produit des résultats fiables, reproductibles et acceptables pour son usage prévu.

    Dans ce cadre :
    - L’**exactitude** mesure l’écart entre la valeur mesurée et la valeur de référence.
    - L’**incertitude** évalue la dispersion des résultats autour de cette valeur.
    
    Cette application utilise plusieurs niveaux de concentration certifiée pour vérifier que les performances analytiques respectent les critères d’acceptabilité.
    """)


# --- Upload multiples fichiers ---
uploaded_files = st.file_uploader("📂 Téléversez les fichiers CSV pour chaque élément :", type="csv", accept_multiple_files=True)

if not uploaded_files:
    st.info("Veuillez téléverser un ou plusieurs fichiers CSV pour commencer.")
    st.stop()

# --- Lecture robuste de chaque fichier ---
data_dict = {}

def try_read_csv(content, encodings=["utf-8", "latin-1"], sep=";"):
    for enc in encodings:
        try:
            return pd.read_csv(io.StringIO(content.decode(enc)), sep=sep)
        except Exception:
            continue
    return None

for file in uploaded_files:
    content = file.read()
    df = try_read_csv(content)
    if df is not None and "Valeur certifiée Vc" in df.columns:
        for col in ["répétition 1", "répétition 2", "répétition 3", "Moyenne"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(",", ".").astype(float)
        data_dict[file.name] = df

if not data_dict:
    st.error("Aucun des fichiers ne contient des colonnes attendues ('Valeur certifiée Vc', 'répétitions', etc.).")
    st.stop()

# --- Choix de l'élément à analyser ---
choix_element = st.selectbox("🔍 Choisissez un élément à analyser :", list(data_dict.keys()))
df = data_dict[choix_element]

# --- Paramètres utilisateur ---
with st.sidebar:
    st.header("⚙️ Paramètres d’analyse")
    gamma = st.slider("Niveau de confiance γ", min_value=0.80, max_value=0.99, value=0.95, step=0.01)
    lambd = st.slider("Paramètre λ", min_value=0.01, max_value=0.2, value=0.05, step=0.01)
    nom_element = st.text_input("Nom affiché de l’élément :", choix_element.split('.')[0])

# --- Fonction d’analyse ---
def analyse_par_niveau(df, gamma=0.95, lambd=0.05):
    grouped = df.groupby("Valeur certifiée Vc")
    results = {}

    for niveau, group in grouped:
        repetitions = group[["répétition 1", "répétition 2", "répétition 3"]].values.flatten()
        valeurs = repetitions[~np.isnan(repetitions)]

        n = len(valeurs)
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
            "l100": round(l100, 2),
            "u100": round(u100, 2),
            "L100": round(L100, 2),
            "U100": round(U100, 2),
            "Erreur relative": round(erreur_relative, 2),
            "n": n
        }
    return results

# --- Analyse ---
st.markdown("---")
st.subheader(f"📊 Analyse de l’élément : **{nom_element}**")

resultats = analyse_par_niveau(df, gamma=gamma, lambd=lambd)

# --- Fonctions de tracé ---
def plot_exactitude(resultats, nom_element="Élément"):
    niveaux = list(resultats.keys())
    L100 = [resultats[n]["L100"] for n in niveaux]
    U100 = [resultats[n]["U100"] for n in niveaux]
    erreur_relative = [resultats[n]["Erreur relative"] for n in niveaux]

    plt.figure(figsize=(10, 6))
    plt.plot(niveaux, L100, 'o-', color='orange', label="Limite basse tolérance (%)")
    plt.plot(niveaux, U100, 'o-', color='orange', label="Limite haute tolérance (%)")
    plt.plot(niveaux, erreur_relative, 'o-', color='black', label="Erreur relative (%)")
    plt.axhline(-15, color='gray', linestyle='--', label="Acceptabilité min (%)")
    plt.axhline(15, color='gray', linestyle='--', label="Acceptabilité max (%)")
    plt.xlabel("Valeur certifiée")
    plt.ylabel("Erreur relative (%)")
    plt.title(f"Plan d'exactitude - {nom_element}")
    plt.grid(True)
    plt.legend()
    st.pyplot(plt.gcf())

def plot_incertitude(resultats, nom_element="Élément"):
    niveaux = list(resultats.keys())
    l100 = [resultats[n]["l100"] for n in niveaux]
    u100 = [resultats[n]["u100"] for n in niveaux]
    erreur_relative = [resultats[n]["Erreur relative"] for n in niveaux]

    plt.figure(figsize=(10, 6))
    plt.plot(niveaux, l100, 'o-', color='green', label="Limite basse incertitude (%)")
    plt.plot(niveaux, u100, 'o-', color='green', label="Limite haute incertitude (%)")
    plt.plot(niveaux, erreur_relative, 'o-', color='black', label="Erreur relative (%)")
    plt.xlabel("Valeur certifiée")
    plt.ylabel("Incertitude (%)")
    plt.title(f"Plan d'incertitude - {nom_element}")
    plt.grid(True)
    plt.legend()
    st.pyplot(plt.gcf())

# --- Graphiques ---
st.markdown("### 📉 Plan d’exactitude")
plot_exactitude(resultats, nom_element=nom_element)

st.markdown("### 📈 Plan d’incertitude")
plot_incertitude(resultats, nom_element=nom_element)


