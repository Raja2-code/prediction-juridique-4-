import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Charger les données
df = pd.read_excel("Base_Litiges_PredictiveAI_ML.xlsx")

# Préparation des données
X = df.drop(columns=["Litige Probable"])
y = df["Litige Probable"]

# Encodage des colonnes catégorielles
X_encoded = pd.get_dummies(X)

# Entraînement du modèle
model = RandomForestClassifier()
model.fit(X_encoded, y)

st.title("🧠 Prédiction Automatique de Litiges")

st.markdown("Ce formulaire permet de prédire si un nouveau dossier présente un risque de **litige probable**.")

# Formulaire utilisateur
with st.form("prediction_form"):
    objet = st.text_input("Objet du litige")
    tribunal = st.selectbox("Tribunal", df["Tribunal"].unique())
    demandeur = st.text_input("Demandeur")
    defendeur = st.text_input("Défendeur")
    avocat_demandeur = st.selectbox("Avocat Demandeur", df["Avocat Demandeur"].unique())
    avocat_defendeur = st.selectbox("Avocat Défendeur", df["Avocat Défendeur"].unique())
    date_depot = st.date_input("Date de dépôt")
    montant_reclame = st.number_input("Montant réclamé", min_value=0.0)
    clause_confidentialite = st.selectbox("Clause de confidentialité", ["Oui", "Non"])
    retards = st.selectbox("Retards signalés", ["Oui", "Non"])
    mediation = st.selectbox("Médiation tentée", ["Oui", "Non"])
    montant_accorde = st.number_input("Montant accordé", min_value=0.0)
    president = st.selectbox("Président de la chambre", df["Président de la chambre"].unique())
    decision = st.selectbox("Décision rendue", df["Décision rendue"].unique())

    submitted = st.form_submit_button("Prédire le litige")

if submitted:
    new_data = pd.DataFrame({
        "Objet du litige": [objet],
        "Tribunal": [tribunal],
        "Demandeur": [demandeur],
        "Défendeur": [defendeur],
        "Avocat Demandeur": [avocat_demandeur],
        "Avocat Défendeur": [avocat_defendeur],
        "Date de dépôt": [date_depot],
        "Montant réclamé": [montant_reclame],
        "Clause de confidentialité": [clause_confidentialite],
        "Retards signalés": [retards],
        "Médiation tentée": [mediation],
        "Montant accordé": [montant_accorde],
        "Président de la chambre": [president],
        "Décision rendue": [decision],
    })

    # Encodage du nouvel exemple pour correspondre aux colonnes d'entraînement
    new_data_encoded = pd.get_dummies(new_data)
    new_data_encoded = new_data_encoded.reindex(columns=X_encoded.columns, fill_value=0)

    prediction = model.predict(new_data_encoded)[0]
    st.subheader("Résultat de la prédiction :")
    if prediction == "Oui":
        st.error("⚠️ Risque élevé de litige probable")
    else:
        st.success("✅ Litige peu probable")
