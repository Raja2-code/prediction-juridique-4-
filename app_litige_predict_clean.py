
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Charger les données d'entraînement
df = pd.read_excel("Base_Litiges_PredictiveAI_ML.xlsx")

# Encodage des variables catégorielles
le_dict = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

# Séparer X et y
X = df.drop(columns=["Issue"])
y = df["Issue"]

# Entraîner le modèle
model = RandomForestClassifier()
model.fit(X, y)

# Interface utilisateur Streamlit
st.title("🔮 Prédiction du Résultat d’un Litige")

# Champs de saisie utilisateur
nouveau_dossier = {}
for col in X.columns:
    val = st.text_input(f"{col}")
    if col in le_dict:
        if val in le_dict[col].classes_:
            nouveau_dossier[col] = le_dict[col].transform([val])[0]
        else:
            st.warning(f"Valeur inconnue pour '{col}'. Veuillez utiliser une valeur connue : {list(le_dict[col].classes_)}")
            nouveau_dossier[col] = 0
    else:
        try:
            nouveau_dossier[col] = float(val)
        except:
            nouveau_dossier[col] = 0

if st.button("Prédire le résultat"):
    try:
        prediction = model.predict([list(nouveau_dossier.values())])[0]
        st.success(f"✅ Résultat prédit : {prediction}")
    except Exception as e:
        st.error("Erreur lors de la prédiction. Vérifiez vos entrées.")
