import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np
from tensorflow.keras.models import load_model as keras_load_model  # Renommer l'import
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import io
import sklearn
import joblib
import pickle
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_array
from sklearn.preprocessing import LabelEncoder


st.title("Détection d'anomalies dans les cotisations URSSAF")


versement_mobilite = { '87005' : 1.80 }
models_info = {
    '6000': {
        'type' : 'joblib',
        'model': './6000.pkl',
        'numeric_cols': ['Rub 6000',  '6000Taux'],
        'categorical_cols': ['Frontalier'],
        'target_col': 'anomalie_frontalier'
    },
    '6002': {
        'type' : 'joblib',
        'model': './6002.pkl',
        'numeric_cols': ['Rub 6002',  '6002Taux'],
        'categorical_cols': ['Region'],
        'target_col': 'anomalie_alsace_moselle'
    },
    '6082': {
        'type' : 'joblib',
        'model': './6082.pkl',
        'numeric_cols': ['Rub 6082', '6082Taux'],
        'categorical_cols': ['Statut de salariés', 'Frontalier'],
        'target_col': 'anomalie_csg'
    },
    '6084': {
        'type' : 'joblib',
        'model': './6084.pkl',
        'numeric_cols': ['Rub 6084', '6084Taux'],
        'categorical_cols': ['Statut de salariés', 'Frontalier'],
        'target_col': 'anomalie_crds'
    },
    '7001': {
            'type' : 'joblib',
            'model': './7001.pkl',
            'numeric_cols': ['Matricule', 'Absences par Jour', 'Absences par Heure', 'PLAFOND CUM', 'ASSIETTE CUM', 'MALADIE CUM', '7001Base', '7001Taux 2', '7001Montant Pat.'],
            'categorical_cols': ['Statut de salariés'],
            'target_col': 'anomalie_maladie_reduite'
        },
    '7002': {
        'type' : 'joblib',
        'model': './7002.pkl',
        'numeric_cols': ['Rub 7002',  '7002Taux 2' , 'ASSIETTE CUM','PLAFOND CUM'],
        'categorical_cols': ['Statut de salariés'],
        'target_col': 'anomalie_maladie_diff'
    },

    '7010': {
        'type' : 'joblib',
        'model': './7010.pkl',
        'numeric_cols': ['Rub 7010',  '7010Taux 2','7010Taux' ,'Effectif'],
        'categorical_cols': ['Statut de salariés'],
        'target_col': 'Anomalie_7010'
    },

    '7015': {
        'type' : 'joblib',
        'model': './7015.pkl',
        'numeric_cols': ['Rub 7015','7015Taux', '7015Taux 2' ,'Effectif'],
        'categorical_cols': ['Statut de salariés'],
        'target_col': 'Anomalie_7015'
    },

    '7020': {
        'type' : 'joblib',
        'model': './7020.pkl',
        'numeric_cols': ['Rub 7020',  '7020Taux 2' ,'Effectif'],
        'categorical_cols': ['Statut de salariés'],
        'target_col': 'anomalie_fnal'
    },

    '7025': {
        'type' : 'joblib',
        'model': './7025.pkl',
        'numeric_cols': ['Rub 7025','7025Taux 2','ASSIETTE CUM','PLAFOND CUM'],
        'categorical_cols': ['Statut de salariés'],
        'target_col': 'anomalie_allocation_diff'
    },
    '7030': {
        'type' : 'joblib',
        'model': './7030.pkl',
        'numeric_cols': ['PLAFOND CUM', 'ASSIETTE CUM','7030Taux 2', 'Rub 7030'],
        'categorical_cols': [],
        'target_col': 'anomalie_allocation_reduite'
    },
    #'7035': {
    #    'type' : 'joblib',
    #    'model': './7035.pkl',
    #    'numeric_cols': ['Rub 7035','7035Taux 2'],
    #    'categorical_cols': ['Statut de salariés'],
    #    'target_col': '7035 Fraud'
    #},
    '7040': {
        'type' : 'joblib',
        'model': './7040.pkl',
        'numeric_cols': ['Effectif', '7040Taux 2' ,'Rub 7040'],
        'categorical_cols': ['Statut de salariés'],
        'target_col': 'anomalie_7040taux 2'
    },
    '7045': {
        'type' : 'joblib',
        'model': './7045.pkl',
        'numeric_cols': ['Effectif', '7045Taux 2'],
        'categorical_cols': ['Etablissement'],
        'target_col': 'anomalie_transport'
    },
    '7050': {
        'type' : 'joblib',
        'model': './7050.pkl',
        'numeric_cols': ['Effectif', '7050Taux 2'],
        'categorical_cols': ['Etablissement'],
        'target_col': 'anomalie_cotisation_accident'
    }
}


csv_upload = st.file_uploader("Entrez votre bulletin de paie (Format csv)", type=['csv'])

def apply_versement_conditions(df):
    required_columns = ['Versement Mobilite Taux', 'Effectif', 'Code Insee', 'Versement mobilite Base']
    
    if not all(col in df.columns for col in required_columns):
        df['Mobilite_Anomalie'] = False
        df['mobilite Fraud'] = 0
        return df

    df['Versement Mobilite Taux Avant'] = df['Versement Mobilite Taux']
    df['Versement Mobilite Taux'] = df.apply(
        lambda row: 0 if row['Effectif'] < 11 else versement_mobilite.get(row['Code Insee'], row['Versement Mobilite Taux']),
        axis=1
    )
    df['Versement Mobilite Montant Pat. Calcule'] = (df['Versement mobilite Base'] * df['Versement Mobilite Taux'] / 100).round(2)
    df['Mobilite_Montant_Anomalie'] = df['Versement Mobilite Montant Pat.'] != df['Versement Mobilite Montant Pat. Calcule']
    df['Mobilite_Anomalie'] = (df['Versement Mobilite Taux Avant'] != df['Versement Mobilite Taux']) | df['Mobilite_Montant_Anomalie']
    df['mobilite Fraud'] = df['Mobilite_Anomalie'].astype(int)

    return df



def encode_categorical_columns(df, categorical_cols):
    encoder = OneHotEncoder(drop='first', sparse_output=False)  # Remplacement de 'sparse' par 'sparse_output'
    encoded_cols = encoder.fit_transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(categorical_cols))
    df = df.drop(columns=categorical_cols)  # Supprimer les colonnes catégorielles originales
    df = pd.concat([df, encoded_df], axis=1)
    return df

def load_model(model_info):
    model_path = model_info['model']
    
    try:
        # Charger un modèle joblib qui pourrait être un dictionnaire ou un modèle direct
        model = joblib.load(model_path)
        print(f"Contenu du modèle {model_path}: {type(model)}")
        
        # Si le modèle est un dictionnaire, nous devons extraire les bons objets
        if isinstance(model, dict):
            # Pour le modèle 7001, nous allons travailler avec 'iso_forest' et 'scaler'
            if 'iso_forest' in model and 'scaler' in model:
                return model  # Retourner directement le dictionnaire contenant ces objets
            else:
                raise ValueError(f"Le modèle {model_path} est un dictionnaire mais ne contient pas de clés appropriées.")
        
        # Si c'est un modèle simple avec une méthode 'predict'
        if hasattr(model, 'predict'):
            return model
        else:
            raise ValueError(f"Le modèle chargé à partir de {model_path} ne semble pas avoir de méthode 'predict'.")
    
    except Exception as e:
        raise ValueError(f"Erreur lors du chargement du modèle joblib : {str(e)}")



def process_7001(df, model_name, info, anomalies_report, model_anomalies):
    # Charger les modèles et scalers depuis le fichier 7001.pkl
    model_dict = load_model(info)
    
    if model_dict is None:
        st.error(f"Le modèle {model_name} n'a pas été chargé correctement.")
        return
    
    # Extraire les objets 'iso_forest' et 'scaler'
    iso_forest = model_dict.get('iso_forest', None)
    scaler = model_dict.get('scaler', None)

    if iso_forest is None or scaler is None:
        st.warning(f"Le modèle 7001 est incomplet (iso_forest ou scaler manquant).")
        return

    # Colonne à traiter : '7001Taux 2'
    colonne = '7001Taux 2'

    # Créer une colonne 'is_nan' pour signaler les NaN, mais NE PAS remplacer les NaN par 0
    df['is_nan'] = df[colonne].isna().astype(int)

    # Filtrer les lignes où la colonne '7001Taux 2' n'est pas NaN (ignorer les lignes avec NaN)
    df_non_nan = df[df[colonne].notna()].copy()

    if df_non_nan.empty:
        st.write(f"Aucune donnée valide à traiter pour le modèle {model_name}.")
        return

    # Vérifier les colonnes que le scaler a vues lors de l'entraînement
    scaler_features = scaler.feature_names_in_

    # Ajouter les colonnes manquantes dans les données test si elles sont présentes dans le scaler
    for feature in scaler_features:
        if feature not in df_non_nan.columns:
            df_non_nan[feature] = 0  # ou np.nan selon ce qui est approprié

    # Filtrer les colonnes de df_non_nan pour correspondre exactement aux colonnes vues par le scaler
    df_scaled = df_non_nan[scaler_features]

    try:
        # Transformer les données avec le scaler
        X_scaled_test = scaler.transform(df_scaled)

        # Appliquer Isolation Forest pour prédire les anomalies
        df_non_nan['Anomaly_IF'] = iso_forest.predict(X_scaled_test)
        df_non_nan['Anomaly_IF'] = df_non_nan['Anomaly_IF'].map({1: 0, -1: 1})  # -1 = anomalie, 1 = normal

        # Réintégrer les résultats dans le DataFrame original
        df['Anomaly_IF'] = np.nan  # Initialiser avec NaN
        df.loc[df_non_nan.index, 'Anomaly_IF'] = df_non_nan['Anomaly_IF']

        # Ajouter les anomalies dans le rapport
        for index in df_non_nan.index:
            if df_non_nan.loc[index, 'Anomaly_IF'] == 1:
                anomalies_report.setdefault(index, set()).add(model_name)
                model_anomalies[model_name] = model_anomalies.get(model_name, 0) + 1

    except ValueError as e:
        st.error(f"Erreur lors de la transformation des données avec le scaler : {e}")




def process_model_with_average(df, model_name, info, anomalies_report, model_anomalies):
    # Charger le modèle et les moyennes depuis le fichier pkl
    model_and_means = joblib.load(info['model'])
    
    # Extraire le modèle et les moyennes
    model = model_and_means.get('model', None)  # Récupérer le modèle depuis le dictionnaire
    average_predicted_taux_by_establishment = model_and_means.get('average_predicted_taux_by_establishment', None)  # Moyennes par établissement

    if model is None:
        st.error(f"Erreur : le modèle pour {model_name} n'a pas été trouvé dans le fichier.")
        return

    if average_predicted_taux_by_establishment is None:
        st.error(f"Erreur : les moyennes par établissement pour {model_name} n'ont pas été trouvées dans le fichier.")
        return

    df_filtered = df.copy()

    # Vérifier si la colonne 'Etablissement' est présente
    if 'Etablissement' not in df_filtered.columns:
        st.error(f"La colonne 'Etablissement' est manquante dans les nouvelles données pour {model_name}.")
        return

    # Appliquer la règle : si Effectif < 11, définir 'Taux 2' à 0 (différence entre 7045Taux 2 et 7050Taux 2 est dans info['numeric_cols'])
    taux_col = info['numeric_cols'][1]  # Le nom de la colonne du taux, qui change selon le modèle
    df_filtered['rule_based_taux'] = df_filtered.apply(lambda row: 0 if row['Effectif'] < 11 else row[taux_col], axis=1)

    # Créer la colonne 'taux_per_effectif'
    df_filtered['taux_per_effectif'] = df_filtered[taux_col] / df_filtered['Effectif']

    # Remplacer les valeurs manquantes par 0
    df_filtered = df_filtered.fillna(0)

    # Préparer les colonnes nécessaires pour le modèle
    X_new = df_filtered[['Etablissement', 'Effectif', 'rule_based_taux', 'taux_per_effectif']]

    # Effectuer les prédictions avec le pipeline
    try:
        df_filtered['predicted_taux'] = model.predict(X_new)
    except ValueError as e:
        st.error(f"Erreur lors de la prédiction avec le modèle {model_name} : {str(e)}")
        return

    # Calculer la différence entre le taux réel et la moyenne prédite
    df_filtered['taux_diff'] = df_filtered.apply(
        lambda row: abs(row[taux_col] - average_predicted_taux_by_establishment.get(row['Etablissement'], 0)), axis=1
    )

    # Définir une anomalie si la différence dépasse un seuil
    threshold = 0.001
    df_filtered['Anomalie'] = df_filtered['taux_diff'].apply(lambda x: 'Oui' if x > threshold else 'Non')

    # Enregistrer les anomalies détectées dans le rapport
    for index, row in df_filtered.iterrows():
        if row['Anomalie'] == 'Oui':
            anomalies_report.setdefault(index, set()).add(model_name)
            model_anomalies[model_name] = model_anomalies.get(model_name, 0) + 1


def process_model(df, model_name, info, anomalies_report, model_anomalies):
    df_filtered = df

    if df_filtered.empty:
        st.write(f"Aucune donnée à traiter pour le modèle {model_name}.")
        return

    required_columns = info['numeric_cols'] + info['categorical_cols']
    missing_columns = [col for col in required_columns if col not in df_filtered.columns]

    if missing_columns:
        st.error(f"Le modèle {model_name} manque des colonnes : {', '.join(missing_columns)}")
        return

    # Filtrer uniquement les colonnes nécessaires
    df_inputs = df_filtered[required_columns].copy()

    # Spécifique aux modèles : Remplacer les NaN par 0 pour certaines colonnes
    if model_name == '6000':
        df_inputs['Rub 6000'] = df_inputs['Rub 6000'].fillna(0)
        df_inputs['6000Taux'] = df_inputs['6000Taux'].fillna(0)
    elif model_name == '6082':
        df_inputs['Rub 6082'] = df_inputs['Rub 6082'].fillna(0)
        df_inputs['6082Taux'] = df_inputs['6082Taux'].fillna(0)
    elif model_name == '6084':
        df_inputs['Rub 6084'] = df_inputs['Rub 6084'].fillna(0)
        df_inputs['6084Taux'] = df_inputs['6084Taux'].fillna(0)

    # Remplir les valeurs manquantes pour les autres colonnes numériques
    df_inputs[info['numeric_cols']] = df_inputs[info['numeric_cols']].fillna(df_inputs[info['numeric_cols']].mean())

    # Encodage des colonnes catégorielles
    if model_name in ['6082', '6084']:
        label_encoder_statut = LabelEncoder()
        label_encoder_frontalier = LabelEncoder()

        # Encodage des colonnes catégorielles
        if 'Statut de salariés' in df_inputs.columns:
            df_inputs['Statut de salariés'] = label_encoder_statut.fit_transform(df_inputs['Statut de salariés'])

        if 'Frontalier' in df_inputs.columns:
            df_inputs['Frontalier'] = label_encoder_frontalier.fit_transform(df_inputs['Frontalier'])

    elif info['categorical_cols']:
        # Utilisation de OneHotEncoder pour les autres modèles
        encoder = OneHotEncoder(drop='first', sparse_output=False)
        encoded_categorical = encoder.fit_transform(df_inputs[info['categorical_cols']])
        df_encoded = pd.DataFrame(encoded_categorical, index=df_inputs.index, columns=encoder.get_feature_names_out(info['categorical_cols']))
        df_inputs = df_inputs.drop(columns=info['categorical_cols'])
        df_inputs = pd.concat([df_inputs, df_encoded], axis=1)

    # Charger le modèle
    model = load_model(info)

    # Vérification de l'alignement des colonnes avec celles vues pendant l'entraînement
    if hasattr(model, 'feature_names_in_'):
        # Réordonner les colonnes en fonction de celles du modèle
        df_inputs = df_inputs.reindex(columns=model.feature_names_in_, fill_value=0)

    try:
        y_pred = model.predict(df_inputs)
    except ValueError as e:
        st.error(f"Erreur avec le modèle {model_name} (joblib) : {str(e)}")
        return

    if y_pred is not None:
        df.loc[df_filtered.index, f'{model_name}_Anomalie_Pred'] = y_pred
        num_anomalies = np.sum(y_pred)
        model_anomalies[model_name] = num_anomalies

        for index in df_filtered.index:
            if y_pred[df_filtered.index.get_loc(index)] == 1:
                anomalies_report.setdefault(index, set()).add(model_name)





def detect_anomalies(df):
    anomalies_report = {}
    model_anomalies = {}

    for model_name, info in models_info.items():
        if model_name in ['7045', '7050']:
            process_model_with_average(df, model_name, info, anomalies_report, model_anomalies)
        elif model_name == '7001':
            process_7001(df, model_name, info, anomalies_report, model_anomalies)
        else:
            process_model(df, model_name, info, anomalies_report, model_anomalies)

    st.write("**Rapport d'anomalies détectées :**")
    total_anomalies = len(anomalies_report)
    st.write(f"**Total des lignes avec des anomalies :** {total_anomalies}")

    report_content = io.StringIO()
    report_content.write("Rapport d'anomalies détectées :\n\n")
    report_content.write(f"Total des lignes avec des anomalies : {total_anomalies}\n")
    for model_name, count in model_anomalies.items():
        report_content.write(f"Un nombre de {int(count)} anomalies a été détecté pour la cotisation {model_name}.\n")

    for line_index, models in anomalies_report.items():
        matricule = df.loc[line_index, 'Matricule']
        report_content.write(f"Matricule {matricule} : anomalie dans les cotisations {', '.join(sorted(models))}\n")

    report_content.seek(0)
    return report_content





def charger_dictionnaire(fichier):
    dictionnaire = {}
    try:
        with open(fichier, 'r', encoding='utf-8') as f:
            for ligne in f:
                code, description = ligne.strip().split(' : ', 1)
                dictionnaire[code] = description
    except FileNotFoundError:
        st.error(f"Le fichier {fichier} n'a pas été trouvé.")
    except Exception as e:
        st.error(f"Erreur lors du chargement du dictionnaire : {e}")
    return dictionnaire

def inspect_pkl_file(model_info):
    model_path = model_info['model']
    
    try:
        # Charger un modèle joblib
        model = joblib.load(model_path)
        
        if isinstance(model, dict):
            st.write(f"Le fichier {model_path} est un dictionnaire avec les clés suivantes : {list(model.keys())}")
        else:
            st.write(f"Le fichier {model_path} n'est pas un dictionnaire, c'est un objet de type {type(model)}")
    
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle joblib : {str(e)}")

# Exemple pour 7001.pkl
inspect_pkl_file(models_info['7001'])

if csv_upload:
    dictionnaire = charger_dictionnaire('./Dictionnaire.txt')
    df_dictionnaire = pd.DataFrame(list(dictionnaire.items()), columns=['Code', 'Description'])

    st.header('Dictionnaire des Codes et Descriptions')
    st.dataframe(df_dictionnaire)
    try:
        df = pd.read_csv(csv_upload, encoding='utf-8', on_bad_lines='skip')
    except pd.errors.EmptyDataError:
        st.error("Le fichier CSV est vide ou mal formaté.")
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(csv_upload, encoding='ISO-8859-1', on_bad_lines='skip')
        except Exception as e:
            st.error(f"Erreur lors de la lecture du fichier CSV : {e}")
    except Exception as e:
        st.error(f"Erreur inattendue lors de la lecture du fichier CSV : {e}")
    else:
        df.columns = df.columns.str.strip()  # Nettoyer les espaces dans les noms de colonnes
        report_content = detect_anomalies(df)
        st.download_button(
            label="Télécharger le rapport d'anomalies",
            data=report_content.getvalue(),
            file_name='anomalies_report.txt',
            mime='text/plain'
        )
