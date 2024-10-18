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
        'type': 'joblib',
        'model': './6082.pkl',
        'numeric_cols': ['Rub 6082', '6082Taux'],
        'categorical_cols': ['Statut de salariés', 'Frontalier'],
        'target_col': 'anomalie_csg'
    },
    '6084': {
        'type': 'joblib',
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
        'model': './7002.pkl',  # Utilisez le chemin vers votre modèle
        'numeric_cols': ['SMIC M CUM', '7002Taux 2', 'ASSIETTE CUM'],
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
            'type': 'joblib',
            'model': './7025.pkl',
            'numeric_cols': ['7025Taux 2', 'ASSIETTE CUM', 'PLAFOND CUM'],
            'categorical_cols': ['Statut de salariés'],
            'target_col': '7025Taux 2'
        },
    '7030': {
        'type' : 'joblib',
        'model': './7030.pkl',
        'numeric_cols': ['PLAFOND CUM', 'ASSIETTE CUM','7030Taux 2', 'Rub 7030'],
        'categorical_cols': [],
        'target_col': 'anomalie_allocation_reduite'
    },
    '7035': {
        'type' : 'joblib',
        'model': './7035.pkl',
        'numeric_cols': ['Rub 7035','7035Taux 2'],
        'categorical_cols': ['Statut de salariés'],
        'target_col': '7035 Fraud'
    },
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

        # Compter les anomalies détectées
        num_anomalies = np.sum(df_non_nan['Anomaly_IF'])
        model_anomalies[model_name] = num_anomalies

        # Ajouter les anomalies dans le rapport
        for index in df_non_nan.index:
            if df_non_nan.loc[index, 'Anomaly_IF'] == 1:
                anomalies_report.setdefault(index, set()).add(model_name)

    except ValueError as e:
        st.error(f"Erreur lors de la transformation des données avec le scaler : {e}")




def process_model_with_average(df, model_name, info, anomalies_report, model_anomalies):
    # Charger le modèle et les moyennes depuis le fichier pkl
    model_and_means = joblib.load(info['model'])
    
    # Extraire le modèle et les moyennes
    model = model_and_means.get('model', None)
    average_predicted_taux_by_establishment = model_and_means.get('average_predicted_taux_by_establishment', None)

    if model is None:
        st.error(f"Erreur : le modèle pour {model_name} n'a pas été trouvé dans le fichier.")
        return

    if average_predicted_taux_by_establishment is None:
        st.error(f"Erreur : les moyennes par établissement pour {model_name} n'ont pas été trouvées dans le fichier.")
        return

    df_filtered = df.copy()

    if 'Etablissement' not in df_filtered.columns:
        st.error(f"La colonne 'Etablissement' est manquante dans les nouvelles données pour {model_name}.")
        return

    taux_col = info['numeric_cols'][1]
    df_filtered['rule_based_taux'] = df_filtered.apply(lambda row: 0 if row['Effectif'] < 11 else row[taux_col], axis=1)
    df_filtered['taux_per_effectif'] = df_filtered[taux_col] / df_filtered['Effectif']
    df_filtered = df_filtered.fillna(0)
    X_new = df_filtered[['Etablissement', 'Effectif', 'rule_based_taux', 'taux_per_effectif']]

    try:
        df_filtered['predicted_taux'] = model.predict(X_new)
    except ValueError as e:
        st.error(f"Erreur lors de la prédiction avec le modèle {model_name} : {e}")
        return

    df_filtered['taux_diff'] = df_filtered.apply(
        lambda row: abs(row[taux_col] - average_predicted_taux_by_establishment.get(row['Etablissement'], 0)), axis=1
    )

    threshold = 0.01
    df_filtered['Anomalie'] = df_filtered['taux_diff'].apply(lambda x: 'Oui' if x > threshold else 'Non')

    # Enregistrer les anomalies détectées dans le rapport
    for index, row in df_filtered.iterrows():
        if row['Anomalie'] == 'Oui':
            if index not in anomalies_report:
                anomalies_report[index] = {}  # Utilisation d'un dictionnaire pour chaque ligne
            anomalies_report[index][model_name] = f"Anomalie détectée pour {model_name} avec une différence de taux."
            model_anomalies[model_name] = model_anomalies.get(model_name, 0) + 1


def verify_montant_conditions(df, model_name, anomalies_report, model_anomalies):
    """
    Vérifie les montants patronaux et salariaux pour un modèle donné et ajoute des anomalies si nécessaire.
    
    Parameters:
    df (pd.DataFrame): Le DataFrame contenant les données.
    model_name (str): Le nom du modèle (par exemple, '7001', '7020', etc.).
    anomalies_report (dict): Dictionnaire contenant les anomalies détectées.
    model_anomalies (dict): Dictionnaire pour compter le nombre d'anomalies par modèle.
    """
    
    montant_pat_col = f"{model_name}Montant Pat."
    taux_2_col = f"{model_name}Taux 2"
    montant_sal_col = f"{model_name}Montant Sal."
    taux_col = f"{model_name}Taux"
    rub_col = f"{model_name}Base"

    tolérance = 0.11  # La marge de différence tolérable exacte

    # Parcourir les lignes du DataFrame pour appliquer les vérifications
    for index, row in df.iterrows():
        # Initialiser les détails de l'anomalie pour cette ligne si ce n'est pas encore un dictionnaire
        if index not in anomalies_report:
            anomalies_report[index] = {}

        anomaly_detected = False  # Flag pour savoir si une anomalie a été détectée

        # Vérification pour Montant Pat.
        if montant_pat_col in df.columns and taux_2_col in df.columns and rub_col in df.columns:
            if not pd.isna(row[montant_pat_col]) and row[montant_pat_col] != 0:
                montant_pat_calcule = round(row[taux_2_col] * row[rub_col] / 100, 2)
                montant_pat_calcule2 = round(row[taux_2_col] * row[rub_col] / 100 * -1, 2)
                montant_pat_reel = round(row[montant_pat_col], 2)
                
                if not np.isclose(montant_pat_reel, montant_pat_calcule, atol=tolérance) and not np.isclose(montant_pat_reel, montant_pat_calcule2, atol=tolérance):
                    # Ajouter l'anomalie si le montant calculé ne correspond pas
                    anomalies_report[index][model_name] = f"Anomalie détectée ({model_name})"
                    anomaly_detected = True  # Flag anomaly detected

        # Vérification pour Montant Sal.
        if montant_sal_col in df.columns and taux_col in df.columns and rub_col in df.columns:
            if not pd.isna(row[montant_sal_col]) and row[montant_sal_col] != 0:
                montant_sal_calcule = round(row[taux_col] * row[rub_col] / 100 * -1, 2)
                montant_sal_calcule2 = round(row[taux_col] * row[rub_col] / 100, 2)
                montant_sal_reel = round(row[montant_sal_col], 2)
                
                if not np.isclose(montant_sal_reel, montant_sal_calcule, atol=tolérance) and not np.isclose(montant_sal_reel, montant_sal_calcule2, atol=tolérance):
                    # Ajouter l'anomalie si le montant calculé ne correspond pas
                    anomalies_report[index][model_name] = f"Anomalie détectée ({model_name})"
                    anomaly_detected = True  # Flag anomaly detected

        # Si une anomalie est détectée pour le modèle, ne comptez qu'une seule anomalie pour ce modèle et cette ligne
        if anomaly_detected:
            model_anomalies[model_name] = model_anomalies.get(model_name, 0) + 1



def process_model_generic(df, model_name, info, anomalies_report, model_anomalies):

    def predict_case(df_case, model_data, numeric_cols, taux_column):
        # Vérifier si le DataFrame est vide
        if df_case.empty:
            print(f"Aucune donnée disponible pour le cas {model_data['case_name']}.")
            return pd.DataFrame()  # Retourner un DataFrame vide

        # S'assurer que la colonne 'taux_per_statut' est présente
        if 'taux_per_statut' not in df_case.columns:
            df_case['taux_per_statut'] = df_case[taux_column]  # Utiliser la colonne taux associée

        # Assurez-vous que la colonne 'taux_per_statut' est incluse dans numeric_cols
        if 'taux_per_statut' not in numeric_cols:
            numeric_cols.append('taux_per_statut')

        # S'assurer que les colonnes numériques sont présentes et non NaN dans les nouvelles données
        df_case[numeric_cols] = df_case[numeric_cols].fillna(0)

        # Définir les features (X) en fonction des colonnes numériques spécifiées
        X_case = df_case[numeric_cols]

        # Faire des prédictions sur les nouvelles données pour ce cas
        df_case['predicted_taux'] = model_data['model'].predict(X_case)

        # Calculer la différence entre le taux réel et la moyenne prédite
        df_case['taux_diff'] = abs(df_case[taux_column] - model_data['average_predicted_taux'])

        # Définir une anomalie si la différence dépasse un seuil
        threshold = 0.01
        df_case['Anomalie'] = df_case['taux_diff'].apply(lambda x: 'Oui' if x > threshold else 'Non')

        return df_case[['Matricule'] + numeric_cols + ['Anomalie']]

    # Charger le modèle et les moyennes depuis le fichier pkl
    model_and_means = joblib.load(info['model'])

    # Extraire les modèles et moyennes pour chaque cas (case_1_1, case_1_2, case_2)
    if 'case_1_1' in model_and_means and 'case_1_2' in model_and_means and 'case_2' in model_and_means:
        case_1_1_data = model_and_means['case_1_1']
        case_1_2_data = model_and_means['case_1_2']
        case_2_data = model_and_means['case_2']
    else:
        print(f"Erreur : les cases 'case_1_1', 'case_1_2', ou 'case_2' sont manquants dans {model_name}.")
        return

    # Calcul des nouvelles colonnes dans le DataFrame
    if model_name == '7002':
        df['PLAFOND CUM MALADIE'] = 2.5 * df['SMIC M CUM']
        plafond_column = 'PLAFOND CUM MALADIE'
        taux_column = '7002Taux 2'
    elif model_name == '7025':
        df['PLAFOND CUM ALLOC'] = 3.5 * df['SMIC M CUM']
        plafond_column = 'PLAFOND CUM ALLOC'
        taux_column = '7025Taux 2'
    
    # Gestion des cas spécifiques (1.1, 1.2, 2)
    df_case_1_1 = df[(df['ASSIETTE CUM'] < df[plafond_column]) & (df['Statut de salariés'] == 'Stagiaire')]
    df_case_1_2 = df[(df['ASSIETTE CUM'] < df[plafond_column]) & (df['Statut de salariés'] != 'Stagiaire')]
    df_case_2 = df[df['ASSIETTE CUM'] > df[plafond_column]]

    # Appliquer les prédictions sur chaque cas en passant les bonnes colonnes selon le modèle
    result_case_1_1 = predict_case(df_case_1_1, case_1_1_data, info['numeric_cols'], taux_column)
    result_case_1_2 = predict_case(df_case_1_2, case_1_2_data, info['numeric_cols'], taux_column)
    result_case_2 = predict_case(df_case_2, case_2_data, info['numeric_cols'], taux_column)

    # Fusionner les résultats des cas sans ignorer les index (conserver les index originaux)
    final_results = pd.concat([result_case_1_1, result_case_1_2, result_case_2])

    # Ajouter les résultats dans le rapport d'anomalies
    for index, row in final_results.iterrows():
        if row['Anomalie'] == 'Oui':
            original_index = row.name  # Utiliser l'index d'origine du DataFrame
            anomalies_report[original_index] = anomalies_report.get(original_index, {})
            anomalies_report[original_index][model_name] = f"Anomalie détectée pour {model_name} avec un taux de différence"
            model_anomalies[model_name] = model_anomalies.get(model_name, 0) + 1



def process_model(df, model_name, info, anomalies_report, model_anomalies):
    df_filtered = df

    if df_filtered.empty:
        st.write(f"Aucune donnée à traiter pour le modèle {model_name}.")
        return

    required_columns = info['numeric_cols'] + info['categorical_cols']
    missing_columns = [col for col in required_columns if col not in df_filtered.columns]

    if missing_columns:
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


def process_model_6082_6084(df, model_name, info, anomalies_report, model_anomalies):
    
    # Charger le modèle spécifique (Random Forest pour 6082 et 6084)
    rf_classifier_loaded = joblib.load(info['model'])
    
    # Charger les encodeurs spécifiques
    label_encoder_statut = joblib.load('label_encoder_statut.pkl')
    label_encoder_frontalier = joblib.load('label_encoder_frontalier.pkl')
    
    # Nettoyer les noms de colonnes pour éviter les espaces accidentels
    df.columns = df.columns.str.strip()
    
    # Sélectionner les mêmes colonnes que celles utilisées pour l'entraînement
    try:
        features_new = df[info['numeric_cols'] + info['categorical_cols']].copy()
    except KeyError as e:
        st.error(f"Erreur : Les colonnes suivantes sont manquantes : {e}")
        return
    
    # Remplacer les NaN dans les colonnes catégorielles par une valeur par défaut (ici 'Unknown' pour éviter l'erreur d'encodage)
    features_new[info['categorical_cols']] = features_new[info['categorical_cols']].fillna('Unknown')

    # Gestion des valeurs non vues : remplacer les valeurs inconnues par 'Unknown' pour une catégorie arbitraire
    def handle_unknown_labels(encoder, series):
        # Remplacer 'Unknown' ou autres valeurs non vues par une nouvelle catégorie 'Unknown'
        series = series.apply(lambda x: x if x in encoder.classes_ else 'Unknown')
        
        # Ajouter 'Unknown' comme nouvelle classe si elle n'est pas déjà présente
        if 'Unknown' not in encoder.classes_:
            encoder.classes_ = np.append(encoder.classes_, 'Unknown')
        
        return encoder.transform(series)

    # Encodage des variables catégorielles avec gestion des catégories inconnues
    try:
        features_new['Statut de salariés'] = handle_unknown_labels(label_encoder_statut, features_new['Statut de salariés'])
        features_new['Frontalier'] = handle_unknown_labels(label_encoder_frontalier, features_new['Frontalier'])
    except ValueError as e:
        st.error(f"Erreur avec l'encodage des catégories pour {model_name}: {str(e)}")
        return
    
    # Réorganiser les colonnes en fonction de celles du modèle
    try:
        # Utiliser l'ordre des colonnes utilisées lors de l'entraînement du modèle
        if hasattr(rf_classifier_loaded, 'feature_names_in_'):
            training_columns = rf_classifier_loaded.feature_names_in_
        else:
            training_columns = info['numeric_cols'] + info['categorical_cols']
        
        # Identifier les colonnes manquantes dans les données test
        missing_cols = set(training_columns) - set(features_new.columns)
        
        # Ajouter les colonnes manquantes avec des valeurs par défaut (ici, 0)
        for c in missing_cols:
            features_new[c] = 0
        
        # Réarranger les colonnes dans le même ordre que lors de l'entraînement
        features_new = features_new.reindex(columns=training_columns)
    
    except AttributeError as e:
        st.error(f"Erreur d'attribut avec le modèle {model_name}: {str(e)}")
        return

    # Faire des prédictions sur les nouvelles données
    try:
        predictions = rf_classifier_loaded.predict(features_new)
    except ValueError as e:
        st.error(f"Erreur avec le modèle {model_name} lors de la prédiction : {str(e)}")
        return

    # Ajouter les résultats de la prédiction au DataFrame
    df[f'{model_name}_Anomalie_Pred'] = predictions

    # Compter les anomalies détectées
    num_anomalies = np.sum(predictions)
    model_anomalies[model_name] = num_anomalies

    # Ajouter les anomalies détectées dans le rapport
    for index in df.index:
        if predictions[index] == 1:
            if index not in anomalies_report:
                anomalies_report[index] = {}  # Utilisation d'un dictionnaire pour chaque ligne
            anomalies_report[index][model_name] = f"Anomalie détectée pour {model_name}"



def process_model_6002(df, model_name, info, anomalies_report, model_anomalies):
    """
    Processus spécifique pour le modèle 6002 avec une gestion de l'encodage de 'Region'.
    """

    # Charger le modèle
    model = load_model(info)
    
    # Préparation des colonnes nécessaires
    df_filtered = df.copy()

    # Vérifier que les colonnes requises sont présentes
    required_columns = info['numeric_cols'] + info['categorical_cols']
    missing_columns = [col for col in required_columns if col not in df_filtered.columns]

    if missing_columns:
        st.error(f"Le modèle {model_name} manque des colonnes : {', '.join(missing_columns)}")
        return

    # Sélectionner les colonnes spécifiques au modèle
    df_inputs = df_filtered[required_columns].copy()

    # Encodage des variables catégorielles
    df_inputs_encoded = pd.get_dummies(df_inputs, columns=info['categorical_cols'], drop_first=True)

    # Vérifier que le nombre de colonnes encodées correspond aux colonnes utilisées pour l'entraînement
    model_features = model.feature_names_in_
    missing_cols = set(model_features) - set(df_inputs_encoded.columns)
    
    # Ajouter les colonnes manquantes avec des valeurs par défaut (0)
    for col in missing_cols:
        df_inputs_encoded[col] = 0
    
    # Réordonner les colonnes dans l'ordre attendu par le modèle
    df_inputs_encoded = df_inputs_encoded.reindex(columns=model_features, fill_value=0)

    try:
        # Faire les prédictions
        predictions = model.predict(df_inputs_encoded)
        df_filtered[f'{model_name}_Anomalie_Pred'] = predictions
        
        # Compter les anomalies détectées
        num_anomalies = np.sum(predictions)
        model_anomalies[model_name] = num_anomalies
        
        # Enregistrer les anomalies détectées dans le rapport
        for index in df_filtered.index:
            if predictions[df_filtered.index.get_loc(index)] == 1:
                anomalies_report.setdefault(index, set()).add(model_name)

    except ValueError as e:
        st.error(f"Erreur lors de la prédiction avec le modèle {model_name} : {e}")





def process_model_6082(df, model_name, info, anomalies_report, model_anomalies):
    """
    Processus spécifique pour le modèle 6082 avec gestion des encodages et colonnes manquantes.
    """
    # Charger le modèle
    model = load_model(info)
    
    # Préparation des colonnes nécessaires
    df_filtered = df.copy()

    # Vérifier que les colonnes requises sont présentes
    required_columns = info['numeric_cols'] + info['categorical_cols']
    missing_columns = [col for col in required_columns if col not in df_filtered.columns]

    if missing_columns:
        st.error(f"Le modèle {model_name} manque des colonnes : {', '.join(missing_columns)}")
        return

    # Sélectionner les colonnes spécifiques au modèle
    df_inputs = df_filtered[required_columns].copy()

    # Encodage des variables catégorielles
    df_inputs_encoded = pd.get_dummies(df_inputs, columns=info['categorical_cols'], drop_first=True)

    # Vérifier que le nombre de colonnes encodées correspond aux colonnes utilisées pour l'entraînement
    model_features = model.feature_names_in_
    missing_cols = set(model_features) - set(df_inputs_encoded.columns)
    
    # Ajouter les colonnes manquantes avec des valeurs par défaut (0)
    for col in missing_cols:
        df_inputs_encoded[col] = 0
    
    # Réordonner les colonnes dans l'ordre attendu par le modèle
    df_inputs_encoded = df_inputs_encoded.reindex(columns=model_features, fill_value=0)

    try:
        # Faire les prédictions
        predictions = model.predict(df_inputs_encoded)
        df_filtered[f'{model_name}_Anomalie_Pred'] = predictions
        
        # Compter les anomalies détectées
        num_anomalies = np.sum(predictions)
        model_anomalies[model_name] = num_anomalies
        
        # Enregistrer les anomalies détectées dans le rapport
        for index in df_filtered.index:
            if predictions[df_filtered.index.get_loc(index)] == 1:
                if index not in anomalies_report:
                    anomalies_report[index] = {}  # Utiliser un dictionnaire pour chaque ligne
                anomalies_report[index][model_name] = "Anomalie détectée pour le modèle 6082"

    except ValueError as e:
        st.error(f"Erreur lors de la prédiction avec le modèle {model_name} : {e}")

def detect_anomalies(df):
    anomalies_report = {}
    model_anomalies = {}

    for model_name, info in models_info.items():
        if model_name in ['7045', '7050']:
            process_model_with_average(df, model_name, info, anomalies_report, model_anomalies)
        elif model_name == '7001':
            process_7001(df, model_name, info, anomalies_report, model_anomalies)
        elif model_name in ['6082', '6084']:  # Appel spécifique pour 6084
            process_model_6082_6084(df, model_name, info, anomalies_report, model_anomalies)
        elif model_name == '6002':  # Appel spécifique pour 6002
            process_model_6002(df, model_name, info, anomalies_report, model_anomalies)
        elif model_name in ['7002', '7025']:
            process_model_generic(df, model_name, info, anomalies_report, model_anomalies)
        else:
            process_model(df, model_name, info, anomalies_report, model_anomalies)

        # Appel de la nouvelle fonction de vérification des montants après chaque process_model
        verify_montant_conditions(df, model_name, anomalies_report, model_anomalies)

    st.write("**Rapport d'anomalies détectées :**")

    report_content = io.StringIO()
    report_content.write("Rapport d'anomalies détectées :\n\n")

    # Créer un rapport d'anomalies pour chaque modèle
    for model_name, count in model_anomalies.items():
        report_content.write(f"Un nombre de {int(count)} anomalies a été détecté pour la cotisation {model_name}.\n")
    
    detailslist = []
    # Boucle sur les lignes avec anomalies détectées uniquement
    for line_index, anomaly_details in anomalies_report.items():
        matricule = df.loc[line_index, 'Matricule']
        # Créer la chaîne de détails des anomalies avec les modèles associés
        details = ', '.join([f"{model}: {desc}" for model, desc in anomaly_details.items()])
        # Ajouter la ligne uniquement si des anomalies sont présentes pour le matricule
        if details:  # Vérifie que des détails existent avant d'ajouter au rapport
            detailslist.append(details)
            report_content.write(f"Matricule {matricule} : anomalie dans les cotisations {details}\n")
        
    total_anomalies = len(detailslist)
    st.write(f"**Total des lignes avec des anomalies :** {total_anomalies}")
    report_content.write(f"Total des lignes avec des anomalies : {total_anomalies}\n")

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
