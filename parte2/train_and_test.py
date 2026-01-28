import pandas as pd
import joblib
import os
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, classification_report

# Caricamento dataset di partenza di un DataFrame 
df_match = pd.read_csv("dataset/match_merged.csv")
df_player = pd.read_csv("dataset/player_merged.csv")
df_team = pd.read_csv("dataset/squadra_merged.csv")

# Uniforma i nomi delle squadre 
for df in [df_match, df_player, df_team]:
    for col in ['HomeTeam', 'AwayTeam', 'Team']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()

# Calcola le statistiche medie di squadre e giocatori 
df_team = pd.concat([df_team[['Team']], df_team.select_dtypes(include='number')], axis=1)
player_stats_by_team = df_player.groupby('Team').mean(numeric_only=True).reset_index()

# funzione per costruire vettore di input per ogni partita
def build_extended_feature_vector(row):
    home = row['HomeTeam']
    away = row['AwayTeam']

    # recupero statistiche squadre e giocatori 
    home_team_stats = df_team[df_team['Team'] == home].mean(numeric_only=True)
    away_team_stats = df_team[df_team['Team'] == away].mean(numeric_only=True)
    home_player_stats = player_stats_by_team[player_stats_by_team['Team'] == home].mean(numeric_only=True)
    away_player_stats = player_stats_by_team[player_stats_by_team['Team'] == away].mean(numeric_only=True)

    # selezione di tutte le partite giocate dalle squadre
    home_matches = df_match[(df_match['HomeTeam'] == home) | (df_match['AwayTeam'] == home)]
    away_matches = df_match[(df_match['HomeTeam'] == away) | (df_match['AwayTeam'] == away)]

    # colonne da utilizzare 
    match_cols = ['HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']
    quote_cols = ['B365H', 'B365D', 'B365A', 'AvgH', 'AvgD', 'AvgA', 'B365>2.5', 'B365<2.5',
                  'Avg>2.5', 'Avg<2.5', 'AHh', 'B365AHH', 'B365AHA',
                  'MaxH', 'MaxD', 'MaxA', 'Max>2.5', 'Max<2.5', 'AvgAHH', 'AvgAHA']

    # calcolo delle statistiche delle partite e le quote per ogni partita 
    try:
        h_match_stats = home_matches[match_cols].mean(numeric_only=True)
        a_match_stats = away_matches[match_cols].mean(numeric_only=True)
        h_quotes = home_matches[quote_cols].mean(numeric_only=True)
        a_quotes = away_matches[quote_cols].mean(numeric_only=True)
    except KeyError:
        return pd.Series()

    # se riga incompleta, scartala
    if any(x.empty for x in [home_team_stats, away_team_stats, home_player_stats, away_player_stats, h_quotes, a_quotes]):
        return pd.Series()

    # crea vettore feature 
    return pd.concat([
        home_team_stats.add_suffix("_home"),
        away_team_stats.add_suffix("_away"),
        (home_team_stats - away_team_stats).add_suffix("_diff"),
        home_player_stats.add_suffix("_p_home"),
        away_player_stats.add_suffix("_p_away"),
        (home_player_stats - away_player_stats).add_suffix("_p_diff"),
        h_quotes.rename(lambda x: f"QUOTE_H_{x}"),
        a_quotes.rename(lambda x: f"QUOTE_A_{x}"),
        h_match_stats.rename(lambda x: f"MATCH_H_{x}"),
        a_match_stats.rename(lambda x: f"MATCH_A_{x}")
    ])

# costruzione datast finale 
features_df = df_match.apply(build_extended_feature_vector, axis=1)
features_df.dropna(inplace=True)
df_match['target'] = df_match['FTR'].map({'H': 0, 'D': 1, 'A': 2})
target = df_match.loc[features_df.index, 'target']

# salvataggio colonne feature
feature_columns = features_df.columns.tolist()
os.makedirs("modelli", exist_ok=True)
joblib.dump(feature_columns, "modelli/feature_columns_v2.pkl")

# addestramento Logistic 
log_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
scores_log = cross_val_score(log_model, features_df, target, cv=5)
y_pred_log = cross_val_predict(log_model, features_df, target, cv=5)
log_model.fit(features_df, target)
joblib.dump(log_model, "modelli/log_model_v2.pkl")

# addestramento random
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
scores_rf = cross_val_score(rf_model, features_df, target, cv=5)
y_pred_rf = cross_val_predict(rf_model, features_df, target, cv=5)
rf_model.fit(features_df, target)
joblib.dump(rf_model, "modelli/rf_model_v2.pkl")

# stampa metrice
print("=== Logistic Regression ===")
print(f"Accuracy: {scores_log.mean():.4f}")                         
print("\nClassification Report:")
print(classification_report(target, y_pred_log))                    
print("\nConfusion Matrix:")
print(confusion_matrix(target, y_pred_log)) 

print("\n=== Random Forest ===")
print(f"Accuracy: {scores_rf.mean():.4f}")                         
print("\nClassification Report:")
print(classification_report(target, y_pred_rf)) 
print("\nConfusion Matrix:")
print(confusion_matrix(target, y_pred_rf))    

print("\nModelli addestrati, validati e salvati con successo.")
