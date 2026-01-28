import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Caricamento dataset di partenza in un DataFrame 
df = pd.read_csv("dataset/dataset_parte1.csv")

df['target'] = df['FTR'].map({'H': 0, 'D': 1, 'A': 2}) # risultato finale della partita (deriva dalla colonna FTR =Full Time Result)
df['HTR'] = df['HTR'].map({'H': 0, 'D': 1, 'A': 2}) # risultato finale del secondo tempo 

# rimozione righe dove mancano i dati di scommessa o di target 
df.dropna(subset=['target', 'B365H', 'B365D', 'B365A'], inplace=True)

# label Encoding squadre
le_home = LabelEncoder() # encoder per le squadre in casa 
le_away = LabelEncoder() # encoder per le squadre in trasferta 

# codifica squadre in numeri
df['HomeTeam_ID'] = le_home.fit_transform(df['HomeTeam'])
df['AwayTeam_ID'] = le_away.fit_transform(df['AwayTeam'])

# calcolo medie statistiche per squadra
stat_columns = df.select_dtypes(include='number').columns.difference(
    ['HomeTeam_ID', 'AwayTeam_ID', 'target']
) # verranno utilizzate tutte le feature a parte gli ID squadra e il target 
media_per_team = df.groupby('HomeTeam')[stat_columns].mean(numeric_only=True) # calcolo media statistiche per squadra 

# funzione per costruire vettore di input per ogni partita
def costruisci_feature(row):
    # estrazioni nomi squadre partecipanti
    home = row['HomeTeam']
    away = row['AwayTeam']

    # se non ho statistiche per una delle due squadre restituisco una riga vuota e ignoro il match
    if home not in media_per_team.index or away not in media_per_team.index:
        return pd.Series()

    # ottengo la media delle feature delle due squadre e calcolo la differenza tra le due 
    feat_home = media_per_team.loc[home]
    feat_away = media_per_team.loc[away]
    diff = feat_home - feat_away

    row_features = pd.concat([
        feat_home.add_prefix("home_"),
        feat_away.add_prefix("away_"),
        diff.add_prefix("diff_")
    ])

    # Aggiunta quote al vettore di prima come feauture numeriche 
    row_features['B365H'] = row['B365H'] 
    row_features['B365D'] = row['B365D']
    row_features['B365A'] = row['B365A']

    # Aggiunta ID numerici
    row_features['HomeTeam_ID'] = row['HomeTeam_ID']
    row_features['AwayTeam_ID'] = row['AwayTeam_ID']

    return row_features

# Costruzione dataset finale
features_df = df.apply(costruisci_feature, axis=1)
features_df.dropna(inplace=True)  # Rimuove righe con valori mancanti

# estraggo la colonna target solo per linee presenti nel nuovo dataset 
target = df.loc[features_df.index, 'target']

# Salvataggio delle feature usate
feature_columns = features_df.columns.tolist()
os.makedirs("modelli", exist_ok=True)
joblib.dump(feature_columns, "modelli/feature_columns.pkl")

# Split del dataset: training e validation 
X_train, X_test, y_train, y_test = train_test_split(features_df, target, test_size=0.2, random_state=42)

# Addestramento Logistic Regression
log_model = LogisticRegression(max_iter=2000)
log_model.fit(X_train, y_train)
joblib.dump(log_model, "modelli/log_model.pkl")

# Addestramento Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
joblib.dump(rf_model, "modelli/rf_model.pkl")

# Salvataggio encoder dei due modelli 
joblib.dump(le_home, "modelli/le_home.pkl")
joblib.dump(le_away, "modelli/le_away.pkl")

print("Modelli addestrati e salvati con successo nella cartella 'modelli/'")
