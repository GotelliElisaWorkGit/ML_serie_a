import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Caricamento dei modelli e oggetti
log_model = joblib.load("modelli/log_model.pkl") # modello Logistic Regression
rf_model = joblib.load("modelli/rf_model.pkl")  # modello Random Forest
feature_columns = joblib.load("modelli/feature_columns.pkl") # elenco delle feature che usano i modelli
le_home = joblib.load("modelli/le_home.pkl") 
le_away = joblib.load("modelli/le_away.pkl")

# caricamento di un nuovo dataset per il testing
df = pd.read_csv("dataset/testing/2024.csv")

df['target'] = df['FTR'].map({'H': 0, 'D': 1, 'A': 2}) # risultato finale della partita (deriva dalla colonna FTR =Full Time Result)
df['HTR'] = df['HTR'].map({'H': 0, 'D': 1, 'A': 2}) # risultato finale del secondo tempo 

# rimozione righe dove mancano i dati di scommessa o di target 
df.dropna(subset=['target', 'B365H', 'B365D', 'B365A', 'HomeTeam', 'AwayTeam'], inplace=True)

# filtro su squadre viste nel training
df = df[df['HomeTeam'].isin(le_home.classes_) & df['AwayTeam'].isin(le_away.classes_)]

# codifico i nomi delle squadre usando gli stessi encoder del training 
df['HomeTeam_ID'] = le_home.transform(df['HomeTeam'])
df['AwayTeam_ID'] = le_away.transform(df['AwayTeam'])

# calcolo statistiche per ogni squadra (serve base da training)
# Carichiamo il dataset usato per calcolare le medie (usato nel training)
df_train = pd.read_csv("dataset/dataset_parte1.csv")

# applico le stesse trasformazioni utilizzate nel training 
df_train['HTR'] = df_train['HTR'].map({'H': 0, 'D': 1, 'A': 2})
df_train.dropna(subset=['B365H', 'B365D', 'B365A'], inplace=True)

# calcolo medie statistiche per squadra come nel training 
stat_columns = df_train.select_dtypes(include='number').columns.difference(['HomeTeam_ID', 'AwayTeam_ID', 'target'])
media_per_team = df_train.groupby('HomeTeam')[stat_columns].mean(numeric_only=True)

# funzione per costruire vettore di input per ogni partita (identica a quella del training)
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

# costruzione dataset finale
features_df = df.apply(costruisci_feature, axis=1)
features_df.dropna(inplace=True) # rimuove righe incorrette 

# estraggo il target per allinearlo con le predizioni
y_test = df.loc[features_df.index, 'target']

# ricostruisco il nuovo dataset in modo che abbia le stesse colonne di quello di training
X_test = features_df[feature_columns].fillna(0)

#utilizzo dei due modelli
y_pred_log = log_model.predict(X_test) # logistic regression
y_pred_rf = rf_model.predict(X_test) # rendom forest 

# stampa dei risultati ottenuti (con Accuracy e matrice di confusione)
print("=== Logistic Regression ===")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_log))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_log))

print("\n=== Random Forest ===")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))
