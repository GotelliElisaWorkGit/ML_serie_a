import pandas as pd
import joblib

# Caricamento dei modelli e colonne
log_model = joblib.load('modelli/log_model.pkl')
rf_model = joblib.load('modelli/rf_model.pkl')
feature_columns = joblib.load('modelli/feature_columns.pkl')
le_home = joblib.load('modelli/le_home.pkl')
le_away = joblib.load('modelli/le_away.pkl')

# caricamento del dataset utilizzato nel training 
df = pd.read_csv("dataset/dataset_parte1.csv")

# applico gli stessi ragionamento fatti nel training
df['target'] = df['FTR'].map({'H': 0, 'D': 1, 'A': 2})  # Risultato finale della partita (deriva dalla colonna FTR =Full Time Result)
df['HTR'] = df['HTR'].map({'H': 0, 'D': 1, 'A': 2}) # risultato finale del secondo tempo
df.dropna(subset=['target', 'B365H', 'B365D', 'B365A'], inplace=True) # rimozione righe dove mancano i dati di scommessa o di target 

# funzion utili per interfaccia 
def normalizza_nome(nome): # toglie in case sensitive mettendo tutto in minuscolo
    return nome.strip().lower()

def capitalizza_nome(nome): # mette la prima lettere in maiuscolo
    return nome.capitalize() if nome else ""

def squadra_vincente(pred, home, away): # prende il risultato in numero e lo trasfroma nella stringa corretta 
    if pred == 1:
        return "Pareggio"
    return f"Vittoria {capitalizza_nome(home) if pred == 0 else capitalizza_nome(away)}"

# estraggo tutte le colonne numeriche (escludendo ID e target), e calcolo la media per squadra (come fatto nel training)
stat_columns = df.select_dtypes(include='number').columns.difference(['HomeTeam_ID', 'AwayTeam_ID', 'target'])
df['HomeTeam_norm'] = df['HomeTeam'].str.lower()
media_per_team = df.groupby('HomeTeam_norm')[stat_columns].mean(numeric_only=True)

# Costruzione del vettore delle feature 
def costruisci_feature(home, away):
    # Se non ho statistiche per una delle due squadre ritorno None per segnalare un errore 
    if home not in media_per_team.index or away not in media_per_team.index:
        return None

    # ottengo la media delle feature delle due squadre e calcolo la differenza tra le due 
    feat_home = media_per_team.loc[home]
    feat_away = media_per_team.loc[away]
    diff = feat_home - feat_away

    row_feat = pd.concat([
        feat_home.add_prefix("home_"),
        feat_away.add_prefix("away_"),
        diff.add_prefix("diff_")
    ])

    # aggiunta quote al vettore di prima come feauture numeriche 
    row_feat['B365H'] = df[df['HomeTeam_norm'] == home]['B365H'].mean()
    row_feat['B365D'] = df[df['HomeTeam_norm'] == home]['B365D'].mean()
    row_feat['B365A'] = df[df['HomeTeam_norm'] == home]['B365A'].mean()

    # codifico i nomi della squadra con gli stessi encoder utilizzati nel training 
    try:
        row_feat['HomeTeam_ID'] = le_home.transform([capitalizza_nome(home)])[0]
        row_feat['AwayTeam_ID'] = le_away.transform([capitalizza_nome(away)])[0]
    except:
        return None

    # ritorno il vettore come DataFrame con una sola riga 
    return row_feat.to_frame().T

# codice interfaccia 
print("             Benvenuto nel modello di calcolo di probabilità delle vittorie in una partita! =)")
print("                                                 Iniziamo!\n")


# ciclo per poter fare più predizioni alla volta
while True:  
    squadra1 = input("Inserisci la squadra in casa: ").strip()
    squadra2 = input("Inserisci la squadra in trasferta: ").strip()

    # normalizzazione dei nomi per un corretto confronto in medie_per_team
    home = normalizza_nome(squadra1)
    away = normalizza_nome(squadra2)

    # errore se inserita due volte la stessa squadra
    if home == away:
        print("\nErrore! Nome squadra inserito incorretto o sconosciuto: e' stata inserita due volte la stessa squadra")
        if input("\nVuoi inserire un'altra partita? (s/n): ").lower() != 's':
            break
        continue

    squadra1_cap = capitalizza_nome(home)
    squadra2_cap = capitalizza_nome(away)

    # controllo che entrambe le squadre siano nel dataset e che gli encoder li conoscano
    if squadra1_cap not in le_home.classes_ or squadra2_cap not in le_away.classes_:
        print("\nErrore! Nome squadra inserito incorretto o sconosciuto: Una delle squadre non è presente nel dataset")
        if input("\nVuoi inserire un'altra partita? (s/n): ").lower() != 's':
            break
        continue

    # costruzione del vettore delle feature
    input_features = costruisci_feature(home, away)
    if input_features is None:
        print("\nErrore! Impossibile calcolare le feature per le squadre inserite: il dataset non presenta abbastanza informazioni per una o entrambe le squadre inserite")
        if input("\nVuoi inserire un'altra partita? (s/n): ").lower() != 's':
            break
        continue

    # riallineo le colonne uguali a quelle del training 
    input_features = input_features[feature_columns].fillna(0)

    # faccio la predizione 
    pred_log = log_model.predict(input_features)[0]
    pred_rf = rf_model.predict(input_features)[0]

    # stampo i risultati per i due modelli utilizzati 
    print("\nRisultati predetti:")
    print(f" - Logistic Regression: {squadra_vincente(pred_log, home, away)}")
    print(f" - Random Forest:       {squadra_vincente(pred_rf, home, away)}\n")

    if input("\nVuoi inserire un'altra partita? (s/n): ").lower() != 's':
        break
