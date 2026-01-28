import pandas as pd
import joblib

# Caricamento dei modelli e colonne
log_model = joblib.load("modelli/log_model_v2.pkl")
rf_model = joblib.load("modelli/rf_model_v2.pkl")
feature_columns = joblib.load("modelli/feature_columns_v2.pkl")

# caricamento dei dataset utilizzato nel training 
df_match = pd.read_csv("dataset/match_merged.csv")
df_player = pd.read_csv("dataset/player_merged.csv")
df_team = pd.read_csv("dataset/squadra_merged.csv")

# applico gli stessi ragionamento fatti nel training
for df in [df_match, df_player, df_team]:
    for col in ['HomeTeam', 'AwayTeam', 'Team']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()

# Calcola le statistiche medie di squadre e giocatori 
df_team = pd.concat([df_team[['Team']], df_team.select_dtypes(include='number')], axis=1)
player_stats_by_team = df_player.groupby('Team').mean(numeric_only=True).reset_index()

# funzioni utili per interfaccia 
def capitalizza_nome(nome):
    return nome[0].upper() + nome[1:].lower() if nome else ""

# Costruzione del vettore delle feature 
def build_feature_vector(home, away):
    h_team = df_team[df_team['Team'] == home].mean(numeric_only=True)
    a_team = df_team[df_team['Team'] == away].mean(numeric_only=True)
    h_player = player_stats_by_team[player_stats_by_team['Team'] == home].mean(numeric_only=True)
    a_player = player_stats_by_team[player_stats_by_team['Team'] == away].mean(numeric_only=True)

    row_match = df_match[(df_match['HomeTeam'] == home) & (df_match['AwayTeam'] == away)].head(1)
    if row_match.empty or h_team.empty or a_team.empty or h_player.empty or a_player.empty:
        return None

    row_match = row_match.iloc[0]
    try:
        quote_features = row_match[['B365H', 'B365D', 'B365A', 'AvgH', 'AvgD', 'AvgA',
                                    'B365>2.5', 'B365<2.5', 'Avg>2.5', 'Avg<2.5',
                                    'AHh', 'B365AHH', 'B365AHA', 'MaxH', 'MaxD', 'MaxA',
                                    'Max>2.5', 'Max<2.5', 'AvgAHH', 'AvgAHA']]
    except KeyError:
        return None

    if quote_features.isnull().any():
        return None

    features = pd.concat([
        h_team.add_suffix("_home"),
        a_team.add_suffix("_away"),
        (h_team - a_team).add_suffix("_diff"),
        h_player.add_suffix("_p_home"),
        a_player.add_suffix("_p_away"),
        (h_player - a_player).add_suffix("_p_diff"),
        quote_features.rename(lambda x: f"QUOTE_{x}")
    ])

    df_feat = pd.DataFrame([features])
    df_feat = df_feat.reindex(columns=feature_columns)
    df_feat = df_feat.apply(pd.to_numeric, errors='coerce').fillna(0)

    return df_feat

def interpret_result(pred, home, away):
    if pred == 1:
        return "Pareggio"
    elif pred == 0:
        return f"Vittoria {capitalizza_nome(home)}"
    else:
        return f"Vittoria {capitalizza_nome(away)}"

# codice interfaccia 
print("             Benvenuto nel modello di calcolo di probabilità delle vittorie in una partita! =)")
print("                                                 Iniziamo!\n")

while True:
    s1 = input("Inserisci la squadra in casa: ").strip().lower()
    s2 = input("Inserisci la squadra in trasferta: ").strip().lower()

    # errore se inserita due volte la stessa squadra
    if not s1 or not s2 or s1 == s2:
        print("\nErrore! Nome squadra inserito incorretto o sconosciuto: e' stata inserita due volte la stessa squadra")
        if input("\nVuoi inserire un'altra partita? (s/n): ").lower() != 's':
            break
        continue

    # controllo che entrambe le squadre siano nel dataset e che gli encoder li conoscano
    squadre_presenti = df_team['Team'].unique()
    if s1 not in squadre_presenti or s2 not in squadre_presenti:
        print("\nErrore! Nome squadra inserito incorretto o sconosciuto: Una delle squadre non è presente nel dataset")
        if input("\nVuoi inserire un'altra partita? (s/n): ").lower() != 's':
            break
        continue

     # costruzione del vettore delle feature
    f_vector = build_feature_vector(s1, s2)
    if f_vector is None:
        print("\nErrore! Impossibile calcolare le feature per le squadre inserite: il dataset non presenta abbastanza informazioni per una o entrambe le squadre inserite")
        if input("\nVuoi inserire un'altra partita? (s/n): ").lower() != 's':
            break
        continue

    # faccio la predizione 
    pred_log = log_model.predict(f_vector)[0]
    pred_rf = rf_model.predict(f_vector)[0]

    res_log = interpret_result(pred_log, s1, s2)
    res_rf = interpret_result(pred_rf, s1, s2)

    # stampo i risultati per i due modelli utilizzati 
    print("\nRisultati predetti:")
    print(f" - Logistic Regression: {res_log}")
    print(f" - Random Forest:       {res_rf}")

    continua = input("\nVuoi inserire un'altra partita? (s/n): ")
    if continua.lower() != 's':
        break
