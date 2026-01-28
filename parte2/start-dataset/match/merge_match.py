import pandas as pd
import os

# === Percorsi file ===
file_matches = "Serie A_matches_23_24_final.csv"
file_bets = "2023_con_utc.csv"
output_file = "../../dataset/match_merged.csv"

# === Carica i dati ===
df_matches = pd.read_csv(file_matches)
df_bets = pd.read_csv(file_bets)

# === Uniforma nomi squadre ===
df_matches['HomeTeam'] = df_matches['HomeTeam'].astype(str).str.strip().str.lower()
df_matches['AwayTeam'] = df_matches['AwayTeam'].astype(str).str.strip().str.lower()
df_bets['HomeTeam'] = df_bets['HomeTeam'].astype(str).str.strip().str.lower()
df_bets['AwayTeam'] = df_bets['AwayTeam'].astype(str).str.strip().str.lower()

# === Merge sulle colonne chiave comuni ===
merged = pd.merge(
    df_matches,
    df_bets,
    on=['HomeTeam', 'AwayTeam'],
    how='inner',
    suffixes=('_match', '_bet')
)

# === Rimozione colonne non più necessarie ===
colonne_da_rimuovere = ['Round Name', 'FTHG_bet', 'FTAG_bet', 'UTC Time_bet']
merged.drop(columns=[col for col in colonne_da_rimuovere if col in merged.columns], inplace=True)

# === Salvataggio del risultato ===
os.makedirs(os.path.dirname(output_file), exist_ok=True)
merged.to_csv(output_file, index=False)
print(f"Merge completato e colonne inutili rimosse. File salvato in: {output_file}")
