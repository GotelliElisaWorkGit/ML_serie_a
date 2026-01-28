import pandas as pd
import os

# === Percorsi ===
cartella_base = "."
output_file = "../../dataset/squadra_merged.csv"

# === Merge progressivo ===
merged_df = None

for nome_file in os.listdir(cartella_base):
    if nome_file.endswith(".csv"):
        path = os.path.join(cartella_base, nome_file)
        try:
            df = pd.read_csv(path)

            if 'Team' in df.columns:
                # Pulisce spazi nei nomi delle squadre
                df['Team'] = df['Team'].astype(str).str.strip()

                # Rimuove colonne inutili
                df = df.drop(columns=[col for col in df.columns if col in ['idx', 'rank', 'ranking']], errors='ignore')

                if merged_df is None:
                    merged_df = df
                else:
                    # Pulisce anche i nomi già nel merged_df
                    merged_df['Team'] = merged_df['Team'].astype(str).str.strip()

                    # Tieni solo le colonne che NON sono già nel merged_df (tranne 'Team' che serve sempre)
                    nuove_colonne = ['Team'] + [col for col in df.columns if col != 'Team' and col not in merged_df.columns]
                    df_filtrato = df[nuove_colonne]

                    # Esegui il merge mantenendo TUTTE le colonne nuove
                    merged_df = pd.merge(merged_df, df_filtrato, on='Team', how='outer')

        except Exception as e:
            print(f"Errore nel file {nome_file}: {e}")

# === Salva output ===
if merged_df is not None:
    print("\nColonne finali nel merged_df:")
    for col in merged_df.columns:
        print("-", col)
    merged_df.to_csv(output_file, index=False)
    print(f"\nDataset completo salvato in: {output_file}")
else:
    print("Nessun dataset valido trovato.")
