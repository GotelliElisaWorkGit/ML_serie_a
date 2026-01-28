import os
import pandas as pd
import numpy as np

# === Percorso alla cartella che contiene i file CSV già estratti ===
folder_path = "."  

# === Elenco dei file CSV nella cartella ===
csv_files = [
    os.path.join(folder_path, f)
    for f in os.listdir(folder_path)
    if f.endswith(".csv")
]

# === Merge sulla colonna 'Player' ===
merged_df = None
for file in csv_files:
    df = pd.read_csv(file)
    df.columns = [col.strip() for col in df.columns]  # rimuove spazi extra

    # Elimina la colonna 'Rank' se esiste
    if 'Rank' in df.columns:
        df.drop(columns=['Rank'], inplace=True)
        
    if 'Total Matches' in df.columns:
        df.drop(columns=['Total Matches'], inplace=True)

    # Esegui il merge
    if merged_df is None:
        merged_df = df
    else:
        merged_df = pd.merge(merged_df, df, on=['Player', 'Minutes', 'Matches', 'Country', 'Team'], how='outer')

# === Salva il dataset finale ===
output_file = "dataset/player_merged.csv"
merged_df.to_csv(output_file, index=False)
print(f"Merge completato. File salvato come: {output_file}")
