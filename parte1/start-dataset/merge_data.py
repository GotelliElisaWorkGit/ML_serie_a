import pandas as pd

#Carica i file,
df_2022 = pd.read_csv("2022.csv")
df_2023 = pd.read_csv("2023.csv")

#Converti le date,
df_2022['Date'] = pd.to_datetime(df_2022['Date'], dayfirst=True, errors='coerce')
df_2023['Date'] = pd.to_datetime(df_2023['Date'], dayfirst=True, errors='coerce')

#Aggiungi la stagione,
df_2022['Season'] = '2022/2023'
df_2023['Season'] = '2023/2024'

#Colonne utili per il modello,
columns_to_keep = [
    'Date', 'HomeTeam', 'AwayTeam',
    'B365H', 'B365D', 'B365A',
    'B365>2.5', 'B365<2.5',
    'AHh', 'B365AHH', 'B365AHA',
    'HTHG', 'HTAG', 'HTR',
    'HF', 'AF', 'HY', 'AY', 'HR', 'AR',
    'Referee', 'Attendance', 'Time', 'FTR', 'Season'
]

#Filtra solo le colonne presenti,
df_2022 = df_2022[[col for col in columns_to_keep if col in df_2022.columns]].copy()
df_2023 = df_2023[[col for col in columns_to_keep if col in df_2023.columns]].copy()

#Unione dei due anni,
df_unito = pd.concat([df_2022, df_2023], ignore_index=True)

#Salva il file unificato,
df_unito.to_csv("../dataset/dataset_parte1.csv", index=False)