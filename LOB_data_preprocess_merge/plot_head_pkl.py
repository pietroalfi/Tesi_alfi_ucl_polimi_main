import pandas as pd

# Specifica il percorso del file
file_path = "/g100/home/userexternal/palfi000/ucl-thesis-main/LOB_data_preprocess_merge/Plot_period_not_filtered_spreads/merged_spread.pkl"

# Carica il file .pkl
data = pd.read_pickle(file_path)

# Stampa la testa del file (le prime 5 righe per default)
print("Head of the file:")
print(data.head())

# Stampa il numero totale di righe nel file
print(f"\nTotal number of rows in the file: {data.shape[0]}")

# Verifica se ci sono valori di spread negativi
negative_spreads = data[data < 0]

if not negative_spreads.empty:
    print(f"\nThere are {len(negative_spreads)} negative spread values:")
    print(negative_spreads)
else:
    print("\nThere are no negative spread values.")

# Calcolo e stampa della media dello spread
print(f"\nMean spread value: {data.mean():.6f}")

# Stampa tutte le categorie di spread (valori unici)
spread_categories = data.value_counts()
print("\nSpread categories and their counts:")
print(spread_categories)
