import pandas as pd
import matplotlib.pyplot as plt

# Path del file contenente gli spread
file_path = "/g100/home/userexternal/palfi000/ucl-thesis-main/LOB_data_preprocess_merge/Plot_period_not_filtered_spreads/merged_spread.pkl"

# Carica i dati dal file .pkl
spread_series = pd.read_pickle(file_path)

# Filtra solo i valori positivi maggiori di 1 (se necessario)
filtered_spread = spread_series

# Assicurati che i valori siano interi
filtered_spread = filtered_spread.astype(int)

# Conta la frequenza di ciascun valore (ogni categoria)
spread_counts = filtered_spread.value_counts().sort_index()

# Filtra le categorie con meno di 5 occorrenze
spread_counts_filtered = spread_counts[spread_counts > 100]

# Stampa le categorie di spread con meno di 5 occorrenze
print("Categories with less than 5 occurrences:")
print(spread_counts_filtered)

# Plot delle frequenze per ciascun valore di spread con meno di 5 occorrenze
plt.figure(figsize=(10, 6))
spread_counts_filtered.plot(kind='bar')

# Aggiungi etichette e titolo
plt.title('Frequency of Spread Values with Less Than 5 Occurrences')
plt.xlabel('Spread Value')
plt.ylabel('Frequency')

# Imposta l'orientamento delle etichette dell'asse x come orizzontale
plt.xticks(rotation=0)

# Salva il grafico come immagine PNG
output_image_path = '/g100/home/userexternal/palfi000/ucl-thesis-main/LOB_data_preprocess_merge/Plot_period_not_filtered_spreads/spread_frequency_histogram_filtered.png'
plt.savefig(output_image_path)

# Chiude il plot per liberare risorse
plt.close()

print(f"Frequency histogram saved as: {output_image_path}")
