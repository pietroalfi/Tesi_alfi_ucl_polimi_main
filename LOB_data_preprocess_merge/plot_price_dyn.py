import pandas as pd
import os
import matplotlib.pyplot as plt

# Directory dove si trovano i file pickle
filtered_data_dir = "/g100/home/userexternal/palfi000/ucl-thesis-main/LOB_data_preprocess_merge/Filtered_data"
# Directory dove salvare i plot
plot_dir = "/g100/home/userexternal/palfi000/ucl-thesis-main/LOB_data_preprocess_merge/Plot_dynamics"

# Assicurati che la cartella di destinazione esista
os.makedirs(plot_dir, exist_ok=True)

# Loop per tutti i file pickle nella cartella
for i in range(19, 32):
    day_str = f"{i:02d}"  # Formatta il giorno con due cifre (01, 02, ..., 18)
    date_str = f"2024-11-{day_str}"
    base_name = f"INTC_2024-11-{day_str}_34140000_57660000_filtered_mid_price.pkl"
    file_path = os.path.join(filtered_data_dir, base_name)

    # Controlla se il file esiste
    if not os.path.exists(file_path):
        print(f"Skipping {file_path}, file not found.")
        continue

    # Carica il file pickle
    data = pd.read_pickle(file_path)
    # Se è una Series, convertila in DataFrame
    if isinstance(data, pd.Series):
        data = data.to_frame(name="mid_price").reset_index()
        data.columns = ["timestamp", "mid_price"]

    # Controlla che ci siano dati
    if data.empty:
        print(f"No data available in {file_path}, skipping plot.")
        continue

    # Plotta il mid price nel tempo
    plt.figure(figsize=(10, 5))
    plt.plot(data["timestamp"], data["mid_price"], label="Mid Price", color='b')
    plt.xlabel("Time")
    plt.ylabel("Mid Price")
    plt.title(f"Stock Price Dynamics on {date_str}")
    plt.legend()
    plt.grid()

    # Salva il grafico come immagine
    output_path = os.path.join(plot_dir, f"price_dyn_{date_str}.png")
    plt.savefig(output_path)
    plt.close()  # Chiude il plot per risparmiare memoria

    print(f"Saved plot: {output_path}")
