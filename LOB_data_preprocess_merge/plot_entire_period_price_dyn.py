import pandas as pd
import matplotlib.pyplot as plt

def plot_mid_price_sequential_with_weekly_labels():
    # Percorsi dei file
    input_file = '/g100/home/userexternal/palfi000/ucl-thesis-main/LOB_data_preprocess_merge/Plot_period_not_filtered/merged_mid_price.pkl'
    output_file = '/g100/home/userexternal/palfi000/ucl-thesis-main/LOB_data_preprocess_merge/Plot_period_not_filtered/mid_price_plot_sequential_weekly.png'

    # Stile del grafico
    plt.style.use('seaborn-v0_8-whitegrid')

    # Caricamento della serie temporale
    series = pd.read_pickle(input_file)
    if not pd.api.types.is_datetime64_any_dtype(series.index):
        series.index = pd.to_datetime(series.index)
    series = series.sort_index().dropna()

    # Eventuale conversione in dollari
    if series.max() > 100:
        series = series / 100

    # Reset dell'indice per usare un indice sequenziale
    df = series.reset_index()
    df.columns = ['timestamp', 'mid_price']

    # Crea asse X sequenziale (numerico) e asse Y (valori)
    x_vals = range(len(df))
    y_vals = df['mid_price'].values

    # Identifica il primo punto di ogni settimana per i tick
    df['week'] = df['timestamp'].dt.to_period('W')
    weekly_ticks = df.groupby('week').head(1)  # un punto per settimana

    xticks = weekly_ticks.index.tolist()
    xlabels = weekly_ticks['timestamp'].dt.strftime('%d/%m/%Y').tolist()

    # Creazione del grafico
    plt.figure(figsize=(16, 6))
    plt.plot(x_vals, y_vals, color='steelblue', linewidth=1.5, label='Mid Price')

    # Etichette settimanali
    plt.xticks(xticks, xlabels, rotation=45, fontsize=14)
    plt.yticks(fontsize=15)
    plt.ylabel('Mid Price ($)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.3)

    # Rimuove i bordi
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    # Aggiunge legenda e salva
    #plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"✅ Grafico salvato con label settimanali: {output_file}")

if __name__ == '__main__':
    plot_mid_price_sequential_with_weekly_labels()
