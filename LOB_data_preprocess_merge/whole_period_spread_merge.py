import pandas as pd
import os
import glob

def merge_pickle_files(input_dir, output_file_pkl, output_file_parquet):
    """
    Merge all pickle files in a directory (che sono già Series) into a single Series.
    Assumes that the files sono già ordinati cronologicamente.
    Salva il risultato in formato Pickle e Parquet.
    """
    # Trova tutti i file pickle nella directory, mantenendo l'ordine alfabetico
    pickle_files = sorted(glob.glob(os.path.join(input_dir, "*.pkl")))

    if not pickle_files:
        print("No .pkl files found in the directory:", input_dir)
        return

    print(f"Found {len(pickle_files)} Pickle files. Starting the merge...")

    # Concatena le Series
    merged = pd.concat((pd.read_pickle(file) for file in pickle_files))

    # Se per qualche motivo merged non è una Series, ma un DataFrame a singola colonna,
    # lo convertiamo in Series
    if not isinstance(merged, pd.Series):
        if isinstance(merged, pd.DataFrame) and merged.shape[1] == 1:
            merged = merged.iloc[:, 0]
        else:
            print("Merged object has unexpected type:", type(merged))
            return

    # Assicuriamoci che l'indice abbia il nome "Date" e la Series il nome "spread"
    merged.index.name = "Date"
    merged.name = "spread"

    print(f"Merge completed. Total rows: {len(merged)}")

    # Salva la Series come file Pickle
    merged.to_pickle(output_file_pkl)
    print(f"Pickle file saved: {output_file_pkl}")

    # Per Parquet, convertiamo la Series in DataFrame
    merged.to_frame().to_parquet(output_file_parquet, engine='pyarrow')
    print(f"Parquet file saved: {output_file_parquet}")

if __name__ == "__main__":
    # Definisci la directory contenente i file pickle filtrati
    input_directory = "/g100/home/userexternal/palfi000/ucl-thesis-main/LOB_data_preprocess_merge/Plot_period_not_filtered_spreads"
    # Definisci i nomi dei file di output
    output_pickle_file = os.path.join(input_directory, "merged_spread.pkl")
    output_parquet_file = os.path.join(input_directory, "merged_spread.parquet")
    # Esegui la funzione di merge
    merge_pickle_files(input_directory, output_pickle_file, output_parquet_file)
