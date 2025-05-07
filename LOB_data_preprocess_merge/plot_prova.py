import pandas as pd

# Carica i dati
file_path = "/g100/home/userexternal/palfi000/ucl-thesis-main/LOB_data_preprocess_merge/Plot_period_not_filtered_spreads/merged_spread.pkl"
spread_series = pd.read_pickle(file_path)

# Conta quanti spread sono > 10
over_10 = (spread_series > 10).sum()
total = len(spread_series)
percentage = over_10 / total * 100

print(f"Spread > 10: {over_10} valori")
print(f"Percentuale: {percentage:.4f}%")
