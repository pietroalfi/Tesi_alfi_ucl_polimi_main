import pandas as pd 
import os

def process_lob_data():
    # Base directory containing the CSV files
    base_dir = '/g100/home/userexternal/palfi000/ucl-thesis-main/LOB_data_preprocess_merge/INTC_2024-11-18_2024-11-29_1'
    # Directory where processed files will be saved
    filtered_data_dir = '/g100/home/userexternal/palfi000/ucl-thesis-main/LOB_data_preprocess_merge/Filtered_data'
    os.makedirs(filtered_data_dir, exist_ok=True)  # Ensure the directory exists

    # Loop over all dates from 01 to 31
    for day in range(18, 31):
        day_str = f"{day:02d}"  # Ensures two-digit format (e.g., 01, 02, ..., 31)
        base_name = f'INTC_2024-11-{day_str}_34140000_57660000'
        
        # Construct file paths
        message_file = os.path.join(base_dir, base_name + '_message_1.csv')
        orderbook_file = os.path.join(base_dir, base_name + '_orderbook_1.csv')
        
        # Check if files exist before processing
        if not os.path.exists(message_file) or not os.path.exists(orderbook_file):
            print(f"Skipping {base_name} as files do not exist.")
            continue
        
        print(f"Processing {base_name}...")

        # Extract date
        date_str = f"2024-11-{day_str}"
        base_date = pd.to_datetime(date_str)

        try:
            # Load the CSV files
            messages = pd.read_csv(message_file, header=None)
            orderbook = pd.read_csv(orderbook_file, header=None)
        except Exception as e:
            print(f"Error loading files for {base_name}: {e}")
            continue

        # Assign column names to messages file
        if messages.shape[1] == 7:
            messages.columns = ['timestamp', 'event_type', 'col3', 'col4', 'col5', 'col6', 'col7']
        elif messages.shape[1] == 6:
            messages.columns = ['timestamp', 'event_type', 'col3', 'col4', 'col5', 'col6']
        else:
            print(f"Unexpected number of columns in the message file ({messages.shape[1]}) for {base_name}")
            continue

        # Assign column names to orderbook file (assuming it has 4 columns)
        if orderbook.shape[1] == 4:
            orderbook.columns = ['bid_price', 'bid_volume', 'ask_price', 'ask_volume']
        else:
            print(f"Unexpected number of columns in the orderbook file ({orderbook.shape[1]}) for {base_name}")
            continue

        # Convert timestamps
        messages['full_timestamp'] = base_date + pd.to_timedelta(messages['timestamp'], unit='s')

        # Define time range (12:00 - 13:00)
        start_time = pd.to_datetime(date_str + ' 12:00:00')
        end_time   = pd.to_datetime(date_str + ' 13:00:00')

        # Filter data within the time range
        mask = (messages['full_timestamp'] >= start_time) & (messages['full_timestamp'] < end_time)
        messages_filtered = messages.loc[mask].reset_index(drop=True)
        orderbook_filtered = orderbook.loc[mask].reset_index(drop=True)

        # Compute mid-price and scale it
        mid_price = ((orderbook_filtered['bid_price'] + orderbook_filtered['ask_price']) / 2) * 0.01

        # Create a Series with timestamp index and mid_price as values
        result = pd.Series(mid_price.values, index=messages_filtered['full_timestamp'], name='mid_price')
        result.index.name = 'Date'

        # Save each file separately in `Filtered_data_NEW`
        output_file = os.path.join(filtered_data_dir, f'{base_name}_filtered_mid_price.pkl')
        result.to_pickle(output_file)
        print(f"Saved: {output_file}")

if __name__ == '__main__':
    process_lob_data()
