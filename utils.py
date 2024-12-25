import pandas as pd

def load_data(file_path):
    try:
        # Load Excel file
        df = pd.read_excel(file_path)
        # Split the data into features (x) and labels (y)
        x_train = df.iloc[:, 0].values  # Assuming first column is features
        y_train = df.iloc[:, 1].values  # Assuming second column is labels
        return x_train, y_train
    except Exception as e:
        print(f"Error reading the Excel file: {e}")
        return None