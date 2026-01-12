
import pandas as pd
from pathlib import Path

def load_data(data_path=None):
    """
    Load CSV data from a file path"""
    if data_path is None:
        # Get the directory where this script is located
        script_dir = Path(__file__).parent.parent
        data_path = script_dir / "notebook" / "data" / "titanic.csv"
    else:
        data_path = Path(data_path)

    # Check if file exists
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    try:
        df = pd.read_csv(data_path)
        print(f"Data loaded successfully from: {data_path}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

if __name__ == "__main__":
    # Example usage: python data_loader.py path/to/titanic.csv
    import sys
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = None
    load_data(file_path)

