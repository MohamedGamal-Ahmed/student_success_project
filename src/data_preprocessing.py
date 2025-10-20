import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(filepath: str) -> pd.DataFrame:
    """
    Reads the dataset from a CSV file.
    
    Parameters
    ----------
    filepath : str
        Path to the CSV file.
    
    Returns
    -------
    df : pd.DataFrame
        Loaded dataset.
    """
    df = pd.read_csv(filepath)
    print(" Data loaded successfully!")
    print(f"Shape: {df.shape}")
    print("\nSample data:")
    print(df.head())
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the dataset by handling missing values and encoding categorical features.

    Parameters
    ----------
    df : pd.DataFrame
        Original dataset.

    Returns
    -------
    df_clean : pd.DataFrame
        Cleaned and encoded dataset.
    """
    # Drop duplicates if any
    df = df.drop_duplicates()

    # Replace NaN with appropriate values (mean for numeric, mode for categorical)
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].mean(), inplace=True)

    # Encode categorical columns
    label_enc = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = label_enc.fit_transform(df[col])

    print(" Data cleaned and encoded successfully!")
    return df


def save_clean_data(df: pd.DataFrame, output_path: str):
    """
    Saves the cleaned dataset to a CSV file.
    
    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset.
    output_path : str
        Path to save the cleaned CSV file.
    """
    df.to_csv(output_path, index=False)
    print(f" Clean dataset saved to: {output_path}")


if __name__ == "__main__":
    # Step 1: Load data
    df = load_data("data/students.csv")

    # Step 2: Clean data
    df_clean = clean_data(df)

    # Step 3: Save cleaned data
    save_clean_data(df_clean, "data/clean_students.csv")
