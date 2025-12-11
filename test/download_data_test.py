import sys 
import os 
import pandas as pd
import pathlib

# appends the parent directory to the system path so we can import files from parent directories
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from scripts.functions.download_data import download_data_from_csv

def test_download_data(tmp_path): # use a temp folder from pytest that is deleted automatically after the test 
    df = pd.DataFrame({
        'relationship_duration': [1.234, None, 5.43],
        'test_col_2': [10, 20, 30]
    })

    input_csv_path = tmp_path / "input.csv"
    df.to_csv(input_csv_path)

    output_csv_path = tmp_path / "download_data_output.csv"

    # return the number of rows after calling the download_data_from_csv and dropping rows that have NaN under the relatiomship duration column
    n_rows = download_data_from_csv(
        input_csv_path,
        output_csv_path,
        'relationship_duration'
    )

    df_output = pd.read_csv(output_csv_path)

    # confirm that the function worked on the toy df and returned the correct number of rows with no NaNs
    assert n_rows == 2 

    # confirm that the df has has no Nan values for its relationship duration column 
    assert df_output['relationship_duration'].isna().sum() == 0
