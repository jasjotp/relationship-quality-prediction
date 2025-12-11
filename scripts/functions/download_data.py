from pathlib import Path
import pandas as pd

def download_data_from_csv(input_path, output_path, drop_column):
    '''
    Download data from a CSV/URL and output to a filepath. 
    
    Reads in a csv from a url, drops NaN rows from the column specified and 
    outputs the file to a filepath. Returns the length of rows in the dataframe 

    Parameters:
    ----------
    input_path : str
        The link that we want to read into a pandas DataFrame from.
    output_path : str
        The filepath that we want to output the raw data from our pandas DataFrame into.
    drop_column : str
        The column name that we want to drop rows from if NaN. 

    Returns:
    -------
    int
        The length of rows of the new dataframe (with the NaN rows of drop_column dropped)
    
    Examples:
    --------
    >>> import pandas as pd
    >>> download_data_from_url( # Replace with your actual input path, output path, and drop column
            'https://cran.r-project.org/incoming/UL/diversedata/data-clean/hcmst.csv', 
            'data/raw/hcmst_raw.csv', 
            'relationship_duration'
        )  
    >>> num_rows_df = download_data_from_url( # Replace with your actual input path, output path, and drop column
            'https://cran.r-project.org/incoming/UL/diversedata/data-clean/hcmst.csv', 
            'data/raw/hcmst_raw.csv', 
            'relationship_duration'
        )  
    >>> print(num_rows_df)
    >>> 1293
    '''
    df = pd.read_csv(input_path).dropna(subset = [drop_column])

    out_path = Path(output_path)

    out_path.parent.mkdir(parents = True, exist_ok = True)

    df.to_csv(out_path, index = False)

    # return the root path of the file
    print(f"Saved df ({len(df)} rows) to {out_path} after dropping NaNs from {drop_column}")

    return len(df)