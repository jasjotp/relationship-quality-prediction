from pathlib import Path
import click 
from functions.download_data import download_data_from_csv

@click.command()
@click.argument("input_url")      
@click.argument("output_csv")  

def main(input_url, output_csv):
    """
    Downloads data from the URL and outputs it to a specified CSV path 
    """
    download_data_from_csv(input_url, output_csv, 'relationship_duration')

if __name__ == "__main__":
    main()
