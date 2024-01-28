import pandas as pd 
import matplotlib

def main():
    metadata = pd.read_excel("/data/isip/data/fccc_dpath/deidentified/v1.0.0/DOCS/fccc_metadata_v26.xlsx")

    print(metadata.head())
    print(metadata.info())
    print(metadata.describe())

main()
