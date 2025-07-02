"""
This script downloads and unzips data files for the TCGA <cancer> cohort from the UCSC Xena data hub.

The data files are:
1. Clinical data: TCGA-<cancer>.clinical.tsv.gz
2. Protein expression data: TCGA-<cancer>.protein.tsv.gz
3. Gene expression data: TCGA-<cancer>.star_counts.tsv.gz
4. Gene-level absolute expression data: TCGA-<cancer>.gene-level_absolute.tsv.gz

The script performs the following steps:
1. Creates a directory named 'tcga_datasets' to store the downloaded files.
2. Downloads each file from the specified URLs.
3. Unzips the downloaded .gz files.
4. Deletes the original .gz files after extraction.

The data can be accessed and explored using the UCSC Xena Browser at the following link:
<Insert_link>
"""


import os
import urllib.request
import gzip
import shutil


def download_and_unzip(url, output_dir):
    filename = url.split("/")[-1]
    gz_path = os.path.join(output_dir, filename)
    output_path = gz_path.replace(".gz", "")
    
    print(f"Downloading {filename}...")
    urllib.request.urlretrieve(url, gz_path)
    print(f"Downloaded {filename} to {gz_path}")
    
    print(f"Unzipping {filename}...")
    with gzip.open(gz_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print(f"Unzipped to {output_path}")
    
    os.remove(gz_path)
    print(f"Removed {gz_path}")

if __name__ == "__main__":

    rootdir = os.getcwd()
    os.chdir(rootdir) 

    output_dir = os.path.join(rootdir, "../datasets_TCGA/tcga_BLCA_ucsc_xena/raw_data")
    os.makedirs(output_dir, exist_ok=True)
    
    urls = [
        "https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-BLCA.clinical.tsv.gz", # Add methylation?
        "https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-BLCA.protein.tsv.gz",
        "https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-BLCA.star_counts.tsv.gz",
        "https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-BLCA.gene-level_absolute.tsv.gz"
    ]
    
    for url in urls:
        download_and_unzip(url, output_dir)
    
    print("All files downloaded and unzipped successfully.")
