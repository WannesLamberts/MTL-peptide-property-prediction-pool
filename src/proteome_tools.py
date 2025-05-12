import requests  # Used for making HTTP requests to retrieve data from web pages or APIs
from bs4 import BeautifulSoup  # Used for parsing HTML and XML documents, especially for web scraping
import os  # Provides a way of interacting with the operating system, such as file and directory manipulation
from concurrent.futures import ThreadPoolExecutor


#link towards the dataset

import os
import requests
import zipfile
import io
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import regex as re
from matplotlib import pyplot as plt
from sklearn import linear_model  # Part of the scikit-learn library, used for implementing linear regression models and other machine learning algorithms
import numpy as np

def scrape_dataset(url, directory_path, max=None, workers=5):
    """
    Scrapes, downloads zip files from the website, and extracts only the evidence.txt
    file from each zip, renamed to match the zip filename.

    Parameters:
    -----------
    url : str
        The URL of the webpage where the dataset links are located.

    directory_path : str
        Path to the directory where the extracted evidence files will be saved.

    max : int, optional
        Maximum number of files to download.

    workers : int, optional
        Number of parallel threads to use for downloading.
    """

    # Send a GET request to retrieve the webpage content
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract all links ending with '.zip'
    links = [url + '/' + node.get('href') for node in soup.find_all('a')
             if node.get('href') and node.get('href').endswith('.zip')]

    if max:
        links = links[:max]

    os.makedirs(directory_path, exist_ok=True)

    def download_and_extract(link):
        zip_filename = link.split('/')[-1]
        base_name = os.path.splitext(zip_filename)[0]  # Remove .zip extension
        evidence_filename = os.path.join(directory_path, f"{base_name}_evidence.txt")

        try:
            # Download the zip file in memory
            with requests.get(link, stream=True) as r:
                r.raise_for_status()

                # Extract only evidence.txt from the zip
                with zipfile.ZipFile(io.BytesIO(r.content)) as zip_ref:
                    # Check if evidence.txt exists in the zip
                    evidence_files = [f for f in zip_ref.namelist() if f.endswith('evidence.txt')]

                    if evidence_files:
                        # Extract the first evidence.txt file found
                        with zip_ref.open(evidence_files[0]) as evidence_file:
                            with open(evidence_filename, 'wb') as f:
                                f.write(evidence_file.read())
                    else:
                        print(f"No evidence.txt found in {zip_filename}")

        except Exception as e:
            print(f"Failed to process {link}: {e}")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        executor.map(download_and_extract, links)

    return


def process_file(filename, directory, chronologer, calibrate=False, remove=False):
    if filename.endswith("evidence.txt"):
        pool = int(re.search("Pool_(\\d+)", filename, re.IGNORECASE).group(0).capitalize().replace("Pool_",""))
        # Read CSV fileget
        df = pd.read_csv(os.path.join(directory, filename), sep='\t')
        df['pool'] = pool
        df = preprocess_dataframe(df)
        df['file'] = filename

        if remove:
            os.remove(os.path.join(directory, filename))
        if calibrate:
            # calibrates the retention times
            df = calibrate_to_iRT(df, chronologer)
            # Check if calibration worked correctly
            if df is None:
                print(f"Not enough calibration peptides found in {filename}")
                return None
        print("processing done: "+filename)
        return df
    return None


def process_files_parallel(directory, chronologer='../raw_data/Chronologer.tsv',output="proteome_tools.parquet", calibrate=True, remove=False, max_workers=10):
    chronologer = pd.read_csv(chronologer, sep='\t')
    evidence_files = [f for f in os.listdir(directory) if f.endswith("evidence.txt")]
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks and collect futures
        futures = [executor.submit(process_file, filename, directory, chronologer, calibrate, remove)
                   for filename in evidence_files]

        # Get results from completed futures
        for future in futures:
            df = future.result()
            if df is not None:
                results.append(df)

    # Combine results
    if results:
        result = pd.concat(results, ignore_index=True)
        result.to_parquet(os.path.join(output), index=False)
        return result
    else:
        print("No valid dataframes to concatenate")
        return None




def get_calibration_peptides(df, calibration_df=None):
    """
    Retrieves a dictionary of calibration peptides and their corresponding iRT (indexed Retention Time) values.

    Author:
    -----------
    Ceder Dens

    Parameters:
    -----------
    df : pandas.DataFrame
        The main DataFrame containing peptide data.

    calibration_df : pandas.DataFrame, optional
        A DataFrame containing reference peptides and their known iRT values. If provided, the function
        will return calibration peptides that overlap between `df` and `calibration_df`. If not provided,
        a default set of iRT calibration peptides will be used.

    Returns:
    --------
    dict
        A dictionary where the keys are peptide sequences (str) and the values are the corresponding iRT values (float).
        If `calibration_df` is provided, the dictionary will contain peptides from the overlap of `df` and `calibration_df`.
        Otherwise, a predefined set of calibration peptides and iRT values is returned.
    """
    if calibration_df is None:
        return {
            "TFAHTESHISK": -15.01839514765834,
            "ISLGEHEGGGK": 0.0,
            "LSSGYDGTSYK": 12.06522819926421,
            "LYSYYSSTESK": 31.058963905737304,
            "GFLDYESTGAK": 63.66113155016407,
            "HDTVFGSYLYK": 72.10102416227504,
            "ASDLLSGYYIK": 90.51605846673961,
            "GFVIDDGLITK": 100.0,
            "GASDFLSFAVK": 112.37148254946804,
        }
    else:
        overlap = df.merge(calibration_df, how="inner", left_on="modified_sequence", right_on="PeptideModSeq")
        return {
            k: v for k, v in zip(overlap["PeptideModSeq"], overlap["Prosit_RT"])
        }

def calibrate_to_iRT(df,calibration_df=None,seq_col="modified_sequence",rt_col="Retention time",
    irt_col="iRT",plot=False,filename=None,take_median=False,):
    """
    Calibrates the retention times in a DataFrame to indexed Retention Time (iRT) values using a set of calibration peptides.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame containing peptide sequences and their respective retention times

    calibration_df : pandas.DataFrame, optional
        A DataFrame containing calibration peptides and their known iRT values. If not provided, a predefined
        set of calibration peptides will be used.

    seq_col : str, optional
        The column name in `df` that contains the peptide sequences. Default is "Modified sequence".

    rt_col : str, optional
        The column name in `df` that contains the retention time values. Default is "Retention time".

    irt_col : str, optional
        The column name where the predicted iRT values will be stored in `df`. Default is "iRT".

    plot : bool, optional
        If True, a scatter plot of the original Retention time values vs. iRT values will be generated along with the fitted regression line.

    filename : str, optional
        If provided, the function will print the number of calibration peptides found in the DataFrame. Useful for logging or debugging.

    take_median : bool, optional
        If True, the median retention time for each calibration peptide will be used. Otherwise, the first occurrence of the Retention time value will be used.

    Returns:
    --------
    pandas.DataFrame or None
        The input DataFrame with an additional column containing the calibrated iRT values.
        If fewer than two calibration peptides are found in the input data, the function returns `None`.

    Process:
    --------
    1. The function first retrieves a dictionary of calibration peptides and their corresponding iRT values.
    2. It loops through the calibration peptides and retrieves the corresponding Retention time values from the input DataFrame.
    3. If `take_median` is True, it uses the median Retention time value for each peptide; otherwise, it uses the first occurrence.
    4. The old Retention time values and iRT values are then used to fit a linear regression model.
    5. The model is used to predict iRT values for all peptides in the input DataFrame.
    6. If `plot` is True, a scatter plot of calibration points and the regression line is displayed.
    7. The function returns the input DataFrame with an additional column for iRT values, or `None` if calibration fails.
    """

    # Get calibration peptides and their corresponding iRT values
    calibration_peptides = get_calibration_peptides(df, calibration_df)
    old_rt = []
    cal_rt = []

    # Loop through each calibration peptide
    for pep, iRT in calibration_peptides.items():
        # Filter the DataFrame to get rows corresponding to the current peptide sequence
        pep_df = df[df[seq_col] == pep]
        if len(pep_df) > 0:
            # Use the median or first occurrence of the RT value based on the `take_median` flag
            if take_median:
                old = np.median(df[df[seq_col] == pep][rt_col])
            else:
                old = df[df[seq_col] == pep][rt_col].iloc[0]

            old_rt.append(old)
            cal_rt.append(iRT)
    # Log the number of calibration peptides found if `filename` is provided
    if filename is not None:
        print(
            f"{filename} had {len(old_rt)}/{len(calibration_peptides)} calibration peptides"
        )
    # If fewer than two calibration peptides are found, return None (unable to perform calibration)
    if len(old_rt) < 2:
        return None

    # Fit a linear regression model using the original RT values and the iRT values
    regr = linear_model.LinearRegression()
    regr.fit(np.array(old_rt).reshape(-1, 1), np.array(cal_rt).reshape(-1, 1))

    # Predict iRT values for all peptides in the input DataFrame
    df[irt_col] = regr.predict(df[rt_col].values.reshape(-1, 1))

    # Plot the calibration points and the fitted regression line if `plot=True`
    if plot:
        plt.scatter(old_rt, cal_rt, label="calibration points")
        plt.plot(
            range(int(min(old_rt) - 5), int(max(old_rt) + 5)),
            regr.predict(
                np.array(
                    range(int(min(old_rt) - 5), int(max(old_rt) + 5))
                ).reshape(-1, 1)
            ),
            label="fitted regressor",
        )
        plt.legend()
        plt.show()

    return df
def preprocess_dataframe(df,format_modified_sequence = True,min_score = 90, max_PEP = 0.01,reset_index = True):
    """
    Preprocess the input DataFrame by formatting the 'Modified sequence' column,
    and filtering based on the minimum score and maximum PEP values. At the end resets the index if chosen.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to be preprocessed.

    format_modified_sequence : bool, optional
        If True, removes the first and last character of the 'Modified sequence' column. Default is True.

    min_score : float, optional
        The minimum score threshold. Rows with a score less than this value will be removed. Default is 90.

    max_PEP : float, optional
        The maximum PEP threshold. Rows with PEP greater than this value will be removed. Default is 0.01.

    reset_index : bool, optional
        If True, resets the index of the dataframe, Default is True.

    Returns:
    --------
    pandas.DataFrame
        The preprocessed DataFrame
    """
    df = df.rename(columns={"Modified sequence": "modified_sequence"})

    # format the 'Modified sequence' by removing the first and last character
    if format_modified_sequence:
        df["modified_sequence"] = df["modified_sequence"].str[1:-1]

    # Filter rows based on the 'Score' and 'PEP' columns
    df = df[df["Score"]>=min_score]
    df = df[df["PEP"]<=max_PEP]

    df = df.reset_index(drop=True)
    df = df[["modified_sequence","Retention time","pool"]]


    return df
if __name__ == "__main__":
    LINK_DATASET = "https://ftp.pride.ebi.ac.uk/pride/data/archive/2017/02/PXD004732/"
    #scrape_dataset(LINK_DATASET,"../data/proteome_tools/")
    process_files_parallel("../data/proteome_tools")
