import pandas as pd
from imblearn.over_sampling import SMOTENC
import csv

def load_data(path='Student_meal_data.xls'):
    """
    This function loads the data from the Excel file.

    Parameters
    ----------
    path : str
        The path to the Excel file.

    Returns
    -------
    data : pandas.DataFrame
        The loaded data.
    """
    data = pd.read_excel(path)
    return data

def extract_features(data):
    """
    This function extracts the features and target variable from the data.
    """
    X = data[['Economic level', 'Month', 'Classification_Code', "Salmonella", "Listeria monocytogenes", "Packing_Code", 'Aerobic plate counts_Normalization', 'Escherichia coli_Normalization', "Bacillus cereus_Normalization", "Year"]]
    Y = data['State']
    return X, Y

def write_data(X_resampled, Y_resampled):
    """
    Write the resampled data to a CSV file.

    Parameters
    ----------
    X_resampled : pandas.DataFrame
        The resampled data.
    Y_resampled : pandas.Series
        The resampled target variable.

    Returns
    -------
    None
    """
    # Open the file in write mode.
    with open('SMOTENC_data.csv', 'w+', newline='', encoding='utf-8') as f:
        # Create a csv writer object.
        writer = csv.writer(f)
        # Write the header line.
        writer.writerow(('Economic level', 'Month', 'Classification', "Salmonella", "Listeria monocytogenes", "Packing", 'Aerobic plate counts', 'Escherichia coli', "Bacillus cereus", "Year", 'State'))
        # Write the data lines.
        for i in range(len(X_resampled['Economic level'])):
            writer.writerow([X_resampled['Economic level'][i], X_resampled['Month'][i],
                             X_resampled['Classification_Code'][i], X_resampled['Salmonella'][i],
                             X_resampled['Listeria monocytogenes'][i], X_resampled['Packing_Code'][i],
                             X_resampled['Aerobic plate counts_Normalization'][i],
                             X_resampled['Escherichia coli_Normalization'][i],
                             X_resampled['Bacillus cereus_Normalization'][i],
                             X_resampled['Year'][i],
                             Y_resampled[i]])

def generate_resampled_data(X, Y):
    """
    This function generates resampled data using the SMOTENC algorithm.
    """
    smotenc = SMOTENC(random_state=5, categorical_features=[1, 2, 3, 4, 9], k_neighbors=2)
    X_resampled, Y_resampled = smotenc.fit_resample(X, Y)
    return X_resampled, Y_resampled

def correct_data(X_resampled, data):
    """
    This function corrects the generated data based on specific criteria.

    Parameters
    ----------
    X_resampled : pandas.DataFrame
        The resampled data.
    data : pandas.DataFrame
        The original data.

    Returns
    -------
    Yt : list
        The corrected target variable.
    """
    Yt = []
    # Calculate the qualified values based on the maximum values in the original data.
    APC_qualified = 10 ** 5 / max(data['Aerobic plate counts'])
    Ecoli_qualified = 10 ** 2 / max(data['Escherichia coli'])
    Bcereu_qualified = 10 ** 5 / max(data['Bacillus cereus'])
    # Iterate over the resampled data and correct the target variable based on the criteria.
    for i in range(len(X_resampled['Economic level'])):
        if X_resampled['Aerobic plate counts_Normalization'][i] < APC_qualified and X_resampled['Escherichia coli_Normalization'][i] < Ecoli_qualified and \
                X_resampled['Salmonella'][i] < 1 and X_resampled['Listeria monocytogenes'][i] < 1 and X_resampled['Bacillus cereus_Normalization'][i] < Bcereu_qualified:
            Yt.append(0)
        else:
            Yt.append(1)
    return Yt


if __name__ == "__main__":
    data = load_data()
    X, Y = extract_features(data)
    X_resampled, Y_resampled = generate_resampled_data(X, Y)
    Yt = correct_data(X_resampled, data)
    write_data(X_resampled, Yt)


