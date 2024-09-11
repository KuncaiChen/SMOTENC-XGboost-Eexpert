import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd

"""
Parameters:
    data_file_path (str): Path to the input data file ('SMOTENC_data.csv').
    test_size_value (float): Proportion of the dataset to be included in the test split (default is 0.2).
    shuffle_value (bool): Boolean indicating whether to shuffle the data before splitting (default is True).
    random_state_value (int): Seed for reproducibility when splitting the data (default is 123).
    learning_rate_value (float): Learning rate for the XGBoost model (default is 0.2).
    random_state_model (int): Random state seed for the XGBoost model (default is 10).
    APC_weight_value (int): Weight for Aerobic plate counts during expert weight adjustment (default is 3).
    Ecoli_weight_value (int): Weight for Escherichia coli during expert weight adjustment (default is 4).
    Salmonella_weight_value (int): Weight for Salmonella during expert weight adjustment (default is 3).
    Lmonocytogenes_weight_value (int): Weight for Listeria monocytogenes during expert weight adjustment (default is 4).
    Bcereus_weight_value (int): Weight for Bacillus cereus during expert weight adjustment (default is 2).
"""

data_file_path = 'SMOTENC_data.csv'
test_size_value = 0.2
shuffle_value = True
random_state_value = 123
learning_rate_value = 0.2
random_state_model = 10
APC_weight_value = 3
Ecoli_weight_value = 4
Salmonella_weight_value = 3
Lmonocytogenes_weight_value = 4
Bcereus_weight_value = 2

def load_and_preprocess_data(file_path):
    """
    Load the data from the specified CSV file and perform preprocessing.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        X (pandas.DataFrame): Feature columns.
        Y (pandas.Series): Target column.
    """
    data = pd.read_csv(file_path)
    X = data[['Aerobic plate counts', 'Escherichia coli', "Salmonella", "Listeria monocytogenes", "Bacillus cereus",
              'Economic level', 'Month', 'Classification', "Year"]]
    Y = data['State']
    return X, Y

def split_data(X, Y, test_size=test_size_value, shuffle=shuffle_value, random_state=random_state_value):
    """
    Split the data into training and testing sets.

    Args:
        X (pandas.DataFrame): Feature columns.
        Y (pandas.Series): Target column.
        test_size (float): Proportion of the dataset to include in the test split.
        shuffle (bool): Whether to shuffle the data before splitting.
        random_state (int): Seed for reproducibility.

    Returns:
        X_train (pandas.DataFrame): Training feature columns.
        X_test (pandas.DataFrame): Testing feature columns.
        Y_train (pandas.Series): Training target column.
        Y_test (pandas.Series): Testing target column.
    """
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, shuffle=shuffle, random_state=random_state)
    return X_train, X_test, Y_train, Y_test

def train_xgboost_model(X_train, Y_train, learning_rate=learning_rate_value, random_state=random_state_model):
    """
    Train the XGBoost model.

    Args:
        X_train (pandas.DataFrame): Training feature columns.
        Y_train (pandas.Series): Training target column.
        learning_rate (float): Learning rate for XGBoost.
        random_state (int): Seed for reproducibility.

    Returns:
        model (xgboost.XGBClassifier): Trained XGBoost model.
    """
    model = xgb.XGBClassifier(learning_rate=learning_rate, random_state=random_state)
    model.fit(X_train, Y_train.astype('int'))
    return model

def evaluate_model(model, X_test, Y_test, model_name='XGboost'):
    """
    Evaluate the trained model on the testing set.

    Args:
        model (xgboost.XGBClassifier): Trained XGBoost model.
        X_test (pandas.DataFrame): Testing feature columns.
        Y_test (pandas.Series): Testing target column.
        model_name (str): Name of the model for printing.
    """
    predictions = model.predict(X_test)
    y_pro = model.predict_proba(X_test)
    print(f'{model_name}  Accuracy:', accuracy_score(Y_test, predictions))
    print(f'{model_name}  Precison:', precision_score(Y_test, predictions))
    print(f'{model_name}  Recall  :', recall_score(Y_test, predictions))
    print(f'{model_name}  F1      :', f1_score(Y_test, predictions))
    print(f'{model_name} Auc: {roc_auc_score(Y_test, y_pro[:, 1])}')

def normalize_weights(model):
    """
    Normalize the feature importances assigned by the model.

    Args:
        model (xgboost.XGBClassifier): Trained XGBoost model.

    Returns:
        APC (float): Normalized weight for Aerobic plate counts.
        Ecoli (float): Normalized weight for Escherichia coli.
        Salmonella (float): Normalized weight for Salmonella.
        Lmonocytogenes (float): Normalized weight for Listeria monocytogenes.
        Bcereus (float): Normalized weight for Bacillus cereus.
    """
    total = []
    for i in range(5):
        total.append(model.feature_importances_[i])
    total_importances = sum(total)

    APC = model.feature_importances_[0] / total_importances
    Ecoli = model.feature_importances_[1] / total_importances
    Salmonella = model.feature_importances_[2] / total_importances
    Lmonocytogenes = model.feature_importances_[3] / total_importances
    Bcereus = model.feature_importances_[4] / total_importances
    return APC, Ecoli, Salmonella, Lmonocytogenes, Bcereus

def adjust_weights(APC, Ecoli, Salmonella, Lmonocytogenes, Bcereus, APC_weight=APC_weight_value, Ecoli_weight=Ecoli_weight_value,
                   Salmonella_weight=Salmonella_weight_value, Lmonocytogenes_weight=Lmonocytogenes_weight_value, Bcereus_weight=Bcereus_weight_value):
    """
    Adjust the normalized weights based on expert ratings.

    Args:
        APC (float): Normalized weight for Aerobic plate counts.
        Ecoli (float): Normalized weight for Escherichia coli.
        Salmonella (float): Normalized weight for Salmonella.
        Lmonocytogenes (float): Normalized weight for Listeria monocytogenes.
        Bcereus (float): Normalized weight for Bacillus cereus.
        APC_weight (int): Weight for Aerobic plate counts.
        Ecoli_weight (int): Weight for Escherichia coli.
        Salmonella_weight (int): Weight for Salmonella.
        Lmonocytogenes_weight (int): Weight for Listeria monocytogenes.
        Bcereus_weight (int): Weight for Bacillus cereus.

    Returns:
        Expert_APC (float): Adjusted weight for Aerobic plate counts.
        Expert_Ecoli (float): Adjusted weight for Escherichia coli.
        Expert_Salmonella (float): Adjusted weight for Salmonella.
        Expert_Lmonocytogenes (float): Adjusted weight for Listeria monocytogenes.
        Expert_Bcereus (float): Adjusted weight for Bacillus cereus.
    """
    Expert_APC = APC * APC_weight
    Expert_Ecoli = Ecoli * Ecoli_weight
    Expert_Salmonella = Salmonella * Salmonella_weight
    Expert_Lmonocytogenes = Lmonocytogenes * Lmonocytogenes_weight
    Expert_Bcereus = Bcereus * Bcereus_weight
    return Expert_APC, Expert_Ecoli, Expert_Salmonella, Expert_Lmonocytogenes, Expert_Bcereus

def normalize_adjusted_weights(Expert_APC, Expert_Ecoli, Expert_Salmonella, Expert_Lmonocytogenes, Expert_Bcereus):
    """
    Normalize the adjusted weights based on expert ratings.

    Args:
        Expert_APC (float): Adjusted weight for Aerobic plate counts.
        Expert_Ecoli (float): Adjusted weight for Escherichia coli.
        Expert_Salmonella (float): Adjusted weight for Salmonella.
        Expert_Lmonocytogenes (float): Adjusted weight for Listeria monocytogenes.
        Expert_Bcereus (float): Adjusted weight for Bacillus cereus.

    Returns:
        Final_APC (float): Final normalized adjusted weight for Aerobic plate counts.
        Final_Ecoli (float): Final normalized adjusted weight for Escherichia coli.
        Final_Salmonella (float): Final normalized adjusted weight for Salmonella.
        Final_Lmonocytogenes (float): Final normalized adjusted weight for Listeria monocytogenes.
        Final_Bcereus (float): Final normalized adjusted weight for Bacillus cereus.
    """
    totalexpert_importances = Expert_APC + Expert_Ecoli + Expert_Salmonella + Expert_Lmonocytogenes + Expert_Bcereus
    Final_APC = Expert_APC / totalexpert_importances
    Final_Ecoli = Expert_Ecoli / totalexpert_importances
    Final_Salmonella = Expert_Salmonella / totalexpert_importances
    Final_Lmonocytogenes = Expert_Lmonocytogenes / totalexpert_importances
    Final_Bcereus = Expert_Bcereus / totalexpert_importances
    return Final_APC, Final_Ecoli, Final_Salmonella, Final_Lmonocytogenes, Final_Bcereus


if __name__ == "__main__":
    # Load and preprocess the data
    X, Y = load_and_preprocess_data(data_file_path)

    # Split the data
    X_train, X_test, Y_train, Y_test = split_data(X, Y)

    # Train the model
    model = train_xgboost_model(X_train, Y_train)

    # Evaluate the model
    evaluate_model(model, X_test, Y_test)

    # Normalize the weights
    APC, Ecoli, Salmonella, Lmonocytogenes, Bcereus = normalize_weights(model)

    # Adjust the weights
    Expert_APC, Expert_Ecoli, Expert_Salmonella, Expert_Lmonocytogenes, Expert_Bcereus = adjust_weights(APC, Ecoli, Salmonella, Lmonocytogenes, Bcereus)

    # Normalize the adjusted weights
    Final_APC, Final_Ecoli, Final_Salmonella, Final_Lmonocytogenes, Final_Bcereus = normalize_adjusted_weights(Expert_APC, Expert_Ecoli, Expert_Salmonella, Expert_Lmonocytogenes, Expert_Bcereus)
