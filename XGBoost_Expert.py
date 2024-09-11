import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score


def load_and_preprocess_data(path='SMOTENC_data.csv'):
    """
    Load the data from the CSV file and perform preprocessing.

    Parameters
    ----------
    path : str
        The path to the CSV file.

    Returns
    -------
    X : pandas.DataFrame
        Feature columns.
    Y : pandas.Series
        Target column.
    """
    # Load the data
    data = pd.read_csv(path)

    # Select the feature columns
    X = data[
        [
            'Aerobic plate counts',
            'Escherichia coli',
            'Salmonella',
            'Listeria monocytogenes',
            'Bacillus cereus',
            'Economic level',
            'Month',
            'Classification',
            'Year',
        ]
    ]

    # Select the target column
    Y = data['State']

    # Return the preprocessed data
    return X, Y



def split_data(X, Y):
    """
    Split the data into training and testing sets.

    Args:
        X (pandas.DataFrame): Feature columns.
        Y (pandas.Series): Target column.

    Returns:
        X_train (pandas.DataFrame): Training feature columns.
        X_test (pandas.DataFrame): Testing feature columns.
        Y_train (pandas.Series): Training target column.
        Y_test (pandas.Series): Testing target column.
    """
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=123)
    return X_train, X_test, Y_train, Y_test


def train_xgboost_model(X_train, Y_train):
    """
    Train the XGBoost model.

    Args:
        X_train (pandas.DataFrame): Training feature columns.
        Y_train (pandas.Series): Training target column.

    Returns:
        model (xgboost.XGBClassifier): Trained XGBoost model.
    """
    model = xgb.XGBClassifier(learning_rate = 0.2, random_state = 10)
    model.fit(X_train, Y_train.astype('int'))
    return model


def evaluate_model(model, X_test, Y_test):
    """
    Evaluate the trained model on the testing set.

    Args:
        model (xgboost.XGBClassifier): Trained XGBoost model.
        X_test (pandas.DataFrame): Testing feature columns.
        Y_test (pandas.Series): Testing target column.
    """
    model_name = 'XGboost'
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


def adjust_weights(APC, Ecoli, Salmonella, Lmonocytogenes, Bcereus):
    """
    Adjust the normalized weights based on expert ratings.

    Args:
        APC (float): Normalized weight for Aerobic plate counts.
        Ecoli (float): Normalized weight for Escherichia coli.
        Salmonella (float): Normalized weight for Salmonella.
        Lmonocytogenes (float): Normalized weight for Listeria monocytogenes.
        Bcereus (float): Normalized weight for Bacillus cereus.

    Returns:
        Expert_APC (float): Adjusted weight for Aerobic plate counts.
        Expert_Ecoli (float): Adjusted weight for Escherichia coli.
        Expert_Salmonella (float): Adjusted weight for Salmonella.
        Expert_Lmonocytogenes (float): Adjusted weight for Listeria monocytogenes.
        Expert_Bcereus (float): Adjusted weight for Bacillus cereus.
    """
    Expert_APC = APC * 3
    Expert_Ecoli = Ecoli * 4
    Expert_Salmonella = Salmonella * 3
    Expert_Lmonocytogenes = Lmonocytogenes * 4
    Expert_Bcereus = Bcereus * 2
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
        Final_Salmonella1 (float): Final normalized adjusted weight for Salmonella.
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
    X, Y = load_and_preprocess_data()

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

