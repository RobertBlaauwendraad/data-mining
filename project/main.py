import numpy as np
from scipy.io import arff
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MultiLabelBinarizer, LabelBinarizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


def load_data(file_path):
    data = arff.loadarff(file_path)
    data_frame = pd.DataFrame(data[0])
    return data_frame


def preprocess_data(data_frame):
    # Drop columns with more than 50% missing values
    data_frame = data_frame.dropna(axis=1, thresh=len(data_frame) / 2)
    # Drop rows with missing values
    data_frame = data_frame.dropna(axis=0)

    # Drop match, decision, and decision_o columns
    X = data_frame.drop(columns=['match', 'decision', 'decision_o'], axis=1)
    # Target variable
    y = data_frame['match']
    # Convert byte strings to strings
    y = y.str.decode('utf-8')
    # Convert y to binary
    lb = LabelBinarizer()
    y_binary = lb.fit_transform(y)

    # Transform categorical columns to one-hot encoding and scale numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = [col for col in X.columns if col not in categorical_cols]
    col_preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(), categorical_cols)
        ]
    )
    X_processed = col_preprocessor.fit_transform(X)
    return X_processed, y_binary.ravel()


def train_random_forest_model(X_train, y_train):
    model = RandomForestClassifier(criterion='gini', max_depth= None,  n_estimators=100, random_state=12)
    model.fit(X_train, y_train)
    return model


def train_svm_model(X_train, y_train):
    svm_model = SVC(kernel='linear', C=10, random_state=12)
    ovr_classifier = OneVsRestClassifier(svm_model)
    ovr_classifier.fit(X_train, y_train)
    return ovr_classifier


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=1)
    return accuracy, report


def tune_hyperparameters(X_train, y_train):
    # Define hyperparameter grids for each model
    rf_param_grid = {
        'criterion': ['gini', 'entropy', 'log_loss'],
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20]
    }

    svm_param_grid = {
        'C': [0.1, 0.5, 1, 5, 10],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    }
    # Perform grid search with cross-validation for random forest
    rf_grid_search = GridSearchCV(RandomForestClassifier(random_state=12), param_grid=rf_param_grid, cv=5,
                                  scoring='recall')
    rf_grid_search.fit(X_train, y_train)

    # Perform grid search with cross-validation for SVM
    svm_grid_search = GridSearchCV(SVC(random_state=12), param_grid=svm_param_grid, cv=5, scoring='recall')
    svm_grid_search.fit(X_train, y_train)

    # Print best hyperparameters and corresponding recall scores
    print("Best Random Forest Hyperparameters (Optimizing for Recall):", rf_grid_search.best_params_)
    print("Best Random Forest Recall (Optimizing for Recall):", rf_grid_search.best_score_)
    print("Best SVM Hyperparameters (Optimizing for Recall):", svm_grid_search.best_params_)
    print("Best SVM Recall (Optimizing for Recall):", svm_grid_search.best_score_)


def main():
    data_frame = load_data('data/speeddating.arff')
    X, y = preprocess_data(data_frame)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

    # Perform hyperparameter tuning on X_train and y_train using X_val and y_val
    # tune_hyperparameters(X_train, y_train)

    random_forest_model = train_random_forest_model(X_train, y_train)
    svm_model = train_svm_model(X_train, y_train)

    # Evaluate the final model on the test set
    rf_accuracy, rf_report = evaluate_model(random_forest_model, X_test, y_test)
    svm_accuracy, svm_report = evaluate_model(svm_model, X_test, y_test)
    print('RF Accuracy: ', rf_accuracy)
    print('RF Classification report: \n', rf_report)
    print('SVM Accuracy: ', svm_accuracy)
    print('SVM Classification report: \n', svm_report)


main()
