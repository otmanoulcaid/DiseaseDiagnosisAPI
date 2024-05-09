import joblib
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier


def parse_input(user_input):
    df = pd.read_csv("MachineLearning/DataSet.csv")
    result = df.iloc[:, -1].unique()
    X_train = df.iloc[: , :-1]
    print(X_train)
    y_train = df.iloc[:, -1]
    features = X_train.columns
    user_feat=[]
    for i in features:
        if i in user_input:
            user_feat.append(1)
        else:
            user_feat.append(0)
    return (X_train, result.tolist(), user_feat)

def trained_model_prediction(input_feat):
    X_train, conditions, user_feat = parse_input(input_feat)
    user_feat_df = pd.DataFrame(columns=X_train.columns)
    user_feat_df.loc[0] = user_feat
    # model = joblib.load("MachineLearning/knn.pkl")
    # model = joblib.load("MachineLearning/DecisionTree.pkl")
    model = joblib.load("MachineLearning/LogisticRegression.pkl")
    predicted = model.predict(user_feat_df)
    return (conditions[predicted[0]])

def Description(ds):
    df = pd.read_csv("MachineLearning/conditionDescriptions.csv")
    return (df[ds])

def symptoms():
    df = pd.read_csv("MachineLearning/DataSet.csv")
    symptoms = df.iloc[: , :-1]
    return (symptoms.columns.tolist())
