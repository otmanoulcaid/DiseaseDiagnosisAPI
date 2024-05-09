import joblib
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier


def parse_input(user_input):
    df = pd.read_csv("MachineLearning/DataSet.csv")
    result = df.iloc[:, -1].unique()
    X_train = df.iloc[: , :-1]
    y_train = df.iloc[:, -1]
    features = X_train.columns
    user_feat=[]
    for i in features:
        if i in user_input:
            user_feat.append(1)
        else:
            user_feat.append(0)
    return (X_train.columns, result.tolist(), user_feat)

def trained_model_prediction(input_feat):
    symptoms, conditions, user_feat = parse_input(input_feat)
    user_feat_df = pd.DataFrame(columns=symptoms)
    user_feat_df.loc[0] = user_feat
    model = joblib.load("MachineLearning/knn.pkl")
    # model = joblib.load("DecisionTree.pkl")
    # model = joblib.load("LogesticRegression.pkl")
    predicted = model.predict(user_feat_df)
    return (conditions[predicted[0]])

def Description(ds):
    df = pd.read_csv("MachineLearning/conditionDescriptions.csv")
    return (df[ds])

def symptoms():
    df = pd.read_csv("MachineLearning/DataSet.csv")
    symptoms = df.iloc[: , :-1]
    return (symptoms.columns.tolist())

# print (symptoms())

#for model testing

# features = [
#             'pus_filled_pimples',
#             'blackheads', 'scurring',
#             'skin_peeling',
#             'silver_like_dusting',
#             'small_dents_in_nails',
#         ]
# condition = trained_model_prediction(features)
# print (condition)
# print (Description(condition)[0])