import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import RFE
import pickle
import matplotlib.pyplot as plt
import numpy as np



def create_model(data):
    X = data.drop('diagnosis', axis=1)
    Y = data['diagnosis']


    # scale the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # spilt the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    # train the model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # test the model
    y_pred = model.predict(X_test)
    print('Accurcay of our model:', accuracy_score(y_test, y_pred))
    print("Classification report: \n", classification_report(y_test, y_pred))


     # Plot feature importances
    feature_importance = np.abs(model.coef_[0])
    feature_names = data.drop('diagnosis', axis=1).columns
    indices = np.argsort(feature_importance)[::-1]

    plt.figure()
    plt.title("Feature Importances")
    plt.bar(range(X.shape[1]), feature_importance[indices], align="center")
    plt.xticks(range(X.shape[1]), feature_names[indices], rotation=90)
    plt.xlim([-1, X.shape[1]])
    plt.show()


     # Feature selection using RFE
    rfe = RFE(model, n_features_to_select=5)
    rfe.fit(X_train, y_train)
    selected_features = np.array(feature_names)[rfe.support_]

    plt.figure()
    plt.title("Top Features Selected by RFE")
    plt.bar(selected_features, model.coef_[0][rfe.support_], align="center")
    plt.xticks(rotation=90)
    plt.show()

    # Training-Validation Accuracy Curve
    train_sizes = np.linspace(0.1, 0.9, 9)
    train_scores = []
    val_scores = []

    for train_size in train_sizes:
        X_train_partial, _, y_train_partial, _ = train_test_split(X_train, y_train, train_size=float(train_size), random_state=42)
        model.fit(X_train_partial, y_train_partial)
        train_scores.append(model.score(X_train_partial, y_train_partial))
        val_scores.append(model.score(X_test, y_test))

    plt.figure()
    plt.plot(train_sizes, train_scores, label='Training Accuracy')
    plt.plot(train_sizes, val_scores, label='Validation Accuracy')
    plt.title('Training-Validation Accuracy Curve')
    plt.xlabel('Training Size')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    return model, scaler



def get_clean_data():
    data = pd.read_csv("data/data.csv")
    

    data = data.drop(['Unnamed: 32', 'id'], axis=1)

    data['diagnosis'] = data['diagnosis'].map({ 'M': 1, 'B': 0 })

    return data

def main():
    data = get_clean_data()
    
    model, scaler = create_model(data)

    with open('model/model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)



if __name__ == '__main__':
    main()
