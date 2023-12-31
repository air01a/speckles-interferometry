from os import listdir
from os.path import splitext, isfile, isdir
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
import joblib

def features_importance(model):
    # Obtenir les importances des caractéristiques
    importances = model.feature_importances_

    # Associer les importances avec les noms des colonnes
    feature_names = X.columns
    feature_importances = pd.DataFrame(importances, index=feature_names, columns=["importance"])

    # Trier les caractéristiques par importance
    feature_importances = feature_importances.sort_values(by="importance", ascending=False)

    # Afficher les résultats
    print(feature_importances)

maindir = "results/"


files = []
for file in listdir(maindir):
    file_path = maindir +  file
    if isfile(file_path) and splitext(file_path)[1]=='.json':
        files.append(file_path)


data=[]
for file in files:
    f = open(file)
    data.append(json.load(f))
df = pd.DataFrame(data)

histograms = df['histogram'].apply(pd.Series)

# Nommer les nouvelles colonnes
histograms.columns = [f'histogram_{i}' for i in range(histograms.shape[1])]

# Joindre ce nouveau DataFrame avec l'original
df = pd.concat([df.drop('histogram', axis=1), histograms], axis=1)
print(df.head())



print("*"*30)
print("N_filter")
X = df.drop(['radius', 'max_filter', 'min_filter', 'n_filter'], axis=1)
Y_n = df[['n_filter']]


X_train, X_test, Y_train, Y_test = train_test_split(X, Y_n, test_size=0.3, random_state=42)
model = RandomForestClassifier(random_state=42,n_estimators=200)
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
print(Y_pred)
print(Y_test)
print(f'MSE ', mean_squared_error(Y_test, Y_pred))

features_importance(model)
joblib.dump(model, 'model_nfilter.pkl')

print("*"*30)
print("max_filter")
X = df.drop(['radius', 'max_filter', ], axis=1)
Y = df[['max_filter']]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y['max_filter'], test_size=0.1, random_state=42)
model = RandomForestRegressor(random_state=42,n_estimators=200)
model.fit(X_train, Y_train)

# Évaluer le modèle
Y_pred = model.predict(X_test)
print(f'MSE ', mean_squared_error(Y_test, Y_pred))
for res in zip(Y_pred, Y_test):
    print(f"p/r : {res[0]}/{res[1]}")

features_importance(model)

print("*"*30)


print("*"*30)
print("max_filter2")
X = df.drop(['radius', 'max_filter', 'min_filter'], axis=1)
Y = df[['max_filter']]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y['max_filter'], test_size=0.1, random_state=42)
model = RandomForestRegressor(random_state=42,n_estimators=200)
model.fit(X_train, Y_train)

# Évaluer le modèle
Y_pred = model.predict(X_test)
print(f'MSE ', mean_squared_error(Y_test, Y_pred))
for res in zip(Y_pred, Y_test):
    print(f"p/r : {res[0]}/{res[1]}")

features_importance(model)
joblib.dump(model, 'model_max_filter.pkl')

print("*"*30)

print("min_filter")
X = df.drop(['radius', 'min_filter'], axis=1)
Y = df[['min_filter']]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y['min_filter'], test_size=0.1, random_state=42)
model = RandomForestRegressor(random_state=42)
model.fit(X_train, Y_train)

# Évaluer le modèle
Y_pred = model.predict(X_test)
print(f'MSE ', mean_squared_error(Y_test, Y_pred))
for res in zip(Y_pred, Y_test):
    print(f"p/r : {res[0]}/{res[1]}")

features_importance(model)
joblib.dump(model, 'model_min_filter.pkl')
print("*"*30)
print("radius")
X = df.drop(['radius'], axis=1)
Y = df[['radius']]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y['radius'], test_size=0.2, random_state=42)
model = RandomForestRegressor(random_state=42)
model.fit(X_train, Y_train)

# Évaluer le modèle
Y_pred = model.predict(X_test)
print(f'MSE ', mean_squared_error(Y_test, Y_pred))
print(Y_test,Y_pred)

features_importance(model)
joblib.dump(model, 'model_radius.pkl')
with pd.ExcelWriter("test.xlsx") as writer:
    df.to_excel(writer)