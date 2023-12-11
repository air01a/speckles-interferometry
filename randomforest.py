from os import listdir
from os.path import splitext, isfile, isdir
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error


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
model = RandomForestClassifier(random_state=42)
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
print(Y_pred)
print(Y_test)
print(f'MSE ', mean_squared_error(Y_test, Y_pred))

print("*"*30)
print("max_filter")
X = df.drop(['radius', 'max_filter', ], axis=1)
Y = df[['max_filter']]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y['max_filter'], test_size=0.1, random_state=42)
model = RandomForestRegressor(random_state=42)
model.fit(X_train, Y_train)

# Évaluer le modèle
Y_pred = model.predict(X_test)
print(f'MSE ', mean_squared_error(Y_test, Y_pred))



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


print(df)
with pd.ExcelWriter("test.xlsx") as writer:
    df.to_excel(writer)