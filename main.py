from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


classifiers = [
    KNeighborsClassifier(10),
    SVC(kernel='rbf', C=1.0),
    DecisionTreeClassifier(max_depth=30),
    RandomForestClassifier(max_depth=30, n_estimators=100)
]

df = pd.read_csv('scrapper/data.csv').drop('id', axis=1)
df = pd.get_dummies(df, columns=['kategoria', 'typ_aplikacji', 'grupa_docelowa', 'wspierany_android'])
replaces = {
    'M': '000',
    'k': '',
    'Varies with device': np.nan,
}
for to_replace, replacer in replaces.items():
    df['rozmiar_aplikacji'].replace(to_replace, replacer, inplace=True, regex=True)
df.fillna(df.mean(), inplace=True)
y = df['srednia_ocena'].round(1)
X = df.drop('srednia_ocena', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

for classifier in classifiers:
    classifier.fit(X_train, y_train)
    score = classifier.score(X_test, y_test)
    print(score)
