import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split

np.random.seed(0)

classifiers = [
    KNeighborsClassifier(15),
    SVC(kernel='rbf'),
    DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators=300)
]

df = pd.read_csv('scrapper/data.csv').drop('id', axis=1)


def change_size(size):
    if 'M' in size:
        x = size[:-1]
        x = float(x) * 1024
        return x
    elif 'k' == size[-1:]:
        x = size[:-1]
        x = float(x)
        return x
    else:
        return None


def change_supported_version(version):
    if version == 'VARY':
        return None
    if '.' in version:
        return version.split('.')[0]
    return None


df['rozmiar_aplikacji'] = df['rozmiar_aplikacji'].map(change_size)
df['wspierany_android'] = df['wspierany_android'].map(change_supported_version)

df.dropna(inplace=True)

df = df.groupby('kategoria').filter(lambda x: len(x) > 300)
df = df.groupby('grupa_docelowa').filter(lambda x: len(x) > 50)
df = df.groupby('wspierany_android').filter(lambda x: len(x) > 50)

df = pd.get_dummies(df, columns=['kategoria', 'grupa_docelowa'])
scaler = MinMaxScaler()

y = df['srednia_ocena'].astype('int')
X = df.drop('srednia_ocena', axis=1)
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

for classifier in classifiers:
    classifier.fit(X_train, y_train)
    score = classifier.score(X_test, y_test)
    cm = confusion_matrix(y_test, classifier.predict(X_test))

    print(cm)
    print(score)
