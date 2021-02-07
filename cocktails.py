from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import json

df = pd.read_csv("cocktail_observations.csv")
y = df.pop('cocktails')
X = df

svc = SVC()
param_grid ={'C': [0.01, 0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
grid = GridSearchCV(svc, param_grid=param_grid, cv=2, iid='True')
grid.fit(X, y)


jsonfile = open('filename.json','r')
userdata = jsonfile.read()
X_test = json.loads(userdata)

if x_test['where'] == 'indoor':
    x_test['where'] = 0
else:
    x_test['where'] = 1
    
if x_test['season'] == 'spring':
    x_test['season'] = 0
elif x_test['season'] == 'summer':
    x_test['season'] = 1
elif x_test['season'] == 'autumn':
    x_test['season'] = 2
else:
    x_test['season'] = 3
    
if x_test['occasions'] == 'other':
    x_test['occasions'] = 0
elif x_test['occasions'] == 'christmas':
    x_test['occasions'] = 1
elif x_test['occasions'] == 'holiday':
    x_test['occasions'] = 2
else:
    x_test['occasions'] = 3
    
X_test = np.array([X_test['sad_happy'],X_test['stressed_relaxed'],
                   X_test['lonely_romantic'],X_test['couchpotato_party'],X_test['where'],
                   X_test['season'],X_test['occasions']])

rankings = grid.decision_function(X_test)

tail_index=[]
for i in range(rankings.shape[1]):
    tail_index.append(np.argmax(rankings))
    rankings[0][np.argmax(rankings)] = np.min(rankings)-1