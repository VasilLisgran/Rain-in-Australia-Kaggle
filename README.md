# Rain-in-Australia-Kaggle
Гипотезы, EDA, исследование по датасету https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package/data
Датасет показался очень прикладным и интересным для исследований, связанных с физикой.

# Гипотезы

## 1. Удаление выбросов улучшит F1-score модели
### Подтверждена 
```python
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer 

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('LogisticRegression', LogisticRegression(class_weight='balanced', random_state=123))
])

score_raw = cross_val_score(pipe, X_train_raw, y_train_raw, cv=cv, scoring='f1_macro')
score_trans = cross_val_score(pipe, X_train_trans, y_train_trans, cv=cv, scoring='f1_macro')

if score_trans.mean() > score_raw.mean():
    print('Подтверилось', score_trans, ' ', score_raw)
else:
    print('Не подтверилось', score_trans, ' ', score_raw)
```
score_trans.mean : [0.72894202 0.73652052 0.73606985 0.73309111 0.73371673]  
score_raw.mean : [0.72866918 0.73539568 0.73240333 0.72985819 0.73193535] 

## 2. Recall покажет картину работы модели лучше, чем Accuracy, т.к. в датасете дисбаланс классов
### Опровержена
```python
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report

pipe.fit(X_train_trans, y_train_trans)
y_pred = pipe.predict(X_test_trans)

acc = accuracy_score(y_test_trans, y_pred)
rec_macro = recall_score(y_test_trans, y_pred, average='macro')

if acc < rec_macro:
    print("Подтверждение: ", acc, " ", rec_macro)
else:
    print("Опровержение: ", acc, " ", rec_macro)
```
accuracy : 0.7865255459052709  
f1_macro : 0.7779313654395769


## 3. При глубине дерева < 6 будет underfitting 
### Подтверждена
| Модель | Глубина | f1_rain | f1_macro |
|--------|---------|-----------|--------|
| Decision Tree | 3 | 0.5632 | 0.7007 |
| Decision Tree | 6 | 0.6055 | 0.7300 |
| Decision Tree | 9 | 0.6068 | 0.7273 |
| Logistic Regression | — | 0.6155 | 0.7339 |
По таблице при повышении **f1_rain** с 3 до 6 прирост **f1_rain** составляет 7.5%, прирост **f1_macro** составляет 4,2%

## 4. Дерево решений покажет более высокий F1-score, чем логистическая регрессия, потому что есть нелинейные признаки
### Опровержена
По таблице из гипотезы 3

## 5. Обучение модели только на признаках, измеренных в 15:00 (3pm), даст качество (F1-score) почти не уступающее модели на всех признаках сразу.
### Подтверждена
**F1-macro** (Все признаки): 0.7339  
**F1-macro** (Только 3pm):   0.7049


