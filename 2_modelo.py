"""
Rodando com train_test_split
"""
"""
Rodar modelos sem train_test_split
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from warnings import filterwarnings
filterwarnings('ignore')


# constantes
PATH_HISTORIC = 'data_split/diabetes_historico.csv'
PATH_FUTURE = 'data_split/diabetes_futuro.csv'
RANDOM_SEED = 42
TEST_SIZE = 0.3

# leitura dos dados
diabetes_historic = pd.read_csv(PATH_HISTORIC)

# separar entre features e labels
X = diabetes_historic.drop(columns='Outcome')
y = diabetes_historic.loc[:, 'Outcome']

# separacao entre dados de teste e treino
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)

# instanciar os modelos
reg_log = LogisticRegression(random_state=RANDOM_SEED)
dec_tree = DecisionTreeClassifier(random_state=RANDOM_SEED)
grad_boost = GradientBoostingClassifier(random_state=RANDOM_SEED)
rand_forest = RandomForestClassifier(random_state=RANDOM_SEED)

# lista de modelos
list_models = [('Logistic Regresion', reg_log),
                ('Decision Tree', dec_tree),
                ('Gradient Boosting', grad_boost),
                ('Random Forest', rand_forest)]

# treinando, prevendo e avaliando modelos
print('\n')
print('Estamos avaliando os modelos em dados diferentes do que eles treinaram.')
print('----------------------------------------------------------------------------------------------')
for name, model in list_models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f'Acurácia do modelo {name}: {accuracy_score(y_test, y_pred)*100:.2f}%')

# avaliando em dados que ele nunca
diabetes_future = pd.read_csv(PATH_FUTURE)

# separar entre features e labels
X_fut = diabetes_future.drop(columns='Outcome')
y_fut = diabetes_future.loc[:, 'Outcome']

print('\n')
print('Avaliando em dados que os modelos nunca viram. Vamos ver se eles são bons.')
print('----------------------------------------------------------------------------------------------')
for name, model in list_models:
    y_pred_fut = model.predict(X_fut)
    print(f'Acurácia do modelo {name}: {accuracy_score(y_fut, y_pred_fut)*100:.2f}%')

