"""
Script de exploracao dos dados
"""
# importando bibliotecas
import pandas as pd

# constantes
PATH_HISTORIC = 'data_split/diabetes_historico.csv'

# leitura de dados
diabetes_historic = pd.read_csv(PATH_HISTORIC)

# olhando para o dataframe
print('Olhando para o DATAFRAME\n')
print(diabetes_historic.head(5))

# infos basicas do df
print('\nInfos básicas do df')
print(diabetes_historic.info())

# linhas duplicadas
quant_dups = diabetes_historic.duplicated().sum()
print(f'Quantidade de duplicados: {quant_dups}')

# estat descritiva
print(diabetes_historic.describe())

# contando classes
cont_valores = diabetes_historic.loc[:, 'Outcome'].value_counts(normalize=True)
print('\n\nContagem de labels')
print(cont_valores)