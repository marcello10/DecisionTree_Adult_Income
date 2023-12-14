# %% [markdown]
# # **1. Preparando o ambiente de trabalho**

# %%
# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as m
import seaborn as sns
from collections import Counter

# %% [markdown]
# # **2. Objetivos**

# %% [markdown]
# O objetivo deste trabalho é realizar uma análise exploratória de dados do conjunto de dados "Adult" do UCI Machine Learning Repository. E criar um modelo de classificação que classifique os respondentes da pesquisa de acordo com a renda a partir das variáveis de característica, obtendo valores corretos para a variável alvo. Utilizando o método de aprendizado de máquina denominado Árvore de decisão. Para saber quais características são importantes para determinar se a renda anaual é superior ou inferior a $50.00 Dólares por ano.
# 

# %% [markdown]
# # **3. O Dataset**

# %% [markdown]
# O conjunto de dados "Adult" contém informações sobre renda, educação, idade, sexo e raça, etc. O conjunto de dados contém *48.842* entradas com um total de 15 colunas representando diferentes atributos das pessoas, conforme a tabela abaixo. O conjunto de dados foi divido previamente em dois datasets, o dataset de de treino **adult.data** que contém *32.560* entradas, e o dataset de testes **adult.test** que contém *16.282* entradas. Os dados foram coletados em 1994 e a variável alvo,  **Renda Anual** (*income*) é dividida em duas classes: *"Alta Renda"* ($\gt50K$) e *"Baixa Renda"* ($\leq50K$).
# 
# 
# 
# 

# %% [markdown]
# | Coluna          | Variável                  | Definição                                                                                     | Tipo de Variável                             | Tipo de Dado |
# | :------------  | :------------            | :------                                                                                       | :------                                      |  ----:       |
# | age             |  Idade                    |   Idade do respondente                                                                        |   Discreta (de 17 a 90)                      |  int64       |
# | workclass       |  Classe de trabalho       |   Classificação do trabalho do respondente                                                    |   Categórica (9 categorias)                     |  object      |
# | fnlwgt          |  Peso final               |   O número de pessoas que acreditam no senso                          |   Discreta                                   |  int64       |
# | education       |  Escolaridade             |   O nível de escolaridade mais elevado obtido pelo respondente                                |   Ordinal (16 categorias)                    |  object      |
# | education-num   |  Número de escolaridade   |   O número de escolaridade associada ao 'education' do respondente                                             |   Discreta (de 1 a 16)                       |  int64       |
# | marital-status  |  Estado Civil             |   O estado civil do respondente                                                               |   Categórica (7 categorias)                     |  object      |
# | occupation      |  Ocupação                 |   Qual o tipo de trabalho do respondente                                                      |   Categórica (15 categorias)                    |  object      |
# | relationship    |  Relacionamento familiar  |   Tipo de relacionamento familiar do respondente                                              |   Categórica (6 categorias)                     |  object      |
# | race            |  Raça                     |   A raça do respondente                                                                       |   Categórica (5 categorias)                     |  object      |
# | sex             |  Sexo                     |   O sexo do do respondente                                                                    |   Categórica (2 categorias)                     |  object      |
# | capital-gain    |  Ganho de capital         |   Valor de ganhos de capital que o respondente obteve sobre sua poupança, investimentos e pensão   |   Contínua                                   |  int64       |
# | capital-loss    |  Perda de capital         |   Valor de perda de capital que o respondente obteve sobre sua poupança, investimentos e pensão    |   Contínua                                   |  int64       |
# | hours-per-week  |  Horas por semana         |   Quantidade de horas trabalhadas por semana pelo respondente                                 |   Discreta (de 1 a 99)                       |  int64       |
# | native-country  |  País de origem           |   Nacionalidade do respondente                                                                |   Categórica (42 países)                        |  object      |
# | income          |  Renda                    |   Classificação se o respondente é de baixa ou de alta renda                                  |   Booleano (≤ USD 50 mil, > USD 50 mil)      |  object      |

# %% [markdown]
# Os dados foram obtidos do repositório *UCI Machine Learning Repository* por meio do download do arquivo *.zip* que contém os arquivos *adult.data*, *adult.names*, *adult.test*, *Index*, *old.adult.names*. Então usamos o arquivo *adult.data*, que contém o dataset de treino, e transformando-o em um Pandas Dataframe.

# %%
# Carregando os dados dos arquivos baixados do site UCI Machine Learning.

colunas = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']

data = pd.read_csv('data/adult.data', sep = ', ', names = colunas, engine = 'python', skiprows = 1)

# %%
# Visualizando o dataset
data.head()

# %%
# Observando o formato do Dataframe
data.shape

# %% [markdown]
# Verificando a distribuição da variável alvo, Renda, no dataset.

# %%
data['income'].value_counts()

# %%
# Criando uma função para avaliar a simetria entre os valores de cada categoria de renda no dataset.

def calcula_porcent_renda(data):
    rotulo = data.values[:, -1]
    contador = Counter(rotulo)
    for key, value in contador.items():
        porcentagem = value / len(rotulo) * 100
        print(f"A Classe: {key}, tem o total de {value} indivíduos, o que representa {porcentagem:.2f}% dos dados coletados.")

# %%
# Calculando a porcentagem de cada categoria de renda

calcula_porcent_renda(data)

# %% [markdown]
# Nota-se que há um grande desbalanço de renda entre os participantes da pesquisa, com aqueles considerados de baixa renda ($\leq50K$) apresentando mais de três vezes o número de participantes considerados de alta renda ($\gt50K$).
# 
# 
# 

# %% [markdown]
# Observamos, então, as variáveis do dataset junto com seus tipos de dados e a presença de valores faltantes.

# %%
# Informações sobre as colunas e os dados
data.info()

# %%
print(f"dados únicos do workclass:{data['workclass'].unique()}\n")
print(f"dados únicos do occupation:{data['occupation'].unique()}\n")
print(f"dados únicos do native-country:{data['native-country'].unique()}\n")

# %% [markdown]
# Embora não tenha-se observado que existem valores faltantes, algumas categorias apresentam dados não respondidos. Por hora, faremos a substituição dos valores faltantes do caractere "**?**" para o objecto **nan**. E em seguida contaremos quantos valores desconhecidos há em cada coluna.

# %%
# Substituindo os valores faltantes do caractere '?' para o objeto nan,

from numpy import nan
data = data.replace('?',nan)

# %%
# Criando funlção para calcular os número e a percentagem de valores faltantes em cada coluna

def valoresFaltantes(data):
		valores_faltantes = data.isnull().sum()
		valores_faltantes = pd.DataFrame(valores_faltantes, columns= ['Valores Faltantes'])
		j=1
		sum_total=len(data)
		valores_faltantes['Porcentagem (%)'] = round(( (valores_faltantes['Valores Faltantes'] / sum_total) * 100), 1)
		return valores_faltantes.sort_values('Porcentagem (%)',ascending=False)


# %%
# Calculando o número, a percentagem dos valores faltantes.

valoresFaltantes(data)

# %% [markdown]
# Há valores faltantes apenas nas variáveis ***workclass***, ***occupation*** e ***native-country***, onde é possível notar que o número de valores faltantes nas variáveis ***workclass*** e ***occupation*** são praticamente iguais. Logo, quase sempre que houver um valor faltante em ***workclass***, haverá um valor faltante em ***occupation***.

# %% [markdown]
# # **4 Explorando as variáveis**

# %% [markdown]
# Separando as variáveis numéricas e categóricas em conjuntos de dados diferentes para tratá-las separadamente.

# %%
data['native-country'].value_counts()

# %% [markdown]
# <p>Observe que a maioria dos respondentes são Estadunidenses. Portanto, vamos considerar os demais países como 'outros'.</p>

# %%
# Criando conjuntos de dados diferentes para as variáveis numéricas e categóricas.

num_data= data[['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']]
cat_data= data.select_dtypes(include=['object'])
cat_data['native-country'] = cat_data['native-country'].apply(lambda country: country if country=='United-States' else 'Others')
cat_data['native-country'].unique()


# %% [markdown]
# ## **4.1 Explorando as variáveis numéricas**

# %% [markdown]
# Usamos o método **.describe** do Pandas para fazemos um sumário com as estatísticas descritivas básicas de todas as vairáveis numéricas, como os valores mínimos, médios, máximos, desvio padrão, etc.

# %%
# Sumário das variáveis numéricas

pd.set_option('display.float_format', lambda x: '%.3f' % x)
data.describe()

# %% [markdown]
# <p>Os dados de 'hours-per-week' possuem valores mínimo de <b>1h</b> e máximo de <b>99h</b>. Ou seja, provavelmente foram informados incorretamente ou erros na entrada dos dados.
# </p>

# %% [markdown]
# Plotamos os histogramas com as distribuições de todas as variáveis numéricas

# %%
num_data.hist(bins=20, figsize=(20, 15), histtype='step')
plt.show()

# %% [markdown]
# ##### **Observações:**

# %% [markdown]
# 
# 
# *   O pico da variável **idade** é idade abaixo de 50 anos.
# *   O máximo da variável **fnlwgts** está abaixo de 400k.
# *   Muito poucas pessoas investem em capital e há alguns casos discrepantes, como:
#     *   Pessoas que ganham mais de US$ 90.000,00 por meio de **ganhos de capital**.
#     *   No entanto, para pessoas que sofreram  **perda de capital** a perda média é cerca de US$ 2.000,00.
#     *   Obseve que a maioria das pessoas não tiveram perda de capital.
# *   A maioria das pessoas trabalha cerca de **40 horas semanais**.
#     *   Também há outliers como alguns casos de pessoas trabalhando **100 horas** e **1 hora** por semana.
# 
# 
# 
# 

# %% [markdown]
# #### **Correlações entre as variáveis numéricas**

# %% [markdown]
# Iremos, então, analisar a correlação entre as variáveis numéricas.

# %%
# Criando Função para gerar o heatmap com as correlações das variáveis numéricas.

def heatMap(data):
  sns.heatmap(data.corr(), annot=True, fmt='.3f')
  #print(data.corr())

# %%
heatMap(num_data)

# %% [markdown]
# ##### **Observações:**

# %% [markdown]
# *   A variável **fnlwgt** não possui correlação significativa com nenhuma das outras variáveis, portanto, é uma variável descartável para a análise.
# *   As variáveis **número de escolaridade**, **horas trabalhadas por semana** e **ganho capital** têm algumas correlações que valem a pena serem exploradas e podem ser melhoradas.

# %% [markdown]
# #### **Relações entre as variáveis numéricas e a Renda**

# %%
pd.pivot_table(data, index= ['income'], values=num_data.columns)

# %% [markdown]
# ##### **Observações:**

# %% [markdown]
# *   Pessoas com maior **renda** apresentam valores elevados em todos os atributos.

# %% [markdown]
# ## **4.2 Explorando as variáveis categóricas**

# %% [markdown]
# Começaremos plotando os gráficos de barra de todas as variáveis categóricas.

# %%
for i in cat_data.columns:
    plt.figure(figsize=(25, 5))
    sns.countplot(data= cat_data, x= i)

# %% [markdown]
# ##### **Observações:**

# %% [markdown]
# 
# 
# *   A maioria das pessoas trabalham no setor privado, com outras distribuídas quase uniformemente entre vários empregos governamentais e como autônomos.
# *   A maioria das pessoas possui diploma de Ensino Superior, do Ensino Médio ou cursou alguma faculdade. A distribuição é muito semelhante à dos anos de escolaridade (education-number), fato que pode ser melhor explorado.
# *   A maioria das pessoas casaram, sendo poucas as que divorciaram ou tornaram-se viúvas. Enquanto as demais não se casaram.
# *   As ocupações estão, em sua maioria, distribuídas de maneira uniforme e é difícil ver uma tendência quando há tantas categorias.
# *   A maioria das pessoas são maridos ou não têm família.
# *   A maioria das pessoas é branca, sendo os negros o único grupo etnico com número considerável na amostra.
# *   Há mais homens do que mulheres na amostra, o que também pode ser inferido do facto de a maioria das pessoas serem maridos.
# *   A grande maioria dos entrevistados é dos EUA.
# *   Como comentado anteriormente, há muito mais pessoas com Baixa Renda (inferior a 50 mil), do que pessoas de Alta Renda (superior à 50 mil), o que mostra que os dados estão muito desequilibrados.

# %% [markdown]
# # **5. Engenharia de atributos**

# %% [markdown]
# A maioria dos atributos categóricos são muito confusos, com número excessivo de categorias. Então, podemos fazer uma engenharia de atributos para agrupar categorias semelhantes para aprimorar os atributos.
# 
# Começamos observando o número de categorias de cada variável categórica.

# %%
# Computando valores únicos de cada variável categórica

cat_rotulos = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
wkclass = data.workclass.unique()
edu = data.education.unique()
estciv = data['marital-status'].unique()
ocup = data.occupation.unique()
relat = data.relationship.unique()
race = data.race.unique()
sex = data.sex.unique()
country = data['native-country'].unique()

# Gerando um array com os valores únicos de cada variável.

data_col=[wkclass, edu, estciv, ocup, relat, race, sex, country]

# Uniformizando o tamanho dos arrays

cat_col = []
for i in data_col:
    cat_col.append(np.pad(i, (0, (len(country)-len(i))), 'empty'))
cat_col.append(country)

# Gerando o dataframe com os valores das variáveis categóricas.

categorias = pd.DataFrame(dict(zip(cat_rotulos, cat_col)))
categorias.head()

# %%


# %% [markdown]
# #### **Classe de trabalho** (*workclass*)

# %% [markdown]
# Iremos iniciar reduzindo o número de categorias da variável Classe de trabalho (workclass), agrupando diversas categorias que apresentam afinidade entre si, criando apenas 4 grupos representando setores distintos da economia, são eles:
# *   Privado
# *   Governo
# *   Autônomo
# *   Outros

# %%
# Agrupando as classes de trabalho por afinidade.

data['workclass'].replace(['State-gov', 'Federal-gov', 'Local-gov'], 'Government', inplace=True)
data['workclass'].replace(['Self-emp-not-inc', 'Self-emp-inc'], 'Self', inplace=True)
data['workclass'].replace(['Without-pay', 'Never-worked'], 'Others', inplace=True)

plt.figure(figsize=(16, 4))
sns.countplot(data=data, x='workclass')
plt.figure(figsize=(16, 4))
sns.countplot(data=data, x='workclass', hue='income')

# %% [markdown]
# #### **Educação e Escolaridade** (*education* e *education-number*)

# %% [markdown]
# Verificando a interseção entre as variáveis Educação e Escolaridade

# %%
# Verificando a interseção entre educação e escolaridade.

data.groupby('education').nunique()['education-num']

# %% [markdown]
# Isso implica que *Educação* (education) e *Escolaridade* (education-num) representam exatamente a mesma informação. E já está diretamente codificada na base de dados, então podemos usar apenas um dos dois e descartar o outro.

# %%
# Agrupando os níveis de escolaridd por afinidade.

data['education'].replace(['11th', '9th', '7th-8th', '5th-6th', '10th', '1st-4th', 'Preschool', '12th'], 'School-Dropout', inplace=True)
data['education'].replace(['Some-college', 'Assoc-acdm', 'Assoc-voc'], 'College', inplace=True)
data['education'].replace('Prof-school', 'Masters', inplace=True)

# %%
# Plotando os gráficos

plt.figure(figsize=(16, 4))
sns.countplot(data=data, x='education')

plt.figure(figsize=(16,4))
sns.countplot(data=data, x='education', hue='income')

# %% [markdown]
# ##### **Observações:**

# %% [markdown]
# *   Mestres e Doutores são muito bem remunerados em comparação com demais níveis de ensino.
# *   A maioria das pessoas de baixa renda abandonou o ensino básico ou concluiu apenas o ensino médio
# 

# %% [markdown]
# #### **Estado Civil** (*marital-status*)

# %% [markdown]
# Iremos reduzir o número de categorias da variável Estado Civil (marital-status), agrupando diversas categorias que apresentam afinidade entre si, criando apenas 4 grupos representando setores distintos da economia, são eles:
# *   Solteiro(a)
# *   Casado(a)
# *   Divorciado(a)
# *   Viúvo(a)  

# %%
# Agrupando os estados civis por afinidade.

data['marital-status'].replace(['Married-AF-spouse', 'Married-civ-spouse', 'Married-spouse-absent'], 'Married', inplace=True)
data['marital-status'].replace('Divorced', 'Separated',inplace=True)

plt.figure(figsize=(16, 4))
sns.countplot(data=data, x='marital-status')

plt.figure(figsize=(16,4))
sns.countplot(data=data, x='marital-status', hue='income')

# %% [markdown]
# ##### **Observações:**

# %% [markdown]
# *   Quase todas as pessoas com renda superior a 50 mil são casadas.
# *   As pessoas solteiras têm renda comparativamente muito mais baixa.

# %% [markdown]
# #### **Ocupação** (*occupation*)

# %% [markdown]
# Iremos iniciar reduzindo o número de categorias da variável Ocupação (occupation), agrupando diversas categorias pelo tipo de trabalho que desempenham, criando apenas 6 grupos representando setores distintos da economia, são eles:
# *   Trabalhadores Operacional (Operários)
# *   Trabalhadores Administrativo e Executivo
# *   Trabalhadores do Conhecimento
# *   Trabalhadores em Vendas e Serviços
# *   Trabalhadores do Setor primário (agricultura, pecuária, pesca, extrativismo)
# *   Militares

# %%
# Agrupando os ocupações por setor de trabalho.

data['occupation'].replace(['Tech-support', 'Craft-repair', 'Handlers-cleaners', 'Transport-moving', 'Machine-op-inspct'], 'Blue-collar', inplace=True)
data['occupation'].replace(['Exec-managerial', 'Adm-clerical'], 'White-collar', inplace=True)
data['occupation'].replace('Prof-specialty', 'Gold-collar', inplace=True)
data['occupation'].replace(['Other-service', 'Sales', 'Priv-house-serv', 'Protective-serv'], 'Pink-collar', inplace=True)
data['occupation'].replace('Farming-fishing', 'Green-collar', inplace=True)
data['occupation'].replace('Armed-Forces', 'Brown-collar', inplace=True)

plt.figure(figsize=(16, 4))
sns.countplot(data=data, x='occupation')

plt.figure(figsize=(16,4))
sns.countplot(data=data, x='occupation', hue='income')

# %% [markdown]
# ##### **Observações:**

# %% [markdown]
# *   Operários e trabalhadores do setor de vendas e serviços têm a renda mais baixa.
# *   Trabalhadores do conhecimento são as profissões mais bem pagas.

# %% [markdown]
# #### **País de origem** (*native-country*)

# %% [markdown]
# Iremos reduzie o número de categorias da variável Paíos de origem (native-country), agrupando as nacionalidades em 2 grupos distintos, são eles:
# *   Estados Unidos
# *   Outros

# %%
# Agrupando as nacionalidades em Estados Unidos e outros.

data['native-country'] = data['native-country'].map(lambda country: 'US' if country == 'United-States' else 'Other')

data.head()

# %%
# Plotando os gráficos

plt.figure(figsize=(16, 4))
sns.countplot(data=data, x='native-country')

plt.figure(figsize=(16,4))
sns.countplot(data=data, x='native-country', hue='income')

# %% [markdown]
# ##### **Observações:**

# %% [markdown]
# *   Há uma quantidade desproporcionalmente maior de pessoas nascidas nos Estados Unidos, portanto a maioria das pessoas de alta renda também provém desse país.

# %% [markdown]
# #### **Idade** (*age*)

# %% [markdown]
#  Separamos as diversas idades em faixas etárias, para melhor caracterizar a distribbuição de idades e rendas do dataset. E em seguida plotamos o gráfico Faixa etária versus renda.

# %%
data['age-range']= pd.cut(data['age'], 10)

plt.figure(figsize=(16,4))
sns.countplot(data=data, x='age-range', hue='income')

# %% [markdown]
# ##### **Observações:**

# %% [markdown]
# *   Pessoas de maior renda tem idades na faixa dos 30 à 50 anos.
# *   Pessoa jovens tem renda muito menor do que pessoas de meia-idade.

# %% [markdown]
# #### **Relacionamento familiar** (*relationship*)

# %% [markdown]
# Primeiro fazemos o gráfico das variáveis Relacionamento familiar versus a Renda.

# %%
plt.figure(figsize=(16,4))
sns.countplot(data=data, x='relationship', hue='income')

# %% [markdown]
# ##### **Observações:**

# %% [markdown]
# *   Maridos e esposas apresentam maior renda familiar, o que é condizente com o fato de que pessoas casadas ganham mais que os demais.

# %% [markdown]
# #### **Sexo** (*sex*)

# %% [markdown]
# Primeiro fazemos o gráfico das variáveis Sexo versus a Renda.

# %%
sns.countplot(data=data, x='sex', hue='income')

# %% [markdown]
# ##### **Observações:**

# %% [markdown]
# *   Mais pessoas do sexo masculino tem alta renda do que do sexo feminino.

# %% [markdown]
# #### **Raça** (*race*)

# %% [markdown]
# Primeiro fazemos o gráfico das variáveis Raça versus a Renda.

# %%
plt.figure(figsize=(16,4))
sns.countplot(data=data, x='race', hue='income')

# %% [markdown]
# ##### **Observações:**

# %% [markdown]
# *   Como há muito mais pessoas brancas do que pertencentes à outras etnias, elas também apresentam maior renda. Entre as outras etnias, os negros são o grupo mais significativo em relação à renda.

# %% [markdown]
# #### **Horas (trabalhadas) por semana** (*hours-per-week*)

# %% [markdown]
#  Separamos as horas trabalhadas por semanas em intervalos para melhor caracterizar a distribbuição entre a quantidade de horas trabalhadas e a renda. E em seguida plotamos o gráfico das faixas de horas trabalhadas versus renda.

# %%
data['h/w-range']= pd.cut(data['hours-per-week'], 5)

plt.figure(figsize=(16,4))
sns.countplot(data=data, x='h/w-range', hue='income')

# %% [markdown]
# ##### **Observações:**

# %% [markdown]
# *   Pessoas com maior renda trabalham entre 20 e 60 horas semanais.

# %% [markdown]
# # **6. Processamento dos dados**

# %% [markdown]
# Para o processamento dos dados realizou-se os seguintes procedimentos:
# *   Removeu-se outliers na variável ganho de capital para generalizar melhor os dados.
# *   Para lidar com os valores ausentes removeu-se registros com quaisquer atributos nulos. Visto que não fará uma diferença significativa no treinamento, pois os registros nulos representam apenas cerca de 5% de todos os dados de treinamento.
# *   Removeu-se duplicatas para evitar overfitting.
# *   Descartou-se a variável **fnlwgt** porque é um recurso inútil.
# *   Entre as variáveis Educação e Número de Educação, usamos apenas Número de Educação, pois contém toda a informação presente em Educação.
# *   Combinou-se ganhos e perdas de capital para torná-lo um único recurso.
# *   Todos os outros recursos permanecem inalterados.
# 

# %%
# Removendo ouliers da variável ganho de capital
outliers= data[data['capital-gain'] > 40000].index
data= data.drop(outliers)

# Removendo linhas com valores faltantes ou duplicadas
data= data.dropna(how='any', axis=0)
data= data.drop_duplicates()

# Removendo as colunas fnlwgt e education
data= data.drop('fnlwgt', axis=1)
data= data.drop(columns='education')

# Combinando as colunas ganho e perda de capital em uma única coluna
data['capital-gain'] = data.apply(lambda capital: (capital['capital-gain'] - capital['capital-loss']), axis=1)

# Removendo demais colunas desnecessárias
data= data.drop(columns='capital-loss')
data= data.drop(columns='age-range')
data= data.drop(columns='h/w-range')

# verificando o novo dataset
data.info()

# %% [markdown]
# Verifica-se que não há colunas com valores nulos.
# Pode-se portanto verificar as estatísticas descritivas das colunas numéricas restantes.

# %%
data.describe()

# %% [markdown]
# Pode-se então verificar como ficaram as correlações após as alterações feitas nos atributos.

# %%
correlacao_nova =data[['age', 'education-num', 'capital-gain', 'hours-per-week']].corr()

heatMap(correlacao_nova)

# %% [markdown]
# ##### **Observações:**

# %% [markdown]
# *   Houve uma melhora significativa nas correlações dos dados numéricos.
# *   Pode-se, então, seguir para a construção do modelo.
# 
# 

# %% [markdown]
# # **7. Construção do modelo**

# %% [markdown]
# Importando os elementos necessários do Sci-Kit Learn.

# %%
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score, cross_val_predict
cv = KFold(6)
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix)
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler

model = []
accuracy = []
f1 = []
auc = []

# %%
data = pd.get_dummies(data, drop_first=True)
data

# %% [markdown]
# Gerando os datasets de treino e normalizando os dados.

# %%
# Dividindo o dataset
X = data[data.columns[:-1]]
y = data[data.columns[-1]]


# %%
# Normalizando os dados

scaler = StandardScaler().fit(X)
x_scaled = scaler.transform(X)

# %% [markdown]
# Importando modelo de Classificador por Árvore de Decisões

# %%
from sklearn.tree import DecisionTreeClassifier, plot_tree

tree_clf_income = DecisionTreeClassifier(max_depth=3,random_state=42,criterion="gini",splitter='best',max_features=7)
tree_clf_income.fit(X,y)

#aram_grid = {'max_depth': [5, 10, 50, 100, None],
#              'criterion': ['gini','entropy']}
#grid2 = GridSearchCV(dtc, param_grid, cv=cv).fit(x_scaled, y)
#print("DTC: ", grid2.best_score_, grid2.best_params_)


# %%
fig,ax = plt.subplots(figsize=(30,30))
plot_tree(tree_clf_income,ax=ax,feature_names=X.columns,
        rounded=True,filled=True,precision=2,class_names=['Pobre','Rico'])
plt.show()

# %%
from sklearn.metrics import classification_report
print(classification_report(y,tree_clf_income.predict(X)))

# %% [markdown]
# # **8. Aplicação do modelo aos dataset de testes**

# %% [markdown]
# Carregando o dataset de testes.

# %%
test= pd.read_csv('data/adult.test', sep=', ', names=colunas, na_values='?', engine='python', skiprows=1)
test.info()

# %% [markdown]
# Processando o dataset de testes

# %%
test= test.dropna(how='any', axis=0)
test= test.drop_duplicates()
test= test.drop('fnlwgt', axis=1)
test= test.drop(columns='education')

test['capital-gain']= test.apply(lambda x: (x['capital-gain'] - x['capital-loss']), axis=1)
test= test.drop(columns='capital-loss')

test.info()

# %% [markdown]
# Realizando engenharia de atributos no dataset de testes.

# %%
test['workclass'].replace(['State-gov', 'Federal-gov', 'Local-gov'], 'Government', inplace=True)
test['workclass'].replace(['Self-emp-not-inc', 'Self-emp-inc'], 'Self', inplace=True)
test['workclass'].replace(['Without-pay', 'Never-worked'], 'Others', inplace=True)

test['marital-status'].replace(['Married-AF-spouse', 'Married-civ-spouse', 'Married-spouse-absent'], 'Married', inplace=True)
test['marital-status'].replace('Divorced', 'Separated',inplace=True)

test['occupation'].replace(['Tech-support', 'Craft-repair', 'Handlers-cleaners', 'Transport-moving', 'Machine-op-inspct'], 'Blue-collar', inplace=True)
test['occupation'].replace(['Exec-managerial', 'Adm-clerical'], 'White-collar', inplace=True)
test['occupation'].replace('Prof-specialty', 'Gold-collar', inplace=True)
test['occupation'].replace(['Other-service', 'Sales', 'Priv-house-serv', 'Protective-serv'], 'Pink-collar', inplace=True)
test['occupation'].replace('Farming-fishing', 'Green-collar', inplace=True)
test['occupation'].replace('Armed-Forces', 'Brown-collar', inplace=True)

test['native-country']= test['native-country'].map(lambda country: 'US' if country == 'United-States' else 'Other')

# %%
test = pd.get_dummies(test, drop_first=True)
test

# %% [markdown]
# Preparando o dataset de testes para rodar o modelo.

# %%
Xtest = test[test.columns[:-1]]
ytest = test[test.columns[-1]]

scaler = StandardScaler().fit(Xtest)
xtest_scaled = scaler.transform(Xtest)

# %%
xtest_scaled.shape

# %% [markdown]
# Rodando o modelo para gerar previsões

# %%
print('Conjunto de Teste:')
print(classification_report(ytest,tree_clf_income.predict(Xtest)))


