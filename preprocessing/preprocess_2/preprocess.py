import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
from scipy.stats import skew
from statsmodels.formula.api import ols
import statsmodels.api as sm

sns.set(color_codes=True)
np.random.seed(sum(map(ord, "regression")))
tips = sns.load_dataset("tips")


# Objetivo principal: Realizar a previsão de dos preços das casas de acorda com um conjunto
# dados de entrada

# 1 - Visão geral dos dados

# Primeiramente, iremos realizar uma análise do conjunto de dados que iremos trabalhar
train_data = pd.read_csv("dataset/train.csv", sep=";", index_col=0)
# Impressão das 5 priemeiras linhas
#print(traindata.head())
# Impressão das dimensões da tabela
#print(traindata.shape)
# Pelo output, vemos que temos um dataset com 1460 exemplos e 80 features, sendo que 1 é o valor das casas

# Impressão das informações do tipo e a quantidade de células não nulas de cada atributo, coluna
#print(traindata.info())

# Contagem de quantas colunas são de cada tipo
print(train_data.get_dtype_counts())
# Podemos observar que praticamente metade das colunas são númericas e a outra metade é object, algo que
# precisamos tratar e que será explorado mais para frente

# 2 - Relação entre os atributos e o preço de venda da casa(SalePrice)
#       Nesta seção, será feita uma análise se um certo atributo possui um impacto no preço de venda da casa,
#       para que possamos saber se este vale ou não apena permaner na tabela.

# 2.1 - Atributos numéricos.

# Extração do nome de todas as colunas que possuem atributos numéricos
numerical_features = train_data.select_dtypes(exclude=['object']).dtypes.index
print(numerical_features)  
 
# No primeiro momento, podemos estudar a correlação entre esses atributos e a o preço de venda, para termos
# uma noção se existe alguma relação entre eles.

print(train_data.corr()["SalePrice"].sort_values(ascending=False))

# Pela tabela, podemos considerar que os atributos que possuem mais que 0,5 de correlação, são fortes
# candidatos a serem usados na classificação. Os que ficaram entre 0 e 0,5 serão feitas mais análises
# no futuro e os que ficaram abaixo de 0, somente o atributo "OverallCond" passará por uma nova análise
# já que este normalmente possui uma relação com o preço final da casa, mas não foi detectado pela equação

# Antes de analisar as variáveis uma por uma, iremos verificar se existem relações entre os parâmetros, multicolinearidade.
# Para isso, plotaremos um heatmap das variáveis que possuem uma correlação com salesprice maior que 0.5.

# Atributos que possuem uma correlação maior que 0.5
corrMatrix=train_data[["SalePrice","OverallQual","GrLivArea","GarageCars",
                  "GarageArea","GarageYrBlt","TotalBsmtSF","1stFlrSF","FullBath",
                  "TotRmsAbvGrd","YearBuilt","YearRemodAdd"]].corr()
sns.set(font_scale=1.10)
#plt.figure(figsize=(10, 10))

#sns.heatmap(corrMatrix, vmax=.8, linewidths=0.01,
 #           square=True,annot=True,cmap='viridis',linecolor="white")
#plt.title('Correlation between features');
#plt.savefig("images/Correlation between features.png")
#plt.show()

# Pelo gráfico, conseguimos ver que alguns atributos possuem uma correlação alta: são eles
# (GarageCars, GarageArea)
# (TotRmsAbvGrd, GrLivArea)
# (TotalBsmtSF, 1stFlrSF)
# (YearBuilt, GarageYrBlt)

# Isso quer dizer que essas variáveis podem ser redundantes, indicando querem dizer a mesma coisa para o resultado final
# para isso, podemos deixar só uma delas, excluindo a outra.

# Variáveis que irão ficar:
# GarageArea
# GrLivArea
# TotalBsmtSF
# YearBuilt

# Portanto, se consideramos apoenas essas operações, as variavies que irão ser usadas para a regressão são:
#OverallQual 
#GrLivArea     
#GarageArea  
#TotalBsmtSF    
#FullBath    
#YearBuilt   
#YearRemodAdd 

#Trabalhando com os dados categoricos
categorical_features = train_data.select_dtypes(include=['object']).dtypes.index
print(categorical_features)

#data = pd.read_csv("dataset/PlantGrowth.csv")
categorical_train = train_data[categorical_features].fillna("None", inplace=True)
categorical_train = pd.concat([categorical_train, train_data["SalePrice"]], axis=1)

dic = {};

# Calculo do teste de hipótese utilziando a distribição ANOVA.
for c in categorical_features:
    t = "SalePrice ~ " + c
    if c != 'Alley' and c != "MasVnrType" and c != "BsmtQual" and c != "BsmtCond" and c!= "BsmtExposure" and c != "BsmtFinType1" and c != "BsmtFinType2" and c != "Electrical" and c != "FireplaceQu" and c != "GarageType" and c != "GarageFinish" and c != "GarageQual" and c != "GarageCond" and c != "PoolQC" and c != "Fence" and c != "MiscFeature":
        print(t)
        mod = ols(t, data=train_data).fit()   
        aov_table = sm.stats.anova_lm(mod, typ=2)
        dic[c] = aov_table['PR(>F)'].iloc[0] 
        print(aov_table['PR(>F)'].iloc[0])

# Somente iremos pegar os valores que forem maior tem um valor maior que 0.05
dic = dict((k, v) for k, v in dic.items() if v >= 0.05)

print(dic)

#Features Selecionadas:
# Utilities
# LandSlope
# Street

#Portanto, todas as features que serão usadas no treinamento são:
#OverallQual 
#GrLivArea     
#GarageArea  
#TotalBsmtSF    
#FullBath    
#YearBuilt   
#YearRemodAdd 
#Utilities
#LandSlope
#Street

# Preparando o arquivo final
x = pd.read_csv("dataset/train.csv", sep=";", index_col=0)
y = pd.read_csv("dataset/test.csv", index_col=0)

features = ["OverallQual", "GrLivArea", "GarageArea", "TotalBsmtSF", "FullBath", "YearBuilt", "YearRemodAdd", "Utilities"]
features2 = ["LandSlope", "Street"]
testdata = y[features]
traindata = x[features]

a = x[features2]
b = y[features2]
a = pd.get_dummies(a)
b = pd.get_dummies(b)

traindata = pd.get_dummies(traindata)
testdata = pd.get_dummies(testdata)

testdata["Utilities_NoSeWa"] = 0.0
    
traindata = pd.concat([traindata, a], axis=1)
testdata = pd.concat([testdata, b], axis=1)

traindata = pd.concat([traindata, x["SalePrice"]], axis=1)

testdata.fillna(0, inplace=True)
print(traindata.info())
print(testdata.info())


traindata.to_csv("dataset/finalTrain.csv", sep=";", index_col=0)
testdata.to_csv("dataset/finalTest.csv", sep=";", index_col=0)





