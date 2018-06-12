###############################################################################################################################
#   Código referente a etapa de regressão do projeto da disciplina de Competições em Ciências de Dados(SCC0277) do ICMC-USP   #
#                                                                                                                             # 
#   Autores: Igor Martinelli                                                                                                  #
#            Zoltán Hirata Jetsmen                                                                                            # 
###############################################################################################################################
#Bibliotecas utilizadas no desenvolvimento do trabalho.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet, LinearRegression, Lasso, LassoCV, RidgeCV, ElasticNetCV
from xgboost import XGBRegressor


#Leitura dos conjuntos de treino e teste.
dataset = pd.read_csv('preprocess_phase2/train.csv', sep= ';')
datatest = pd.read_csv('preprocess_phase2/test.csv', sep = ';')

#Adquirindo variáveis.
X_test = datatest.iloc[:, 1:15]
X_train = dataset.iloc[:, 1:15]
y_train = dataset.iloc[:, 15]
ids = np.asarray(datatest.iloc[:, 0])

#Aplicando log no valor das casas para se obter um melhor valor na regressão.
y_train = np.log1p(np.asarray(y_train))

#Função model_rmse_cv -> tal função auxiliou nas etapas de teste manual realizadas, mensurando o erro quadrático médio dos modelos testados.
def model_rmse_cv(model, Xtrain, y_train, X_test, y_test):
    regressor = model
    regressor.fit(Xtrain, y_train)
    
    rmse = np.sqrt(-cross_val_score(regressor, X_train, y_train, scoring='neg_mean_squared_error', cv=10))
    return np.mean(rmse)

#Função answer_generate -> função responsável por gerar o csv para submissão no site kaggle.
def answer_generate(model, X_test, ids):
    y_pred = np.asarray(np.expm1(model.predict(X_test)))
    csv_file = pd.DataFrame({"Id": ids, "SalePrice": y_pred})
    csv_file.to_csv("sol.csv", index=False)

###############################################################################################################################
#                                                                                                                             #
#                                                    ElasticNet Model                                                         #
#                                                                                                                             #  
###############################################################################################################################
def elastic_net(X_train, y_train, X_test, ids):
    
    elasticNet = ElasticNetCV(l1_ratio = [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1],
                              alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 
                                        0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6], 
                              max_iter = 50000, cv = 10)
    elasticNet.fit(X_train, y_train)
    alpha = elasticNet.alpha_
    ratio = elasticNet.l1_ratio_
    print("Best l1_ratio :", ratio)
    print("Best alpha :", alpha )
    
    print("Try again for more precision with l1_ratio centered around " + str(ratio))
    elasticNet = ElasticNetCV(l1_ratio = [ratio * .85, ratio * .9, ratio * .95, ratio, ratio * 1.05, ratio * 1.1, ratio * 1.15],
                              alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6], 
                              max_iter = 50000, cv = 10)
    elasticNet.fit(X_train, y_train)
    if (elasticNet.l1_ratio_ > 1):
        elasticNet.l1_ratio_ = 1    
    alpha = elasticNet.alpha_
    ratio = elasticNet.l1_ratio_
    print("Best l1_ratio :", ratio)
    print("Best alpha :", alpha )
    
    print("Now try again for more precision on alpha, with l1_ratio fixed at " + str(ratio) + 
          " and alpha centered around " + str(alpha))
    elasticNet = ElasticNetCV(l1_ratio = ratio,
                              alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85, alpha * .9, 
                                        alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15, alpha * 1.25, alpha * 1.3, 
                                        alpha * 1.35, alpha * 1.4], 
                              max_iter = 50000, cv = 10)
    elasticNet.fit(X_train, y_train)
    if (elasticNet.l1_ratio_ > 1):
        elasticNet.l1_ratio_ = 1    
    alpha = elasticNet.alpha_
    ratio = elasticNet.l1_ratio_
    print("Best l1_ratio :", ratio)
    print("Best alpha :", alpha )
    
    answer_generate(elasticNet, X_test, ids)


###############################################################################################################################
#                                                                                                                             #
#                                                     Lasso Model                                                             #
#                                                                                                                             #  
###############################################################################################################################
def lasso(X_train, y_train, X_test, ids):

    lasso = LassoCV(alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 
                              0.3, 0.6, 1], 
                    max_iter = 50000, cv = 10)
    lasso.fit(X_train, y_train)
    alpha = lasso.alpha_
    print("Best alpha :", alpha)
    
    print("Try again for more precision with alphas centered around " + str(alpha))
    lasso = LassoCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, 
                              alpha * .85, alpha * .9, alpha * .95, alpha, alpha * 1.05, 
                              alpha * 1.1, alpha * 1.15, alpha * 1.25, alpha * 1.3, alpha * 1.35, 
                              alpha * 1.4], 
                    max_iter = 50000, cv = 10)
    lasso.fit(X_train, y_train)
    alpha = lasso.alpha_
    print("Best alpha :", alpha)
    
    answer_generate(lasso, X_test, ids)


###############################################################################################################################
#                                                                                                                             #
#                                                       Ridge Model                                                           #
#                                                                                                                             #                                                              `                                               
###############################################################################################################################
def ridge(X_train, y_train, X_test, ids):

    ridge = RidgeCV(alphas = [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60])
    ridge.fit(X_train, y_train)
    alpha = ridge.alpha_
    print("Best alpha :", alpha)
    
    print("Try again for more precision with alphas centered around " + str(alpha))
    ridge = RidgeCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85, 
                              alpha * .9, alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15,
                              alpha * 1.25, alpha * 1.3, alpha * 1.35, alpha * 1.4], 
                    cv = 10)
    ridge.fit(X_train, y_train)
    alpha = ridge.alpha_
    print("Best alpha :", alpha)

    answer_generate(ridge, X_test, ids)
    
###############################################################################################################################
#                                                                                                                             #
#                                                   XGBRegressor Model                                                        #
#                                                                                                                             #                                                              `                                               
###############################################################################################################################
def xgb(X_train, y_train, X_test, ids):
    model_xgb = XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1)
    model_xgb.fit(X_train, y_train)
   
    answer_generate(model_xgb, X_test, ids)

#Functions Call
lasso(X_train, y_train, X_test, ids) 
ridge(X_train, y_train, X_test, ids) 
elastic_net(X_train, y_train, X_test, ids) 
xgb(X_train, y_train, X_test, ids) 

    
    
    
    