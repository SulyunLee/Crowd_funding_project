from sklearn.linear_model import LinearRegression,LassoCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import Imputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np
import pandas as pd

def calculate_vif(X, thresh=5.0):
    dropped = True
    while dropped:
        variables = X.columns
        dropped = False
        vif = [variance_inflation_factor(X[variables].values, X.columns.get_loc(var)) for var in X.columns]
        max_vif = max(vif)
        if max_vif > thresh:
            maxloc = vif.index(max_vif)
            X = X.drop([X.columns.tolist()[maxloc]], axis=1)
            dropped = True
        return X

def linreg_model(feature_name, label_name, df, write_filename):
    X = df[feature_name]
    X = sm.add_constant(X)
    # X = calculate_vif(X)
    y = df[label_name]
    model = sm.OLS(y, X)
    results = model.fit()

    # write to csv file
    with open(write_filename, 'w') as fh:
        fh.write(results.summary().as_csv())
    print(results.summary())

    return results

def lasso_model(feature_name, label_name, df, write_filename):
    df_modified = df[feature_name + label_name].dropna()
    print(df_modified.shape)
    clf = LassoCV(cv=5)
    clf.fit(df_modified[feature_name], df_modified[label_name])
    print(clf.coef_)
    print(clf.score(df_modified[feature_name], df_modified[label_name]))

def model_rf(train_X, train_y, test_X, test_y, n_estimators):
    model = RandomForestRegressor(n_estimators=n_estimators)
    model.fit(train_X, train_y)
    y_pred = model.predict(test_X)
    test_r2 = r2_score(test_y, y_pred)
    return test_r2

def model_svm(train_X, train_y, test_X, test_y, kernel):
    model = SVR(kernel=kernel)
    model.fit(train_X, train_y)
    y_pred = model.predict(test_X)
    test_r2 = r2_score(test_y, y_pred)
    return test_r2

def rf_model_find_param(feature_name, label_name, df):
    X = df[feature_name]
    y = df[label_name]

    n_estimator_lst = [2, 4, 8, 16, 32]
    results = []
    for n_estimator in n_estimator_lst:
        cv = StratifiedKFold(n_splits=5, shuffle=True)
        
        score_sum = 0
        for (train,test), i in zip(cv.split(X, y), range(5)):
            test_r2 = model_rf(X.iloc[train], y.iloc[train], X.iloc[test], y.iloc[test], n_estimator)
            score_sum += test_r2
        results.append(score_sum / 5)

    best_param = n_estimator_lst[results.index(max(results))]

    return best_param, max(results)

def svm_model_find_param(feature_name, label_name, df):
    X = df[feature_name]
    y = df[label_name]

    kernel_lst = ['linear', 'rbf', 'sigmoid']
    results = []
    for kernel in kernel_lst:
        cv = StratifiedKFold(n_splits=5, shuffle=True)

        score_sum = 0
        for (train,test), i in zip(cv.split(X, y), range(5)):
            test_r2 = model_svm(X.iloc[train], y.iloc[train], X.iloc[test], y.iloc[test], kernel)
            score_sum += test_r2
        results.append(score_sum / 5)

    best_param = kernel_lst[results.index(max(results))]

    return best_param, max(results)

def rf_simulate(best_param, model_feature, model_label, df, changing_var_name, changing_value):
    model = RandomForestRegressor(n_estimators = best_param)
    X = df[model_feature]
    y = df[model_label]
    model.fit(X, y)
    
    df_changed = df.copy()
    df_changed[changing_var_name] = changing_value
    X_changed = df_changed[model_feature]

    y_pred_simulated = model.predict(X_changed)
    df_changed[model_label] = y_pred_simulated

    result_df = df_changed.groupby('Year')[model_label].sum()

    return result_df

def linreg_simulate(model_feature, model_label, df, changing_var_name, changing_value):
    model = LinearRegression()
    X = df[model_feature]
    y = df[model_label]
    model.fit(X, y)

    df_changed = df.copy()
    df_changed[changing_var_name] = changing_value
    X_changed = df_changed[model_feature]

    y_pred_simulated = model.predict(X_changed)
    df_changed[model_label] = y_pred_simulated

    result_df = df_changed.groupby('Year')[model_label].sum()

    return result_df

def linreg_simulate_num_SSPs(model_feature, model_label, df):
    # change the number of SSPs to 0
    simulation_result1 = linreg_simulate(model_feature, model_label, df, "NUMBER OF SSPs", 0)

    # change the number of SSPs to half
    simulation_result2 = linreg_simulate(model_feature, model_label, df, "NUMBER OF SSPs", df['NUMBER OF SSPs']/2)

    # change the number of SSPs to double
    simulation_result3 = linreg_simulate(model_feature, model_label, df, "NUMBER OF SSPs", df['NUMBER OF SSPs']*2)

    result_df = pd.concat([df.groupby('Year')[model_label].sum(), simulation_result1, simulation_result2, simulation_result3], axis=1)
    result_df.columns = ['Actual', 'Simulated zero', 'Simulated half', 'Simulated double']

    result_df.to_csv("results/linreg_{}_numSSPs_simulation_result.csv".format(model_label))

def linreg_simulate_SSP_legality(model_feature, model_label, df):
    # change the SSP legality to 0
    simulation_result1 = linreg_simulate(model_feature, model_label, df, "SSP LEGALITY BINARY", 0)
    
    # change the SSP legality to 1
    simulation_result2 = linreg_simulate(model_feature, model_label, df, "SSP LEGALITY BINARY", 1)

    result_df = pd.concat([df.groupby('Year')[model_label].sum(), simulation_result1, simulation_result2], axis=1)
    result_df.columns = ['Actual', 'All SSP illegal', "All SSP legal"]

    result_df.to_csv("results/linreg_{}_SSPLegality_simulation_result.csv".format(model_label))

def rf_simulate_num_SSPs(rf_best_param, model_feature, model_label, df):
    # change the number of SSPs to 0
    simulation_result1 = rf_simulate(rf_best_param, model_feature, model_label, df, "NUMBER OF SSPs", 0)

    # change the number of SSPs to half
    simulation_result2 = rf_simulate(rf_best_param, model_feature, model_label, df, "NUMBER OF SSPs", df['NUMBER OF SSPs']/2)

    # change the number of SSPs to double
    simulation_result3 = rf_simulate(rf_best_param, model_feature, model_label, df, "NUMBER OF SSPs", df['NUMBER OF SSPs']*2)

    result_df = pd.concat([df.groupby('Year')[model_label].sum(), simulation_result1, simulation_result2, simulation_result3], axis=1)
    result_df.columns = ['Actual', 'Simulated zero', 'Simulated half', 'Simulated double']

    result_df.to_csv("results/rf_{}_numSSPs_simulation_result.csv".format(model_label))

def rf_simulate_SSP_legality(rf_best_param, model_feature, model_label, df):
    # change the SSP legality to 0
    simulation_result1 = rf_simulate(rf_best_param, model_feature, model_label, df, "SSP LEGALITY BINARY", 0)

    # change the SSP legality to 1
    simulation_result2 = rf_simulate(rf_best_param, model_feature, model_label, df, "SSP LEGALITY BINARY", 1)
    
    result_df = pd.concat([df.groupby('Year')[model_label].sum(), simulation_result1, simulation_result2], axis=1)
    result_df.columns = ['Actual', 'All SSP illegal', 'All SSP legal']

    result_df.to_csv("results/rf_{}_SSPLegality_simulation_result.csv".format(model_label))

if __name__ == "__main__":
    data_dir = "cleaned_dataset/"
    results_dir = "results/"

    dataset_filename = data_dir + "cleaned_final_dataset.csv"
    df = pd.read_csv(dataset_filename)
    df_death = df[df['Year'] < 2018]

    # model_feature_withcontrol = ["RW_client", "NUMBER OF SSPs", "SSP LEGALITY BINARY", "No HS diploma", "Poverty", "Uninsured"]
    model_feature = ["NUMBER OF SSPs", "SSP LEGALITY BINARY", "No HS diploma", "Poverty", "Uninsured"]

    ################# predict HIV diagnoses #######################
    model_label = "HIV diagnoses"

    # linear regression model
    linreg_result = linreg_model(model_feature, model_label, df, "{}{}_linreg_model.csv".format(results_dir, model_label))

    # linear regression simulate
    linreg_simulate_num_SSPs(model_feature, model_label, df)
    linreg_simulate_SSP_legality(model_feature, model_label, df)

    # find the best random forest parameters
    rf_best_param, rf_best_result = rf_model_find_param(model_feature, model_label, df)
    print(rf_best_result)
    #imulate
    rf_simulate_num_SSPs(rf_best_param, model_feature, model_label, df)
    rf_simulate_SSP_legality(rf_best_param, model_feature, model_label, df)

    ################# predict HIV deaths ###########################3
    model_label = "HIV deaths"

    # linear regression model
    linreg_result = linreg_model(model_feature, model_label, df_death, "{}{}_linreg_model.csv".format(results_dir, model_label))

    # linear regression simulate
    linreg_simulate_num_SSPs(model_feature, model_label, df_death)
    linreg_simulate_SSP_legality(model_feature, model_label, df_death)

    # find the best random forest parameters
    rf_best_param, rf_best_result = rf_model_find_param(model_feature, model_label, df_death)
    print(rf_best_result)
    #imulate
    rf_simulate_num_SSPs(rf_best_param, model_feature, model_label, df_death)
    rf_simulate_SSP_legality(rf_best_param, model_feature, model_label, df_death)

    ################## predict AIDS diagnoses ##########################
    model_label = "AIDS diagnoses"

    # linear regression model
    linreg_result = linreg_model(model_feature, model_label, df, "{}{}_linreg_model.csv".format(results_dir, model_label))

    # linear regression simulate
    linreg_simulate_num_SSPs(model_feature, model_label, df)
    linreg_simulate_SSP_legality(model_feature, model_label, df)

    # find the best random forest parameters
    rf_best_param, rf_best_result = rf_model_find_param(model_feature, model_label, df)
    print(rf_best_result)
    #imulate
    rf_simulate_num_SSPs(rf_best_param, model_feature, model_label, df)
    rf_simulate_SSP_legality(rf_best_param, model_feature, model_label, df)

    #################### predict AIDS deaths ######################
    model_label = "AIDS deaths"

    # linear regression model
    linreg_result = linreg_model(model_feature, model_label, df_death, "{}{}_linreg_model.csv".format(results_dir, model_label))

    # linear regression simulate
    linreg_simulate_num_SSPs(model_feature, model_label, df_death)
    linreg_simulate_SSP_legality(model_feature, model_label, df_death)

    # find the best random forest parameters
    rf_best_param, rf_best_result = rf_model_find_param(model_feature, model_label, df_death)
    print(rf_best_result)
    #imulate
    rf_simulate_num_SSPs(rf_best_param, model_feature, model_label, df_death)
    rf_simulate_SSP_legality(rf_best_param, model_feature, model_label, df_death)
    

