from ast import literal_eval
from math import sqrt
from multiprocessing import cpu_count
from warnings import catch_warnings
from warnings import filterwarnings

import pandas
from joblib import Parallel
from joblib import delayed
from matplotlib import pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX


# plots the dataaet using several graphs to find general characteristics
def plot_data(dataset):
    dataset = dataset.drop(columns="LICENCE_ID")

    # univariate plots
    dataset.plot(kind='box', subplots=True, sharex=False, sharey=False)
    plt.savefig("Box Graphs.png")

    # histograms
    dataset.hist()
    plt.savefig("Histograms.png")
    plt.close()
    plt.clf()

    # scatterplot
    scatter_matrix(dataset, figsize=(12, 12))
    plt.savefig("scatter_matrix.png")
    plt.close()


# convert the dataset into format of months against total sales
def get_sales(dataset):
    dataset = pandas.to_datetime(dataset["LICENCE_START_DT"])
    times_month = dataset.dt.to_period("M")
    sales_dataset = times_month.value_counts(sort=False)
    return sales_dataset.sort_index()


# plots the sales dataset to get characteristics of total sales per month
def plot_sales(sales_dataset):
    # plots all data into single line graph
    sales_dataset = sales_dataset.to_frame("Sales")
    sales_dataset.plot()
    plt.savefig("Sales Per Month.png")

    # plots different years onto same scale of months
    sales_dataset.index.name = "Months"
    print(sales_dataset)
    print("index type is ", sales_dataset.index.dtype)
    sales_2014 = sales_dataset.loc['2014-01':'2014-12']
    sales_2015 = sales_dataset.loc['2015-01':'2015-12']
    sales_2016 = sales_dataset.loc['2016-01':'2016-12']
    sales_2017 = sales_dataset.loc['2017-01':'2017-12']
    sales_2018 = sales_dataset.loc['2018-01':'2018-12']
    plt.close()
    plt.plot(dates_to_months(sales_2014))
    plt.plot(dates_to_months(sales_2015))
    plt.plot(dates_to_months(sales_2016))
    plt.plot(dates_to_months(sales_2017))
    plt.plot(dates_to_months(sales_2018))
    plt.legend(["2014", "2015", "2016", "2017", "2018"], loc='upper left')
    plt.savefig("Sales Month Comparison.png")

    # plots autocorelation graph
    plt.close()
    plot_acf(sales_dataset)
    plt.savefig("autocorrelation.png")

    # plots partial autocorelation graph
    plt.close()
    plot_pacf(sales_dataset)
    plt.savefig("Partial autocorrelation.png")


# convert pandas date and year objects to months
def dates_to_months(dates):
    dates.index = [dates.month for dates in dates.index]
    return dates


# trains SARIMA model and preidcts history
def sarima_forecast(history, config):
    order, sorder, trend = config
    # define model
    model = SARIMAX(history, order=order, seasonal_order=sorder, trend=trend, enforce_stationarity=False,
                    enforce_invertibility=False)
    # fit model
    model_fit = model.fit(disp=False)
    # make one step forecast
    yhat = model_fit.predict(len(history), len(history))
    return yhat[0]


# gets RMSE for evaluation
def measure_rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))


# split the dataset into test and training, keeping the order of data
def train_test_split(dataset, n_test):
    return dataset[:-n_test], dataset[-n_test:]


# walk-forward validations each model with given parameters
def walk_forward_validation(dataset, n_test, parameters):
    predictions = list()
    # split dataset
    train, test = train_test_split(dataset, n_test)
    # seed history with training dataset
    history = [x for x in train]
    # step over each time-step in the test set
    for i in range(len(test)):
        # fit model and make forecast for history
        yhat = sarima_forecast(history, parameters)
        # store forecast in list of predictions
        predictions.append(yhat)
        # add actual observation to history for the next loop
        history.append(test[i])
    # estimate prediction error
    error_rmse = measure_rmse(test, predictions)
    return error_rmse


# evaluates model with given parameters using RMSE
def evaluate_model(dataset, n_test, parameters, debug=False):
    result = None
    # convert config to a key
    key = str(parameters)
    # show all warnings and fail on exception if debugging
    if debug:
        result = walk_forward_validation(dataset, n_test, parameters)
    else:
        # one failure during model validation suggests an unstable config
        try:
            # never show warnings when grid searching, too noisy
            with catch_warnings():
                filterwarnings("ignore")
                result = walk_forward_validation(dataset, n_test, parameters)
        except:
            error = None
    # check for an interesting result
    if result is not None:
        print(' > Model[%s] %.3f' % (key, result))
    return key, result


# trains all possible SARIMA models with given parameters
def grid_search(dataset, cfg_list, n_test, parallel=True):
    if parallel:  # searches using multiple CPU cores if possible
        # execute configs in parallel
        executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
        tasks = (delayed(evaluate_model)(dataset, n_test, cfg) for cfg in cfg_list)
        errors = executor(tasks)
    else:  # if only one CPU
        errors = [evaluate_model(dataset, n_test, cfg) for cfg in cfg_list]
    # remove empty results
    errors = [r for r in errors if r[1] is not None]
    # sort configs by RMSE, ascending order
    errors.sort(key=lambda tup: tup[1])
    return errors


# create a list of all SARIMA parameters for grid search
def sarima_configs():
    models = list()
    # define config lists
    p_params = [1, 2, 3]
    d_params = [1]
    q_params = [1, 2, 3]
    t_params = ['ct']
    P_params = [1, 2, 3]
    D_params = [0]
    Q_params = [1, 2, 3]
    m_params = [12]
    # create config instances
    for p in p_params:
        for d in d_params:
            for q in q_params:
                for t in t_params:
                    for P in P_params:
                        for D in D_params:
                            for Q in Q_params:
                                for m in m_params:
                                    params_all = [(p, d, q), (P, D, Q, m), t]
                                    models.append(params_all)
    return models


# predicts next three months into 2019 and plots
def predict_next_sales(best_params, dataset):
    order, sorder, trend = best_params
    model = SARIMAX(dataset, order=order, seasonal_order=sorder, trend=trend, enforce_stationarity=False,
                    enforce_invertibility=False)
    model = model.fit(disp=False)

    predictions = list()
    # split dataset
    train, test = train_test_split(dataset, num_test)
    # seed history with training dataset
    history = [x for x in train]
    # step over each time-step in the test set
    for i in range(len(test)):
        # fit model and make forecast for history
        yhat = sarima_forecast(history, best_params)
        # store forecast in list of predictions
        predictions.append(yhat)
        # add actual observation to history for the next loop
        history.append(test[i])
    # estimate prediction error

    # prints and saves accuracy final model
    error_2019 = measure_rmse(test, predictions)
    print("Estimated RMSE is ", error_2019)
    plt.close()
    plt.plot(predictions)
    plt.plot(history[-len(test):])
    plt.savefig("Final Model 2018 estimated sales.png")
    plt.close()

    predictions = model.forecast(3)
    print("Predictions are")
    print(predictions)

    # plot bar graph of predictions
    predictions.plot.bar()
    plt.savefig("2019 Forecast Bar Chart.png")
    plt.close()

    # plot line graph of predictions
    predictions.plot()
    plt.savefig("2019 Forecast Line plot.png")


if __name__ == '__main__':
    data_filepath = "test_data.csv"
    pandas.set_option('display.expand_frame_repr', False)  # display all pandas information when print

    # initial investigations to dataset
    data = pandas.read_csv(data_filepath)
    print("shape of dataset is", data.shape, "\n")
    print("types of columns\n", data.dtypes, "\n")
    print(data.head())
    print(data.columns)
    print(data.nunique())  # number unique values each column
    data = data.drop(columns=["POSTCODE_AREA"])
    print("\nNumber of null values")
    print(data.isnull().sum(axis=0))
    print()
    description = data.describe(include='all')
    print(description)
    res = {col: data[col].value_counts() for col in data[['LICENCE_START_DT', 'PAY_SCHEME',
                                                          'POSTCODE_DISTRICT', 'POSTCODE_SECTOR', 'MOSAIC_GROUP']]}
    print(res)

    # plot datasets
    sales = get_sales(data)
    plot_data(data)
    plot_sales(sales)

    num_test = 12  # data split
    list_params = sarima_configs()  # model configs
    scores = grid_search(sales, list_params, num_test)  # grid search
    print('done')
    # list
    for cfg, error in scores:
        print(cfg, error)

    print(scores[0][0])
    final_params = literal_eval(scores[0][0])
    print(final_params)
    # best perams = [(2, 1, 3), (3, 0, 3, 12), 'ct']

    predict_next_sales(final_params, sales)
