import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from flask import abort
import json
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

years = [2017,2018,2019]
activity_url = 'https://amphis.rails-apps.de/sites/{}/activity.csv?year={}'
sites = [3, 4]
forecast_url = 'https://amphis.rails-apps.de/sites/{}/forecasts.csv'
class_weight = {"0": 1, "1-50": 1, "50-150": 5, "150-300": 20, ">300": 40}

def predict_toad_migration(request):
    df_training = get_training_data()
    df_forecast = get_forecast_data()
    df_predictions = df_forecast[['Datum', 'Site']]
    classifier, df_forecast_new = train(pd.concat([df_training, df_forecast]))
    y_pred_live = classifier.predict(df_forecast_new)
    df_predictions['Anzahl'] = y_pred_live
    return json.dumps(df_predictions.to_dict(orient='records'))

def get_training_data():
    df4 = pd.concat([pd.read_csv(activity_url.format(4, year)) for year in years])
    df3 = pd.concat([pd.read_csv(activity_url.format(3, year)) for year in years])
    df_training = pd.concat([df3, df4])
    df_training=df_training[['Datum','Luftfeuchtigkeit [%]','Anzahl','Site','Temperature min','Temperature max','Windspeed avg','Windspeed max']]
    df_training.columns = df_training.columns.str.replace('[^a-zA-Z]', '')
    df_training['purpose']='training'
    df_training.dropna(inplace=True)
    return df_training

def get_forecast_data():
    sites = [3, 4]
    forecast_url = 'https://amphis.rails-apps.de/sites/{}/forecasts.csv'
    dfs = []
    for site in sites:
        try:
            dfs.append(pd.read_csv(forecast_url.format(site)))
        except:
            print("No forecast data for site %d" % site)
    df_forecast = pd.concat(dfs)
    df_forecast.rename(columns={'Standort-ID':'Site', \
                           'Humidity avg':'Luftfeuchtigkeit', \
                           'Min. Temp. (°C)': 'Temperaturemin', \
                           'Max. Temp. (°C)': 'Temperaturemax', \
                           'Wind speed avg kmh': 'Windspeedavg', \
                           'Wind speed max kmh': 'Windspeedmax'}, inplace=True)
    df_forecast=df_forecast[['Datum','Site','Luftfeuchtigkeit','Temperaturemin','Temperaturemax','Windspeedavg', 'Windspeedmax']]
    df_forecast['purpose']='forecast'
    df_forecast.dropna(inplace=True)
    return df_forecast

def train(df):
    df['date'] = df['Datum'].astype('datetime64')
    df = df.drop(columns=['Datum'])
    df['doy'] = df.apply(lambda x: x.date.timetuple().tm_yday, axis=1)
    df['year'] = df.apply(lambda x: x.date.year, axis=1)

    df['passed'] = df.apply(lambda x: passed(df, x.Site, x.year, x.doy), axis=1)

    df_hist = add_history_data(df)
    df_hist['bucket'] = df_hist.apply(lambda x: bucket(x.Anzahl, x.purpose), axis=1)

    df_hist = df_hist.fillna(0)
    df_hist = float2int(df_hist)

    df_training_new = df_hist.query("purpose=='training'")
    df_forecast_new = df_hist.query("purpose=='forecast'")
    df_x = df_training_new.drop(columns=['Anzahl', 'date', 'bucket', 'purpose'])
    df_y = df_training_new[['bucket']]
    df_forecast_new.drop(columns=['purpose', 'date', 'bucket', 'Anzahl'], inplace=True)

    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.20)
    classifier_train = LogisticRegression(class_weight=class_weight)
    classifier_train.fit(x_train, y_train)
    y_pred = classifier_train.predict(x_test)

    print("--- Metrics: accuracy_score, classification_report, confusion_matrix ---")
    print(accuracy_score(y_pred, y_test))
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    classifier= LogisticRegression(class_weight=class_weight)
    classifier.fit(df_x, df_y)
    return classifier, df_forecast_new

def passed(df, Site, year, doy):
    df_filtered = df[(df['Site'] == Site) \
       & (df['year']==year) \
       & (df['doy'] < doy) ]
    return df_filtered.Anzahl.sum()

def bucket(Anzahl, purpose):
    if(purpose=='forecast'):
        return ""
    if(Anzahl==0):
        return "0"
    if(Anzahl<50):
        return "1-50"
    if(Anzahl<150):
        return "50-150"
    if(Anzahl<300):
        return "150-300"
    return ">300"

def add_history_data(df):
    df_day1 = df.copy().drop(columns=['purpose']).rename(index=str, columns={"Luftfeuchtigkeit": "Luftfeuchtigkeit_day1", "Anzahl": "Anzahl_day1", \
                                                   "Temperaturemin": "Temperaturemin_day1",
                                                   "Temperaturemax": "Temperaturemax_day1", \
                                                   "Windspeedavg": "Windspeedavg_day1",
                                                   "Windspeedmax": "Windspeedmax_day1", "passed": "passed_day1"})
    df_day1['date'] = pd.to_datetime(df_day1['date']).apply(pd.DateOffset(-1))
    df_day1.drop(columns=['year', 'doy'], inplace=True)
    df_hist1 = pd.merge(df, df_day1, how='left', on=['date', 'Site'])

    df_day2 = df.copy().drop(columns=['purpose']).rename(index=str, columns={"Luftfeuchtigkeit": "Luftfeuchtigkeit_day2", "Anzahl": "Anzahl_day2", \
                                                   "Temperaturemin": "Temperaturemin_day2",
                                                   "Temperaturemax": "Temperaturemax_day2", \
                                                   "Windspeedavg": "Windspeedavg_day2",
                                                   "Windspeedmax": "Windspeedmax_day2", "passed": "passed_day2"})
    df_day2['date'] = pd.to_datetime(df_day2['date']).apply(pd.DateOffset(-2))
    df_day2.drop(columns=['year', 'doy'], inplace=True)
    df_hist2 = pd.merge(df_hist1, df_day2, how='left', on=['date', 'Site'])

    df_day3 = df.copy().drop(columns=['purpose']).rename(index=str, columns={"Luftfeuchtigkeit": "Luftfeuchtigkeit_day3", "Anzahl": "Anzahl_day3", \
                                                   "Temperaturemin": "Temperaturemin_day3",
                                                   "Temperaturemax": "Temperaturemax_day3", \
                                                   "Windspeedavg": "Windspeedavg_day3",
                                                   "Windspeedmax": "Windspeedmax_day3", "passed": "passed_day3"})
    df_day3['date'] = pd.to_datetime(df_day3['date']).apply(pd.DateOffset(-3))
    df_day3.drop(columns=['year', 'doy'], inplace=True)
    df_hist3 = pd.merge(df_hist2, df_day3, how='left', on=['date', 'Site'])
    return df_hist3

def float2int(df):
    float_col = df.select_dtypes(include = ['float64']) # This will select float columns only
    for col in float_col.columns.values:
        df[col] = df[col].astype('int64')
    return df