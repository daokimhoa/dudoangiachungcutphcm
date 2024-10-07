# Importing required functions 
import flask
from flask import Flask, render_template
from flask import request
import pandas as pd
import requests
import json
import datetime
#from statsmodels.tsa.arima.model import ARIMA
#from pmdarima import auto_arima
#from pmdarima import auto_arima
#from pmdarima import auto_arima
from joblib import dump, load
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from dateutil.relativedelta import relativedelta
#import numpy as np
#import warnings
#warnings.filterwarnings("ignore", message="numpy.dtype size changed")
app = flask.Flask(__name__, template_folder='templates')
with open(f'lstm_model_2024.joblib', 'rb') as f:
    model = load(f)
@app.route('/')
def main():
 
    # Define Plot Data 
    labels = [
        'January',
        'February',
        'March',
        'April',
        'May',
        'June',
    ]
    #now = datetime.datetime.now()
    thoigian=''
    data = ''
    data_dubao=''
    data_dudoan=''
    sothangdubao=12
    list_dudoan=list()
    #data = list(data)
    #data = list(data2)
    if flask.request.method == 'GET':     
        gia_du_doan='Chưa xác định'
        str_chuoimorong=''
        str_quan_huyen='Thành Phố Hồ Chí Minh'
        #url = "https://gateway.chotot.com/v1/public/api-pty/market-price/chart?cg=1010&region=13000"+str_chuoimorong
        #response = requests.get(url)
        data_tp=''
        data=''
        labels=''        
        ma_quanhuyen=request.args.get('select_quan','')
        str_file_csv='hcm_final.csv'
        if(ma_quanhuyen):
            if(ma_quanhuyen==str(1)):
                str_file_csv="quan_1_final.csv"
                str_quan_huyen="Quận 1, Thành Phố Hồ Chí Minh"
            if(ma_quanhuyen==str(2)):
                str_file_csv="quan_2_final.csv"
                str_quan_huyen="Quận 2, Thành Phố Hồ Chí Minh"
            if(ma_quanhuyen==str(3)):
                str_file_csv="quan_3_final.csv"
                str_quan_huyen="Quận 3, Thành Phố Hồ Chí Minh"
            if(ma_quanhuyen==str(4)):
                str_file_csv="quan_4_final.csv"
                str_quan_huyen="Quận 4, Thành Phố Hồ Chí Minh"
            if(ma_quanhuyen==str(5)):
                str_file_csv="quan_5_final.csv"
                str_quan_huyen="Quận 5, Thành Phố Hồ Chí Minh"
            if(ma_quanhuyen==str(7)):
                str_file_csv="quan_7_final.csv"
                str_quan_huyen="Quận 7, Thành Phố Hồ Chí Minh"
            if(ma_quanhuyen==str(9)):
                str_file_csv="quan_9_final.csv"
                str_quan_huyen="Quận 9, Thành Phố Hồ Chí Minh"
            if(ma_quanhuyen==str(10)):
                str_file_csv="quan_10_final.csv"
                str_quan_huyen="Quận 10, Thành Phố Hồ Chí Minh"
            if(ma_quanhuyen==str(12)):
                str_file_csv="quan_2_final.csv"
                str_quan_huyen="Quận 12, Thành Phố Hồ Chí Minh"                
            if(ma_quanhuyen==str(13)):
                str_file_csv="quan_binhthanh_final.csv"
                str_quan_huyen="Quận Bình Thạnh, Thành Phố Hồ Chí Minh"  
            if(ma_quanhuyen==str(14)):
                str_file_csv="quan_phunhuan_final.csv"
                str_quan_huyen="Quận Phú Nhuận, Thành Phố Hồ Chí Minh" 
            if(ma_quanhuyen==str(15)):
                str_file_csv="quan_thuduc_final.csv"
                str_quan_huyen="Quận Thủ Đức, Thành Phố Hồ Chí Minh" 
                
        df=pd.read_csv(str_file_csv,  delimiter=',')        
        #train_data = df
        data_du_lieu = df.drop(['median'], axis = 1)
        #data_du_lieu_du_bao = df.iloc[-12:]
        data2 = data_du_lieu['mean']/1000000
        data= list(data2)
        list_thoigian = list()
        #print(data)
        for index, row in df.iterrows():
            thang_ht = pd.to_datetime(row['time_1'])  
            #thang_ht = "1/1/2024"
            thang_ht = thang_ht.strftime('%m/%d/%Y')
            list_thoigian.append(thang_ht)
        labels = list_thoigian
        #term_day='1/1/2024'
        term_day=list_thoigian[len(list_thoigian)-1]
        #term_day=list_thoigian[len(list_thoigian)-1]
        #term_day=term_day.strftime('%m/%d/%Y')
        term_day = pd.to_datetime(term_day, format='%m/%d/%Y')
        #term_day=pd.to_datetime(term_day, unit='s')
        #term_day = datetime.datetime.strptime(term_day,'%m/%d/%Y')
        
        #for x in range(sothangdubao):
        #        term_day = term_day + relativedelta(months=+1)
        #        term_day_2 =term_day.strftime('%m/%Y')
        #        list_dudoan.append(term_day_2) 
                
        
        X=data
        history = X
        
        #LSTM dự đoán giá theo chuổi thời gian        
        scaler = MinMaxScaler()
        #test_data_lstm=data.iloc[:, 0:1].values
        test_data_lstm = pd.DataFrame(X)
        test_data_lstm = test_data_lstm.iloc[-12:]        
        #test_data_lstm = test_data_lstm.iloc[:, 0:1].values
        input_variables = scaler.fit_transform(test_data_lstm)
        predictions = model.predict(input_variables)[0]
        lstm_predictions_array_2d = predictions.reshape(-1, 1)
        #print(lstm_predictions_array_2d)
        lstm_predictions = scaler.inverse_transform(lstm_predictions_array_2d)
        array_2d = lstm_predictions.reshape(-1, 1)
        print(array_2d)
        #print(input_variables)
        
        #end LSTM dự đoán giá theo chuổi thời gian
        #best_model = auto_arima(data, start_p=1, start_q=1,
        #            test='adf',       # sử dụng Augmented Dickey-Fuller test để xác định d
        #            max_p=15, max_q=15, # giới hạn tối đa của p và q
        #            m=1,              # tần suất chuỗi thời gian (m=1 nếu không phải chuỗi mùa vụ)
        #            d=None,           # để auto_arima tự động xác định d
        #            seasonal=False,   # không sử dụng mô hình mùa vụ
        #            start_P=0, 
        #            D=0, 
        #            trace=True,
        #            error_action='ignore',  
        #            suppress_warnings=True, 
        #           stepwise=True)
        #output1=best_model.predict(n_periods=sothangdubao)
        #best_model.forecast(12)
        gia_du_doan=str(array_2d)
        #gia_du_doan=str(output1)
        data_dudoan = list(array_2d)
        
    return render_template('main.html',
        data=data,
        labels=labels,
        data_dudoan=data_dudoan,
        list_dudoan=list_dudoan,
        str_quan_huyen=str_quan_huyen,
        gia_du_doan=gia_du_doan
        
    )
 
# Main Driver Function 
if __name__ == '__main__':
    # Run the application on the local development server ##
    app.run(debug=True)