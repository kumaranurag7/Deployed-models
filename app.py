from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
#from sklearn.preprocessing import StandardScaler as ss

le1 = pickle.load(open('le1.pkl', 'rb'))
le2 = pickle.load(open('le2.pkl', 'rb'))
le3 = pickle.load(open('le3.pkl', 'rb'))
ss = pickle.load(open('ss.pkl', 'rb'))
xgb = pickle.load(open('xgb.pkl', 'rb'))
lat = pd.read_pickle("lat.pkl")
lon = pd.read_pickle("lon.pkl")

app = Flask(__name__)

@app.route('/')
def man():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def home():
    min_nights = request.form['min_nights']
    nreviews = request.form['no_reviews']
    host_listings = request.form['host_listings']
    availability = request.form['availability']
    room_type = request.form['type']
    borough = request.form['borough']
    neighbourhood = (request.form['neighbourhood'])
    
    lat1 = 40.73
    lon1 = -73.95
    
    #treat skew
    min_nights = np.log1p(int(min_nights))
    nreviews = np.log1p(int(nreviews))
    host_listings = np.log1p(int(host_listings))
    availability = np.log1p(int(availability))
    
    # prepare a df
    pred_df=pd.DataFrame({'neighbourhood_group':borough, 'neighbourhood':neighbourhood,'latitude':lat1,
                            'longitude':lon1, 'room_type':room_type, 'minimum_nights':min_nights,
                            'number_of_reviews':nreviews, 'calculated_host_listings_count':host_listings,
                            'availability_365':availability}, index = [0])
    
    
    #le1
    pred_df.neighbourhood = le1.transform(pred_df.neighbourhood)

    #le2
    pred_df.neighbourhood_group = le2.transform(pred_df.neighbourhood_group)

    #le3
    pred_df.room_type = le3.transform(pred_df.room_type)
    print(pred_df.neighbourhood_group)
    print(pred_df.neighbourhood)
    print(pred_df.latitude)
    print(pred_df.longitude)
    print(pred_df.room_type)
    print(pred_df.minimum_nights)
    print(pred_df.number_of_reviews)
    print(pred_df.calculated_host_listings_count)
    print(pred_df.availability_365)
    # ss
    pred_df = ss.transform(pred_df)

    pred = round(float(np.expm1(xgb.predict(pred_df))),2)
    print(pred)
    return render_template('after.html', pred=(pred)) #, place = place.lower()

if __name__ == "__main__":
    app.run(debug=False)















