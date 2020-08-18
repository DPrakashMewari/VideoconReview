# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 11:04:56 2020

@author: hp
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 16:07:32 2020

@author: hp
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 12:43:50 2020

@author: hp
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 17:19:54 2020

@author: hp
"""
import pickle
from flask import Flask,jsonify,request,render_template
import pandas as pd
from recommender_engine import CosineSimilarity
app = Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')



@app.route("/submit", methods=['GET','POST'])    
def predict():
    if(request.method == 'POST'):
        with open('amazonReviews.pickle', 'rb') as f:
            dt = pickle.load(f)
        
        keywords = request.values['comments']
        
        dt['CosineScore'] = 0
        for index, row in dt.iterrows():
            value = CosineSimilarity.cosine_similarity_of(row['processed'], keywords)
            dt.iloc[index,-1] = value 
        outputDataFrame = dt.sort_values('CosineScore',ascending = False).head(10).drop('processed',axis = 1)
        
        if(outputDataFrame.size!=0):
             return  render_template('index1.html',  tables=[outputDataFrame.to_html(classes='data', header="true")])
        else:
             return  render_template('index1.html')
             
               

        
    else:
        s='Please Try Again Later'
        return jsonify(s)
if __name__ == '__main__':
    app.run()