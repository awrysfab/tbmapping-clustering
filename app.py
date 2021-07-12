import matplotlib.pyplot as plt

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import mysql.connector as mysql
import csv
import json
import sys
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler


hostname = 'localhost'
username = 'root'
password = ''
database = 'tbmapping'
app = Flask(__name__)
CORS(app)
app.config["DEBUG"] = True

def get_id_kecamatan():
    myConnection = mysql.connect(host=hostname, user=username, password=password, database=database)
    cur = myConnection.cursor()
    sql = "SELECT id FROM subdistricts" + " ORDER BY id ASC;"
    cur.execute(sql)
    data = cur.fetchall()
    myConnection.close()
    return data

def get_namakecamatan():
    myConnection = mysql.connect(host=hostname, user=username, password=password, database=database)
    cur = myConnection.cursor()
    sql = "SELECT name FROM subdistricts" + " ORDER BY id ASC;"
    cur.execute(sql)
    data = cur.fetchall()
    myConnection.close()
    return data

def tupple_to_list(data):
    tmp_data = []
    for item in data:
        tmp_it = []
        if len(item) > 1:
            for i in item:
                tmp_it.append(i)
            tmp_data.append(tmp_it)
        elif len(item) == 1:
            tmp_data.append(item[0])
    return tmp_data

def get_data(table, field:tuple, tahun):
    myConnection = mysql.connect(host=hostname, user=username, password=password, database=database)
    list_id_kecamatan = tupple_to_list(get_id_kecamatan())
    data_rasio = np.zeros((20,4))
    for (ds, c_row) in zip(list_id_kecamatan, range(0, 20)):
        sql = "SELECT "
        for item in field:
            sql += item + ","
        sql = sql[:-1] + " FROM " + table + " WHERE subdistrict_id=" + str(ds) + " AND year=" + str(tahun) + " ORDER BY year ASC;"
        cur = myConnection.cursor()
        cur.execute(sql)
        data = cur.fetchall()
        col_data = data[0][0]
        if col_data == None:
            col_data = 0
        data_rasio[c_row] = (ds, col_data, 0, 0)
    myConnection.close()
    return data_rasio


def get_status(item):
    status = ""
    if item == 0:
        status = "Sedang"
    if item == 1:
        status = "Rendah"
    if item == 2:
        status = "Tinggi"
    return status

@app.route('/data/<int:tahun>')
def hitung_per_tahun(tahun):
    case = get_data("cluster_attributes", ("`case`",), tahun)
    target_case = get_data("cluster_attributes", ("target_case",), tahun)
    death_rate = get_data("cluster_attributes", ("death_rate",), tahun)
    density = get_data("cluster_attributes", ("density",), tahun)
    df1 = pd.DataFrame(case)
    df2 = pd.DataFrame(target_case)
    df3 = pd.DataFrame(death_rate)
    df4 = pd.DataFrame(density)

    #Create file csv
    data = pd.concat([
      df1.loc[:, 1] ,
      df2.loc[:,1], 
      df3.loc[:, 1], 
      df4.loc[:, 1]
      ], 
      axis=1)
    X = data.iloc[:,[0,1,2,3]].values

    #Normalisasi data ke skala 0-1
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(X)
    print(x_scaled)

    #Proses Kmeans
    kmeans=KMeans(n_clusters= 3, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0).fit(x_scaled)
    # centers = kmeans.cluster_centers_
    labels = kmeans.predict(x_scaled)

    print(labels)

    hasil_cluster = labels.tolist()
    list_nama_kecamatan = tupple_to_list(get_namakecamatan()) 
    list_jml_case = pd.to_numeric(data.iloc[:,0].values) 
    list_jml_target_case = pd.to_numeric(data.iloc[:,1].values) 
    list_jml_death_rate = pd.to_numeric(data.iloc[:,2].values) 
    list_jml_density = pd.to_numeric(data.iloc[:,3].values)

    result = []
    stts = ""
    for (
      item, 
      jml_case, 
      jml_target_case, 
      jml_death_rate, 
      jml_density, 
      cluster 
      ) in zip(
        list_nama_kecamatan, 
        list_jml_case, 
        list_jml_target_case, 
        list_jml_death_rate, 
        list_jml_density, 
        hasil_cluster 
        ):
        status = get_status(cluster)
        result.append(
          {
            "kecamatan": item, 
            "jml_case":jml_case, 
            "jml_target_case":jml_target_case, 
            "jml_death_rate":jml_death_rate, 
            "jml_density":jml_density, 
            "cluster": status
          })
    
    return json.dumps(result)

if __name__ == '__main__':
      app.run(debug='true' )
