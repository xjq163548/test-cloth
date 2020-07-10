#-*-coding:utf-8-*-
from __future__ import absolute_import, division, print_function, unicode_literals
#import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.pardir)
from PIL import Image
import numpy as np
from tensorflow import keras
import tensorflow as tf
import os
import time
import datetime as dt
from flask import Flask, jsonify, Response, request
from cassandra.cluster import Cluster
from judge.classify_api import classify
#from cassandra.query import SimpleStatement
import logging
log = logging.getLogger()
log.setLevel('INFO')
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
log.addHandler(handler)
img_folder="./result"

global fr
#cluster = Cluster(contact_points=['172.19.0.3'], port=9042) #容器里的ip（cassandra的ip）
#cluster = Cluster(contact_points=['127.0.0.1'], port=9042) #本地ip
cluster = Cluster(contact_points=['192.168.116.100'], port=9042) #电脑ip
#cluster = Cluster(contact_points=['172.17.0.5'], port=9042)

session = cluster.connect()
KEYSPACE = "mykeyspace"
app = Flask(__name__)



os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# TensorFlow and tf.keras
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def createKeySpace():
   log.info("Creating keyspace...")
   try:
       session.execute("""
           CREATE KEYSPACE IF NOT EXISTS %s
           WITH replication = { 'class': 'SimpleStrategy', 'replication_factor': '2' }
           """ % KEYSPACE)
       log.info("setting keyspace...")
       session.set_keyspace(KEYSPACE)
       log.info("creating table...")
       session.execute("""
       
           CREATE TABLE mytable (
               time text,
               name text,
               result text,
               PRIMARY KEY (name)
           )
           """)
   except Exception as e:
       log.error("Unable to create keyspace")
       log.error(e)

createKeySpace();

def insertdata(fr,perdict):

    try:
        n = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        session.execute("""
            INSERT INTO mytable (time, name, result)
            VALUES ('%s', '%s', '%s')
            """ % (n, fr, perdict))
        log.info("%s, %s, %s" % (n, fr, perdict))
        log.info("Data stored!")
        session.execute("""Select * from mytable;""")
    except Exception as e:
        log.error("Unable to insert data!")
        log.error(e)


def plot_image(i, predictions_array, img):
    predictions_array, img = predictions_array, img[i]
    #plt.grid(False)
    #plt.xticks([])
    #plt.yticks([])
   # plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    return predicted_label

def anal(fr):
    model = keras.models.load_model(
        './my_picmodel.h5')
    test_images = Image.open(fr)
    test_images = np.invert(test_images.convert('L'))
    test_images = test_images / 255.0

    probability_model = tf.keras.Sequential([model,
                                             tf.keras.layers.Softmax()])
    test_images = (np.expand_dims(test_images, 0))
    predictions_single = probability_model.predict(test_images)
    b = plot_image(0, predictions_single[0], test_images)
    return b

@app.route('/', methods=['GET' , 'POST'])
def index():
    return """
<h1>image choose API!</h1>
<form action="/classify" method="post" enctype="multipart/form-data">
    <div style="align-items: center">
        file:<br/>
        <input type='file' name='file'/>
        <br/>
        <input type="submit" value="submit">
    </div>
"""
@app.route('/classify', methods=['GET' , 'POST'])
def cls():
    file = request.files.get('file', None)
    if file:
        file.save("./result/test.jpg")
    pic = tf.io.read_file('./result/test.jpg')
    a=classify(pic)
    print(a)
    if(a == 0):
        context='t-shirt'
    elif(a==1):
        context = 'trousers'
    elif (a == 2):
        context = 'pullover'
    elif (a == 3):
        context = 'dress'
    elif (a == 4):
        context = 'coat'
    elif (a == 5):
        context = 'sandal'
    elif (a == 6):
        context = 'shirt'
    elif (a == 7):
        context = 'shoes'
    elif (a == 8):
        context = 'bag'
    else:
        context = 'boots'
    return context



@app.route('/file/<file>', methods=['GET'])
def anas(file):
    fr = os.path.join(img_folder,file)
    print('{}'.format(fr))
    perdict = anal(fr)
    print('perdict - {}'.format(perdict))
    insertdata(fr, perdict)
    print(type(perdict))

    return Response('Success!')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
#anas('cap.png')
