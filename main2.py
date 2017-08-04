# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# [START app]
import numpy as np 
import tensorflow as tf
import os
import ast
import csv

import logging
import base64
import oauth2client.service_account
from httplib2 import Http
import json
from string import Template
# from google.cloud import storage
from copy import deepcopy
from tensorflow.python.lib.io import file_io

from flask import Flask, render_template
from bokeh.models import (HoverTool, FactorRange, Plot, 
                          LinearAxis, Grid, Range1d, LabelSet)
from bokeh.models.glyphs import VBar
from bokeh.plotting import figure
from bokeh.charts import Bar
from bokeh.embed import components
from bokeh.models.sources import ColumnDataSource

# flask constructors
app = Flask(__name__)

# gcloud constructors
# client = storage.Client(project='qwiklabs-gcp-dbd8214a842cac50')
# bucket = client.get_bucket('youtube8m-ml-us-east1')
# validate_blobs = list(bucket.list_blobs(prefix='1/video_level/validate/'))

bucket_string = "youtube8m-ml-us-east1"
video_bucket = "gs://youtube8m-ml-us-east1/1/video_level/validate"
validate_records = file_io.list_directory(video_bucket)

@app.route('/')
def hello():
    """Return a friendly HTTP greeting."""
    return '''
    <h1>Bose Best Website</h1>

    <iframe width="560" height="315" src="https://www.youtube.com/embed/iVecSyZJbUU" frameborder="0" allowfullscreen></iframe>'''

@app.route('/videos/<vid>')
def videos(vid):
    vidtemplate = Template("""
      <h2>
        YouTube video link:
        <a href="https://www.youtube.com/watch?v=${youtube_id}">
          ${youtube_id}
        </a>
      </h2>

      <iframe src="https://www.youtube.com/embed/${youtube_id}" width="853" height="480" frameborder="0" allowfullscreen></iframe>
    """)

    return vidtemplate.substitute(youtube_id=vid)

@app.route('/random/')
def random():
    model_name = 'test_Moe'
    model_version = 'v1'
    with open('qwiklabs-gcp-dbd8214a842cac50-317227781517.json') as data_file:
          key = json.load(data_file)
    api = 'https://ml.googleapis.com/v1/projects/{project}/models/{model}/versions/{version}:predict'
    url = api.format(project='qwiklabs-gcp-dbd8214a842cac50',
                 model=model_name,
                 version=model_version)
    headers = {'Content-Type': 'application/json'}

    credentials = oauth2client.service_account.ServiceAccountCredentials.from_json_keyfile_dict(
            keyfile_dict = key, 
            scopes=['https://www.googleapis.com/auth/cloud-platform'])

    http_auth = credentials.authorize(Http(timeout=30))

    # video_folder = "gs://youtube8m-ml-us-east1/1/video_level/validate"
    # video_lvl_records = [i for i in os.listdir(video_folder) if i.endswith("tfrecord")]
    # video_lvl_record = os.path.join(video_folder, np.random.choice(video_lvl_records))
    
    # example_blob = np.random.choice(validate_blobs[1:])
    # import ipdb; ipdb.set_trace()
    # file_name = example_blob.name[example_blob.name.index('validate/')+9:]
    file_name = np.random.choice(validate_records)
    full_record_path = os.path.join(video_bucket,file_name).encode('ascii')
    # example_blob_string = example_blob.download_as_string(client)

    #example = np.random.choice(list(tf.python_io.tf_record_iterator(video_lvl_record)))
    example = np.random.choice(list(tf.python_io.tf_record_iterator(full_record_path)))

    # tf_example = tf.train.Example.FromString(example)
    # video_url = tf_example.features.feature['video_id']
    example = base64.b64encode(example)

    # in order to do inference in the cloud you have to do a base64
    body = {'instances': [{"b64": example}]}
    http_body = json.dumps(body, sort_keys=True)
    response, response_body = http_auth.request(uri=url, method='POST', body=http_body, headers=headers)
    response_body_dict = ast.literal_eval(response_body)

    pred_dict = response_body_dict['predictions'][0]
    video_id = pred_dict['video_id']
    predictions = pred_dict['predictions']
    class_indeces = pred_dict['class_indexes']

    # create chart
    plot = create_bar_chart(data = pred_dict, 
                            title = "Top K Probabilities", 
                            x_name = "class_indexes",
                            y_name = "predictions", 
                            hover_tool = None)
    script, div = components(plot)

    return render_template("chart.html", bars_count=len(predictions),
                           youtube_id = video_id, the_div=div, the_script=script)


@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500

'''
Returns a Dictionary storing information by video ID.
Ex:    ID = 123
       name = video_info[ID]['Name']          # Returns the name of the video
       url = video_info[ID]['WikiUrl']        # Returns the wikipedia URL
       category = video_info[ID]['Vertical1'] # Returns the first vertical (category)
'''
def video_info(csv_path):

    video_info = {}
    info_cols = ['Index', 'Name', 'WikiUrl', 'Vertical1']

    with open(csv_path) as csvfile:
        csvinfo = csv.reader(csvfile, delimiter=',')
        cols = csvinfo.next()
        indices = [cols.index(col) for col in info_cols]

        for row in csvinfo:
            vid_id = int(row[indices[0]])
            video_info[vid_id] = {}
            for i, info_col in enumerate(info_cols[1:]):
                video_info[vid_id][info_col] = row[indices[i+1]]
                
    return video_info

def sort_dict(d, x_name, y_name, topK=5):
    # import ipdb; ipdb.set_trace()
    zipped = zip(d[x_name], d[y_name])
    zipped.sort(key=lambda x: x[1], reverse=True)
    VIDEO_DECODER = video_info('vocabulary.csv')
    named = [(VIDEO_DECODER[id[0]]['Name'], id[1]) for i, id in enumerate(zipped)]
    # import ipdb; ipdb.set_trace()

    sorted_dict = {x_name: [i[0] for i in named[:topK]], y_name: [i[1] for i in named[:topK]]}
    return sorted_dict

def create_bar_chart(data, title, x_name, y_name, hover_tool=None,
                     width=100, height=100):
    """Creates a bar chart plot with the exact styling for the centcom
       dashboard. Pass in data as a dictionary, desired plot title,
       name of x axis, y axis and the hover tool HTML.
    """
    # import ipdb; ipdb.set_trace()
    vid = deepcopy(data)
    video_id = vid.pop('video_id', None)
    data_sorted = sort_dict(vid, x_name, y_name)
    title = "ID " + video_id + " " + title

    source = ColumnDataSource(data_sorted)
    xdr = FactorRange(factors=[str(i) for i in data_sorted[x_name]])
    ydr = Range1d(start=0,end=1.01)

    tools = []
    if hover_tool:
        tools = [hover_tool,]

    plot = figure(title= title, 
                  x_range=xdr, y_range=ydr, plot_width=width,
                  plot_height=height, h_symmetry=False, v_symmetry=False,
                  min_border=0, toolbar_location="above", tools=tools,
                  responsive=True, outline_line_color="#666666")

    glyph = VBar(x=x_name, top=y_name, bottom=0, width=.8,
                 fill_color="#e12127")
    plot.add_glyph(source, glyph)

    # labels = LabelSet(x=xdr, y=ydr, text=xdr, level='glyph', source=source, render_mode='canvas')
    # plot.add_layout(labels)

    xaxis = LinearAxis()
    yaxis = LinearAxis()

    plot.add_layout(Grid(dimension=0, ticker=xaxis.ticker))
    plot.add_layout(Grid(dimension=1, ticker=yaxis.ticker))
    plot.toolbar.logo = None
    plot.min_border_top = 0
    plot.xgrid.grid_line_color = None
    plot.ygrid.grid_line_color = "#999999"
    plot.yaxis.axis_label = "Probability"
    plot.ygrid.grid_line_alpha = 0.1
    plot.xaxis.axis_label = "Class"
    plot.xaxis.major_label_orientation = 1
    return plot


if __name__ == '__main__':
    # This is used when running locally. Gunicorn is used to run the
    # application on Google App Engine. See entrypoint in app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)
# [END app]
