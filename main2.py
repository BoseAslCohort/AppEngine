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

import logging
from flask import Flask
import base64
import oauth2client.service_account
from httplib2 import Http
import json

from string import Template
import os
import ast

app = Flask(__name__)

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

    vidtemplate = Template("""
      <h2>
        Video ID:${video}\n
        Predictions:${preds}\n
        Class Indices:${classes}\n\n

        YouTube video link: \n 
        <a href="https://www.youtube.com/watch?v=${youtube_id}">
          ${youtube_id}
        </a>
      </h2>

      <iframe src="https://www.youtube.com/embed/${youtube_id}" width="853" height="480" frameborder="0" allowfullscreen></iframe>
    """)

    video_folder = "./video_data/"
    video_lvl_records = [i for i in os.listdir(video_folder) if i.endswith("tfrecord")]
    video_lvl_record = os.path.join(video_folder, np.random.choice(video_lvl_records))
    # frame_lvl_record = "./frame_data/traina0.tfrecord"

    vid_ids = []
    labels = []
    mean_rgb = []
    mean_audio = []

    # for example in tf.python_io.tf_record_iterator(video_lvl_record):
    #     tf_example = tf.train.Example.FromString(example)
    #     vid_ids.append(tf_example.features.feature['video_id'].bytes_list.value[0].decode(encoding='UTF-8'))
    #     labels.append(tf_example.features.feature['labels'].int64_list.value)
    #     mean_rgb.append(tf_example.features.feature['mean_rgb'].float_list.value)
    #     mean_audio.append(tf_example.features.feature['mean_audio'].float_list.value)
    
    # pull a random item from generator
    # iterate_until = np.random.randint(10)
    # for item_index, item in enumerate(tf.python_io.tf_record_iterator(video_lvl_record)):
    #   if item_index == iterate_until:
    #     example=item
    example = np.random.choice(list(tf.python_io.tf_record_iterator(video_lvl_record)))
    example = base64.b64encode(example)

    # in order to do inference in the cloud you have to do a base64
    body = {'instances': [{"b64": example}]}
    http_body = json.dumps(body, sort_keys=True)
    response, response_body = http_auth.request(
      uri=url, method='POST', body=http_body, headers=headers)
    response_body_dict = ast.literal_eval(response_body)

    video_id = response_body_dict['predictions'][0]['video_id']
    predictions = response_body_dict['predictions'][0]['predictions']
    class_indeces = response_body_dict['predictions'][0]['class_indexes']

    return vidtemplate.substitute(
      youtube_id=video_id, video = video_id, 
      preds = predictions, classes = class_indeces)


@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


if __name__ == '__main__':
    # This is used when running locally. Gunicorn is used to run the
    # application on Google App Engine. See entrypoint in app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)
# [END app]
