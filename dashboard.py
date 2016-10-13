#!/usr/bin/env python
'''Show streaming graph of stock.'''

from jinja2 import Template
from flask import Flask, jsonify
from six.moves.urllib.request import urlopen
from six.moves.urllib.parse import urlencode
import datetime
from collections import deque
from threading import Thread
from time import time, sleep
import csv
import codecs
import boto3
import json



STREAM = 'flappy-kinesis'


html = Template('''\
<!DOCTYPE html>
<html>
  <head>
    <title>Streaming Stocks</title>
    <style>
      #chart {
        min-height: 300px;
      }
    </style>
    <link
      rel="stylesheet"
      href="//maxcdn.bootstrapcdn.com/bootstrap/3.2.0/css/bootstrap.min.css">
  </head>
  <body>
    <div class="container">
    <h4 class="label label-primary">{{ stock }}</h4>
    <div id="chart"></div>
  </body>
  <script
    src="//ajax.googleapis.com/ajax/libs/jquery/1.11.1/jquery.min.js">
  </script>
  <script
    src="//cdnjs.cloudflare.com/ajax/libs/flot/0.8.2/jquery.flot.min.js">
  </script>
  <script
    src="//cdnjs.cloudflare.com/ajax/libs/flot/0.8.2/jquery.flot.time.min.js">
  </script>

  <script>
  var chart;

  function get_data() {
    $.ajax({
        url: '/data',
        type: 'GET',
        dataType: 'json',
        success: on_data
    });
  }

  function on_data(data) {
    chart.setData([{data: data.values}]);
    chart.setupGrid();
    chart.draw();

    setTimeout(get_data, 1000);
  }

  $(function() {
    chart = $.plot("#chart", [ ], {xaxis: {minTickSize: 1}});
    get_data();
  });

    </script>
</html>
''')

app = Flask(__name__)
stock = 'FlappyBird'
# In memory RRDB
values = deque(maxlen=1000)



def poll_data():
    global shard_it
    while True:
        ts = int(time()) - 4000
        shard_it = \
        client.get_shard_iterator(StreamName=STREAM, ShardId='shardId-000000000000', ShardIteratorType='AT_TIMESTAMP',
                                  Timestamp=ts)["ShardIterator"]
        output = client.get_records(ShardIterator=shard_it,Limit=10000)
        records = output["Records"]
        if len(records):
          record = records[-1]
          datum = json.loads(record["Data"])
          x = int(datum['trial'])
          y = float(datum['score'])
          values.append([x, y])

        sleep(3)








@app.route('/')
def home():
    return html.render(stock=stock)


@app.route('/data')
def data():
    # * 1000 to convert to javascript time
    print jsonify(values=[(trial,score) for trial, score in values])
    return jsonify(values=[(trial,score) for trial, score in values])



def main(argv=None):
    global stock

    import sys



    thr = Thread(target=poll_data)
    thr.daemon = True
    thr.start()

    stock = 'Flappy Bird'
    # debug will reload server on code changes
    # 0.0.0.0 means listen on all interfaces
    app.run(host='0.0.0.0', debug=False)


if __name__ == '__main__':
    main()