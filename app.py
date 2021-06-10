from flask import Flask, request
import time
import os
from inference import Inference
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
# CORS(app)
app.config.from_object(__name__)

model = Inference()
@app.route('/')
def index():
    return "index"


@app.route('/files', methods=["POST"])
def filesapi():
    if not os.path.isdir("userfiles/"):
        os.mkdir("userfiles/")
    myfile = request.files.get('file')
    if myfile == None:
        return {'states': 'error', 'msg': 'no file'}, 400
    fileName = str(time.time())+"_"+myfile.filename
    filepath =  "userfiles/"+fileName
    myfile.save(filepath)
    model.inference(filepath)
    os.remove(filepath)
    return {'filename': fileName, 'msg': 'success'}, 200
