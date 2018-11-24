import os
import sys
import urllib.request
import xml.etree.ElementTree as ET
from flask_cors import CORS
from flask import Flask, render_template, request, Response, redirect

app = Flask(__name__)
CORS(app)

sys.path.insert(0, '../Classification/yolo.v3.pytorch/')
import detector


UPLOAD_FOLDER = os.path.basename('uploads')
BUDGET = 0
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
fileChanged = 0 # for observe file change


#기본 메인 url
@app.route('/')
def hello_world():
    return render_template('index.html')

def detect_from_image():
    detector.detect_object(image_folder="uploads", config_path="../Classification/yolo.v3.pytorch/config/yolov3.cfg",
                           weights_path="../Classification/yolo.v3.pytorch/weights/yolov3.weights",
                           class_path="../Classification/yolo.v3.pytorch/data/coco.names")


#예산, 이미지 제출 post
@app.route('/', methods=['POST'])
def upload_file():
    file = request.files['image']    
    # 확장자 명 추출
    # extension = file.filename.rsplit('.', 1)[1].lower()
    f = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded.jpg')        
    file.save(f)
    detect_from_image()
    return render_template('index.html')




# naver api 예시로 사용
@app.route("/apiTest")
def apiTest() :
   return render_template('naver.html')
    

# naver api post, xml 파일 만들어주자
@app.route('/apiTest', methods=['POST'])
def apiTestPost() : 
  # 예산 설정 전역변수
  term = request.form['term']

  client_id = "nFWOo7gr6A0C4umAP_Ln"
  client_secret = "7UwjXua3oQ"

  encText = urllib.parse.quote(term)
  url = "https://openapi.naver.com/v1/search/shop.xml?display=10&query=" + encText
  
  request2 = urllib.request.Request(url)
  request2.add_header("X-Naver-Client-Id",client_id)
  request2.add_header("X-Naver-Client-Secret",client_secret)
  response = urllib.request.urlopen(request2)
  rescode = response.getcode()

  if(rescode==200):
      # print(response.read())
      # return response.read()
      response_body = response.read()
      
      xmlData = Response(response_body.decode('utf-8'), mimetype='text/xml')
      tree = ET.XML(response_body.decode('utf-8'))
      
      with open("static/products.xml", "wb") as f:
        f.write(ET.tostring(tree))

      return redirect('/', code=302) 
  else:
      print("Error Code:" + rescode)

if __name__ == '__main__' :
  app.run(debug = True, host="0.0.0.0", port=int(os.getenv('VCAP_APP_PORT', '10000')))













