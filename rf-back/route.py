import os
import sys
import urllib.request
import xml.etree.ElementTree as ET
from flask_cors import CORS
from flask import Flask, render_template, request, Response, redirect

sys.path.insert(0, "../Classification/")
import detector
sys.path.insert(0, '../Similarity/')
import check_similarity

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = os.path.basename('uploads')
BUDGET = 0
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
fileChanged = 0 # for observe file change

client_id = "nFWOo7gr6A0C4umAP_Ln"
client_secret = "7UwjXua3oQ"


def naverSearch(term, index):  
  encText = urllib.parse.quote(term)
  url = "https://openapi.naver.com/v1/search/shop.xml?display=10&query=" + encText
  
  request2 = urllib.request.Request(url)
  request2.add_header("X-Naver-Client-Id",client_id)
  request2.add_header("X-Naver-Client-Secret",client_secret)
  response = urllib.request.urlopen(request2)
  rescode = response.getcode()
  if(rescode==200):
      response_body = response.read()
      
      xmlData = Response(response_body.decode('utf-8'), mimetype='text/xml')
      tree = ET.XML(response_body.decode('utf-8'))
      
      with open(f"static/products{index}.xml", "wb") as f:
        f.write(ET.tostring(tree))
      return 0
  else:
      print("Error Code:" + rescode)    
      return 1

def test():    
    info = [
        {"index" : 0, "label" : "fork"}, 
        {"index" : 1, "label" : "fork"}, 
        {"index" : 2, "label" : "vase"}, 
        {"index" : 3, "label" : "diningtable"}, 
        {"index" : 4, "label" : "chair"}, 
    ]

    lists = []

    for i in info:
        suc = naverSearch(i['label'], i["index"])
        if suc == 0:
            print('success')
        else : print("Error handling")
                      
    
    for i in info:
        lists = []
        doc = ET.parse(f"static/products{i['index']}.xml")
        root = doc.getroot()
        for a in root.iter('image'):
            lists.append(a.text)
        print(lists)



        print('--------------')



#기본 메인 url
@app.route('/')
def hello_world():
    # test()
    return render_template('index.html')

#예산, 이미지 제출 post
@app.route('/', methods=['POST'])
def upload_file():
    file = request.files['image']    
    # 확장자 명 추출
    # extension = file.filename.rsplit('.', 1)[1].lower()
    f = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded.jpg')        
    file.save(f)
    print("ready to detect")
    info = detector.detect_object(image_folder="uploads/", output="static/output",
                                   config_path="../Classification/config/yolov3.cfg",
                                   weights_path="../Classification/weights/yolov3.weights",
                                   class_path="../Classification/data/coco.names")
    print(info)
    #classification 함수 사용! 해서 output폴더로 해서 나오고 dictionary 받음

    for i in info:
        suc = naverSearch(i['label'], i["index"])
        if suc == 0:
            print('success')
        else:
            print("Error handling")

    for i in info:
        lists = []
        doc = ET.parse(f"static/products{i['index']}.xml")
        root = doc.getroot()
        for a in root.iter('image'):
            lists.append(a.text)
        # check_similarity.img_sim()
            
    # idx = 0
    # url = f'http://175.193.43.115:10000/static/output/{idx}.png'
    return render_template('index.html')




# naver api 예시로 사용
@app.route("/apiTest")
def apiTest() :
   return render_template('naver.html')
    

# naver api post, xml 파일 만들어주자
@app.route('/apiTest', methods=['POST'])
def apiTestPost() : 
  # 예산 설정 전역변수
    result = naverSearch('chair')
    print(result)
    if result == 0 :
        return "success"
    else :
        return "fail"



    

if __name__ == '__main__' :
  app.run(debug = True, host="0.0.0.0", port=int(os.getenv('VCAP_APP_PORT', '10000')))