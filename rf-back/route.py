import os
import sys
import urllib.request
import xml.etree.ElementTree as ET
from flask_cors import CORS
from flask import Flask, render_template, request, Response, redirect

try:
    from ai_funcs import detectObj, img2vec
except ImportError:
    from sample import detectObj, img2vec


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



#기본 메인 url
@app.route('/')
def hello_world():
    return render_template('index.html')


#예산, 이미지 제출 post
@app.route('/', methods=['POST'])
def upload_file():
    file = request.files['image']        
    f = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded.jpg')        
    file.save(f)
    
    # import 설정으로 info 사용(협업->딥러닝, 개인-> 샘플 info)
    info = detectObj()
    lists = []

    for i in info:
        suc = naverSearch(i['label'], i["index"])
        if suc == 0:
            print(f'products{i["index"]}.xml made successfully!')
        else : print("Error handling")
                      
    # i["lavel"] = fork
    result = []
    for i in info:
        lists = []
        doc = ET.parse(f"static/products{i['index']}.xml")
        root = doc.getroot()
        for img,lp,hp,link,title  in zip(root.iter('image'),root.iter('lprice'),root.iter('hprice'),root.iter('link'), root.iter('title')):            
            q = {}
            q["img"] = img.text
            q["lp"] = lp.text
            q["hp"] = hp.text     
            q["link"] = link.text       
            q["title"] = title.text
            lists.append(q)
                    
        
        #print('--------------------------------------')

        # image similarity check start
        sresult = img2vec(f'./output/{i["index"]}.jpg',lists, 0.7)  #list안에 dic img : 이미지 주소랑 result : true/false        
        result.append({'sresult': sresult, 'idx': i['index'], 'label':i['label'], 'is_check':True})
        #print(sresult)

    print(result)    

        #여기서 similarity 검사        





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