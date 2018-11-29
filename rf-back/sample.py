def detectObj() :
  info = [
        {"index" : 0, "label" : "fork"}, 
        {"index" : 1, "label" : "fork"}, 
        {"index" : 2, "label" : "vase"}, 
        {"index" : 3, "label" : "diningtable"}, 
        {"index" : 4, "label" : "chair"}, 
  ]

  return info


def img2vec(path, list, a=0.7):
  a = [
    {"img" : "https://shopping-phinf.pstatic.net/main_8046667/80466670839.1.jpg", "result" : True, "hp" : "10000","lp":"5000"} for _ in range(100)
  ]  
  return a;


def autoRecommend(insert) :
  result = [{'sresult': [], 'idx': 0, 'label': 'fork', 'is_check': True}, {'sresult': [{'img': 'https://shopping-phinf.pstatic.net/main_8113215/81132157533.11.jpg', 'sim': 0.7014142, 'result': True, 'lp': '26000', 'hp': '0', 'link': 'http://search.naver.com', 'title': "Naver Open API - shop ::'fork'"}, {'img': 'https://shopping-phinf.pstatic.net/main_1605475/16054755744.jpg', 'sim': 0.7561141, 'result': True, 'lp': '50666', 'hp': '0', 'link': 'http://search.shopping.naver.com/gate.nhn?id=13030622364', 'title': '(우와몰) 스테인레스 커트러리 양식기 세트'}, {'img': 'https://shopping-phinf.pstatic.net/main_1568050/15680508720.jpg', 'sim': 0.7042755, 'result': True, 'lp': '35090', 'hp': '0', 'link': 'http://search.shopping.naver.com/gate.nhn?id=16054755744', 'title': '큐티폴 고아 디너 4종세트(스푼+<b>포크</b>+나이프+젓가락)'}, {'img': 'https://shopping-phinf.pstatic.net/main_1599524/15995246582.jpg', 'sim': 0.7546325, 'result': True, 'lp': '49050', 'hp': '0', 'link': 'http://search.shopping.naver.com/gate.nhn?id=15680508720', 'title': '큐티폴 고아 블랙 디저트 <b>포크</b> 4P 세트'}, {'img': 'https://shopping-phinf.pstatic.net/main_1605493/16054936267.jpg', 'sim': 0.7561141, 'result': True, 'lp': '49049', 'hp': '0', 'link': 'http://search.shopping.naver.com/gate.nhn?id=15559839198', 'title': '[큐티폴]고아 과일(오리엔탈) <b>포크</b>(4P)'}], 'idx': 1, 'label': 'fork', 'is_check': True}, {'sresult': [{'img': 'https://shopping-phinf.pstatic.net/main_1594822/15948224346.jpg', 'sim': 0.72185767, 'result': True, 'lp': '33920', 'hp': '0', 'link': 'http://search.shopping.naver.com/gate.nhn?id=15989061395', 'title': 'Yosoo 68051 Heavy Duty Aluminium Alloy Rotating Bearing Turntable Round Dining Table Smooth Swivel P'}], 'idx': 2, 'label': 'diningtable', 'is_check': True}, {'sresult': [], 'idx': 3, 'label': 'vase', 'is_check': True}, {'sresult': [], 'idx': 4, 'label': 'vase', 'is_check': True}, {'sresult': [], 'idx': 5, 'label': 'knife', 'is_check': True}] 
  return result
