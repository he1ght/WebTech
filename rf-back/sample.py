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
    {"img" : "https://shopping-phinf.pstatic.net/main_8046667/80466670839.1.jpg", "result" : True} for _ in range(100)
  ]
  
  return a;