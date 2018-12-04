import sys
import os

sys.path.append("..")  # Adds higher directory to python modules path.
from img_to_vec import Img2Vec
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import torch


# input_path = './test_images'
input_path = sys.argv[1]

img2vec = Img2Vec(model='resnet-101')

# For each test image, we store the filename and vector as key, value in a dictionary
pics = {}
for file in os.listdir(input_path):
    filename = os.fsdecode(file)
    img = Image.open(os.path.join(input_path, filename))
    vec = img2vec.get_vec(img)
    pics[filename] = vec

pic_name = ""
while pic_name != "exit":
    pic_name = str(input("Input file name : "))
    # pic_name = 'cat.jpg'

    try:
        sims = {}
        for key in list(pics.keys()):
            if key == pic_name:
                continue

            sims[key] = cosine_similarity(pics[pic_name].reshape((1, -1)), pics[key].reshape((1, -1)))[0][0]

        d_view = [(v, k) for k, v in sims.items()]
        d_view.sort(reverse=True)
        print("\nPrediction ...")
        for v, k in d_view:
            print("{:10s} : {} [{}]".format(k, 'O' if v >= 0.6 else 'X',  v))

    except KeyError as e:
        print('Could not find filename %s' % e)

    except Exception as e:
        print(e)
