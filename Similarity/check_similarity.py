import sys
import os

from img_to_vec import Img2Vec
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import torch
import requests


def img_sim(input_path, target_list, threshold=0.7, use_gpu=True):
    img2vec = Img2Vec(model='resnet-101', cuda=use_gpu)
    try:
        key_vec = img2vec.get_vec(Image.open(input_path))
    except OSError:
        key_vec = img2vec.get_vec(Image.open(requests.get(input_path, stream=True).raw))
    pics = {}
    for file in target_list:
        # filename = os.fsdecode(file)
        try:
            img = Image.open(file)
        except OSError:
            img = Image.open(requests.get(file, stream=True).raw)
        vec = img2vec.get_vec(img)
        pics[file] = vec
    result = []
    try:
        sims = {}
        for key in list(pics.keys()):
            sims[key] = cosine_similarity(key_vec.reshape((1, -1)), pics[key].reshape((1, -1)))[0][0]

        d_view = [(v, k) for k, v in sims.items()]
        # d_view.sort(reverse=True)
        for v, k in d_view:
            # print("{:10s} : {} [{}]".format(k, 'O' if v >= threshold else 'X', v))
            result.append({'img': k, 'sim': v, 'result': v >= threshold})

    except KeyError as e:
        print('Could not find filename %s' % e)

    except Exception as e:
        print(e)
    return result


if __name__ == "__main__":
    result = img_sim(input_path="https://shopping-phinf.pstatic.net/main_1547247/15472472225.20180927155901.jpg",
                     target_list=["./example/test/sim_test1.JPG", "./example/test/sim_test2.jpg", "./example/test/sim_test3.jpeg"])
    print(result)
