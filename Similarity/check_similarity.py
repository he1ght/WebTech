import sys
import os

from img_to_vec import Img2Vec
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import torch
import requests


def img_sim(input_path, target_list, threshold=0.7, use_gpu=True, prune=True):
    img2vec = Img2Vec(model='resnet-101', cuda=use_gpu)
    try:
        key_img = Image.open(input_path)
    except OSError:
        key_img = Image.open(requests.get(input_path, stream=True).raw)
    key_img = key_img.convert("RGB")
    key_vec = img2vec.get_vec(key_img)
    pics = {}
    for file in target_list:
        # filename = os.fsdecode(file)
        try:
            img = Image.open(file['img'])
        except OSError:
            img = Image.open(requests.get(file['img'], stream=True).raw)
        img = img.convert("RGB")
        vec = img2vec.get_vec(img)
        pics[file['img']] = vec
    result = []
    try:
        sims = {}
        for key in list(pics.keys()):
            sims[key] = cosine_similarity(key_vec.reshape((1, -1)), pics[key].reshape((1, -1)))[0][0]

        d_view = [(v, k) for k, v in sims.items()]
        # d_view.sort(reverse=True)
        for (v, k), file in zip(d_view, target_list):
            if prune and not v >= threshold:
                continue
            # print("{:10s} : {} [{}]".format(k, 'O' if v >= threshold else 'X', v))
            assert k == file['img']
            result.append({'img': k,
                           'sim': v,
                           'result': 1 if v >= threshold else 0,
                           'lp': file['lp'],
                           'hp': file['hp'],
                           'link': file['link'],
                           'title': file['title']
                           })

    except KeyError as e:
        print('Could not find filename %s' % e)

    except Exception as e:
        print(e)
    return result


if __name__ == "__main__":
    result = img_sim(input_path="https://shopping-phinf.pstatic.net/main_1547247/15472472225.20180927155901.jpg",
                     target_list=[{'img': "./example/test/sim_test1.JPG", 'lp': 5, 'hp': 0, 'link': ".", 'title': "1"},
                                  {'img': "./example/test/sim_test2.jpg", 'lp': 0, 'hp': 100, 'link': ".", 'title': "1"},
                                  {'img': "./example/test/sim_test3.jpeg", 'lp': 10, 'hp': 300, 'link': ".", 'title': "1"}])
    print(result)
