import sys
import os

from img_to_vec import Img2Vec
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import torch


def img_sim(input_path, target_list, threshold=0.7):
    img2vec = Img2Vec(model='resnet-101')
    key_vec = img2vec.get_vec(Image.open(input_path))
    pics = {}
    for file in target_list:
        # filename = os.fsdecode(file)
        img = Image.open(file)
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
            result.append({'img': k, 'result': v >= threshold})

    except KeyError as e:
        print('Could not find filename %s' % e)

    except Exception as e:
        print(e)
    return result


if __name__ == "__main__":
    result = img_sim(input_path="./example/test/sim_test1.JPG",
                     target_list=["./example/test/sim_test2.jpg", "./example/test/sim_test3.jpeg"])
    print(result)
