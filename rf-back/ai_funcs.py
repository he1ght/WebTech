import sys

popular_label = {
    'tvmonitor': 'monitor',
    'diningtable': 'dining table'
}
fixed_label = {
    'chair': '거실 의자',
    'dining table': '식탁',
    'vase': '화병',
    'microwave': '전자레인지',
    'sink': '싱크대',
    'cup': '유리컵',
    'pottedplant': '화초'
}

sys.path.insert(0, "../Classification/")
import detector

sys.path.insert(0, '../Similarity/')
import check_similarity


def detectObj(path_id = ""):
    return detector.detect_object(image_folder="uploads/", output="static/output",
                                  config_path="../Classification/config/yolov3.cfg",
                                  weights_path="../Classification/weights/yolov3.weights",
                                  class_path="../Classification/data/coco.names",
                                  path_id=path_id)


def img2vec(input_path, target_list, threshold=0.7):
    return check_similarity.img_sim(input_path, target_list, threshold=threshold)


def autoRecommend(result, budget=0):
    # todo return 값 reuslt ['is_check'] true/false 추가
    price_sum = 0
    for product in result:
        best_sim_item = None
        for item in product['sresult']:
            item['lp'], item['hp'] = int(item['lp']), int(item['hp'])
            if item['lp'] != 0 and item['hp'] != 0:
                item['price'] = int((item['lp'] + item['hp']) / 2)
            elif item['lp'] != 0:
                item['price'] = item['lp']
            else:
                item['price'] = item['hp']
            if not best_sim_item or best_sim_item['sim'] < item['sim']:
                best_sim_item = item
            item['is_check'] = 0
        if best_sim_item:
            best_sim_item['is_check'] = 1
            price_sum += best_sim_item['price']
    for product in result:
        for item in product['sresult']:
            item['sim'] = 0
    return result


if __name__ == "__main__":
    pass
