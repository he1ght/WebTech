import sys

sys.path.insert(0, "../Classification/")
import detector

sys.path.insert(0, '../Similarity/')
import check_similarity


def detectObj():
    return detector.detect_object(image_folder="uploads/", output="static/output",
                                  config_path="../Classification/config/yolov3.cfg",
                                  weights_path="../Classification/weights/yolov3.weights",
                                  class_path="../Classification/data/coco.names")


def img2vec(input_path, target_list, threshold=0.7):
    return check_similarity.img_sim(input_path, target_list, threshold=threshold)


def autoRecommend(result, budget=0):
    # todo return 값 reuslt ['is_check'] true/false 추가
    price_sum = 0
    for product in result:
        cheap_item = None
        for item in product['sresult']:
            item['lp'], item['hp'] = int(item['lp']), int(item['hp'])
            if item['lp'] != 0 and item['hp'] != 0:
                item['price'] = int((item['lp'] + item['hp']) / 2)
            elif item['lp'] != 0:
                item['price'] = item['lp']
            else:
                item['price'] = item['hp']
            if not cheap_item or cheap_item['price'] > item['price']:
                cheap_item = item
            item['is_check'] = 0
        if cheap_item:
            cheap_item['is_check'] = 1
            price_sum += cheap_item['price']
    return result


if __name__ == "__main__":
    pass
