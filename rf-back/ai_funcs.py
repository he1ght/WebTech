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
                item['price'] = (item['lp'] + item['hp']) / 2
            elif item['lp'] != 0:
                item['price'] = item['lp']
            else:
                item['price'] = item['hp']
            if not cheap_item or cheap_item['price'] > item['price']:
                cheap_item = item
            item['is_check'] = False
        if cheap_item:
            cheap_item['is_check'] = True
            price_sum += cheap_item['price']
    return result


if __name__ == "__main__":
    temp = [{'sresult': [], 'idx': 0, 'label': 'fork', 'is_check': True},
            {'sresult': [
                {'img': 'https://shopping-phinf.pstatic.net/main_8113215/81132157533.11.jpg', 'sim': 0.7014142,
                 'result': True,
                 'lp': '26000', 'hp': '0', 'link': 'http://search.naver.com',
                 'title': "Naver Open API - shop ::'fork'"},
                {'img': 'https://shopping-phinf.pstatic.net/main_1605475/16054755744.jpg', 'sim': 0.7561141,
                 'result': True,
                 'lp': '50666', 'hp': '0', 'link': 'http://search.shopping.naver.com/gate.nhn?id=13030622364',
                 'title': '(우와몰) 스테인레스 커트러리 양식기 세트'},
                {'img': 'https://shopping-phinf.pstatic.net/main_1568050/15680508720.jpg', 'sim': 0.7042755,
                 'result': True,
                 'lp': '35090', 'hp': '0', 'link': 'http://search.shopping.naver.com/gate.nhn?id=16054755744',
                 'title': '큐티폴 고아 디너 4종세트(스푼+<b>포크</b>+나이프+젓가락)'},
                {'img': 'https://shopping-phinf.pstatic.net/main_1599524/15995246582.jpg', 'sim': 0.7546325,
                 'result': True,
                 'lp': '49050', 'hp': '0', 'link': 'http://search.shopping.naver.com/gate.nhn?id=15680508720',
                 'title': '큐티폴 고아 블랙 디저트 <b>포크</b> 4P 세트'},
                {'img': 'https://shopping-phinf.pstatic.net/main_1605493/16054936267.jpg', 'sim': 0.7561141,
                 'result': True,
                 'lp': '49049', 'hp': '0', 'link': 'http://search.shopping.naver.com/gate.nhn?id=15559839198',
                 'title': '[큐티폴]고아 과일(오리엔탈) <b>포크</b>(4P)'}], 'idx': 1, 'label': 'fork', 'is_check': True},
            {'sresult': [
                {'img': 'https://shopping-phinf.pstatic.net/main_1594822/15948224346.jpg', 'sim': 0.72185767,
                 'result': True,
                 'lp': '33920', 'hp': '0', 'link': 'http://search.shopping.naver.com/gate.nhn?id=15989061395',
                 'title': 'Yosoo 68051 Heavy Duty Aluminium Alloy Rotating Bearing Turntable Round Dining Table Smooth Swivel P'}],
                'idx': 2,
                'label': 'diningtable',
                'is_check': True},
            {'sresult': [], 'idx': 3, 'label': 'vase', 'is_check': True},
            {'sresult': [], 'idx': 4, 'label': 'vase', 'is_check': True},
            {'sresult': [], 'idx': 5, 'label': 'knife', 'is_check': True}]
    result, price_sum = autoRecommend(temp)
    print(result)
    print(price_sum)
