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

def autoRecommend(result):
    # todo return 값 reuslt ['is_check'] true/false 추가
    pass