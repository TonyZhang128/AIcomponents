import numpy as np
from .iou import compute_iou
# NMS是一种用于去除冗余的边界框的方法，它首先将所有的边界框按照置信度排序，然后从中选择置信度最高的边界框，
# 并移除所有与它有高IoU的边界框。这个过程会重复进行，直到没有剩余的边界框。
def compute_NMS(boxes, scores, threshold):
    # boxes: 边界框列表，每个框是一个格式为 [x1, y1, x2, y2] 的列表
    # scores: 每个边界框的得分列表
    # threshold: NMS的IoU阈值
    sorted_id = np.argsort(scores)
    boxes = [boxes[i] for i in sorted_id]
    scores = [scores[i] for i in sorted_id]

    keep_id = []

    while boxes:
        current_box = boxes.pop()
        current_score = scores.pop()

        keep_id.append(sorted_id[-1])
        sorted_id = sorted_id[:-1]

        discard_ids = []
        for i, box in enumerate(boxes):
            iou = compute_iou(current_box, box)
            if iou > threshold:
                discard_ids.append(i)
        
        for i in discard_ids[::-1]:
            boxes.pop(i)
            scores.pop(i)
            sorted_id = np.delete(sorted_id, i)
        
    return keep_id