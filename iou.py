import numpy as np
def compute_iou(boxA, boxB):
    [x1,x2,y1,y2], [x3,x4,y3,y4] = boxA, boxB  #左上&右下
    # 注意x向右为+；y向下为+。因为是矩阵
    # 确定inter区域左上右下
    loc_1_x = max(x1, x3)
    loc_1_y = max(y1, y3)
    loc_2_x = min(x2, x4)
    loc_2_y = min(y2, y4)

    inter = np.max(loc_2_x-loc_1_x, 0) * np.max(loc_2_y-loc_1_y, 0)
    union = (x2-x1)*(y2-y1) + (x4-x3)*(y4-y3) - inter
    iou = inter / union
    # print(inter, union)
    return iou

if __name__ == '__main__':
    boxA = [1,3,1,3]
    boxB = [2,4,2,4]
    iou = compute_iou(boxA, boxB)
    print('\n')
    print(iou)