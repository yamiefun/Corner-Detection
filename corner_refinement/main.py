import glob
import json
import cv2
import shutil
import os
import numpy as np


def parse_gt_json(json_path):
    """
    Parse the ground truth json file.
    You should modify this function so that you could get corner points
    selected manually.

    Args:
        json_path (str): The path to the json ground truth file.

    Returns:
        gt_list (list): The ground truth parse from GT file.
        total_gt_count (int): The number of corners in the GT file.
    """
    
    js_file = open(json_path)
    gt = json.load(js_file)
    gt_list = gt['gt']
    total_gt_count = 0
    for img in gt_list:
        total_gt_count += len(img['corner'])
    return gt_list, total_gt_count


def draw_user_selection(img, user_pnts):
    """
    Draw selected corners (pixels) on the image.

    Args:
        img (numpy.ndarray): The image that corners should be drawn on.
        user_pnts (list): List of corners pixel coordinates.

    Returns:
        None
    """

    ret = img.copy()
    for i, j in user_pnts:
        cv2.circle(ret, (i, j), 1, (0, 0, 255), 2)
        cv2.circle(ret, (i, j), 5, (0, 0, 255), 2)
    cv2.imwrite("tmp.png", ret)


def refine(img, img_name, user_pnts, crop_r=5):
    """
    Crop a small neighbor region of selected corners and refine the location. 
    You could customize the output for other application. For example, you
    could return the refined corner coordinates to frontend for UI rendering.

    Args:
        img (numpy.ndarray): The image that corners should be drawn on.
        img_name (str): The name of the input image.
        user_pnts (list): List of corners pixel coordinates.
        crop_r (int): The neighbor region size will be 2 * crop_r + 1.

    Returns:
        None
    """

    folder = f"./output/{img_name.split('.')[0]}"
    shutil.rmtree(folder, ignore_errors=True)
    os.makedirs(folder)
    ret = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for idx, (i, j) in enumerate(user_pnts):
        save_name = f"{folder}/{str(idx)}.png"
        ret[j, i] = [0, 0, 255]
        cv2.circle(ret, (i, j), 10, (0, 0, 255))
        cv2.circle(ret, (i, j), 2, (0, 0, 255), -1)

        # Some image preprocess here if needed.
        # img = cv2.GaussianBlur(img, (3, 3), 0)
        # img = cv2.Canny(img, 150, 200, apertureSize=3)
        # kernel = np.ones((3, 3), np.uint8)
        # img = cv2.dilate(img, kernel, iterations=1)
        # img = cv2.erode(img, kernel, iterations=1)
        # cv2.imwrite("tmp.jpg", img)
        
        crop = img[j-crop_r:j+crop_r, i-crop_r:i+crop_r]
        corner_i, corner_j = get_corner_avg(crop)
        # user select a ridiculous pixel that there's no corner in the neighbor
        if corner_i < 0 or corner_j < 0:
            continue
        i_diff = corner_i-crop_r
        j_diff = corner_j-crop_r
        trans_i = i+i_diff
        trans_j = j+j_diff
        # for x, y in corners:
        #     crop[x, y] = [0, 0, 255]
        # ret[trans_j, trans_i] = [0, 255, 0]
        cv2.circle(ret, (trans_i, trans_j), 10, (0, 255, 0))
        cv2.circle(ret, (trans_i, trans_j), 2, (0, 255, 0), -1)
    cv2.imwrite(save_name, ret)


def get_corner_avg(img):
    """
    Find corners in a given image, and return one significant corner.

    Args:
        img (numpy.ndarray): Input image.

    Returns:
        avg_j (int): pixel coordinate y
        avg_i (int): pixel coordinate x
    """

    kernel = 3
    dst = cv2.cornerHarris(img, kernel, 3, 0.04)
    dst = cv2.dilate(dst, None)
    dst_max = dst.max()
    ret = [(i, j) for i in range(dst.shape[0])
           for j in range(dst.shape[1]) if dst[i][j] > dst_max*0.5]
    if len(ret) == 0:
        return -1, -1
    avg_i, avg_j = 0, 0
    for i, j in ret:
        avg_i += i
        avg_j += j
    avg_i /= len(ret)
    avg_j /= len(ret)
    return int(avg_j), int(avg_i)


def main():
    """
    Please customize I/O and filesystem manipulation for your application.
    """
    
    files = glob.glob('input/*')
    shutil.rmtree('./output', ignore_errors=True)
    os.makedirs('./output')
    gt_list, _ = parse_gt_json('gt.json')

    for img_path in files:
        img = cv2.imread(img_path, 0)
        img_name = img_path.split('/')[-1]
        for inst in gt_list:
            if inst['image_name'] == img_name:
                user_pnts = inst['corner']
                break
        else:
            user_pnts = []
        # draw_user_selection(img, user_pnts)
        crop_r = 10
        refine(img, img_name, user_pnts, crop_r)


if __name__ == "__main__":
    main()
