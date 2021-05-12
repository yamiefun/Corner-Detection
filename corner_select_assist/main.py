import argparse
import cv2
import numpy as np
from inference import *
import os
import shutil
import glob
import time
import tensorflow as tf
import json

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str)
    parser.add_argument("--model", type=str, default='./restore_weights/')
    return parser.parse_args()


def canny(img_path, img):
    """
    Canny edge detector.

    Args:
        @img_path (str): Path to the input image.
        @img (numpy.ndarray): Input image.

    Returns:
        Edge of the input image in BGR format. (numpy.ndarray)
    """
    img = cv2.medianBlur(img, 3)
    img = cv2.medianBlur(img, 3)
    img = cv2.medianBlur(img, 3)
    print("blur image")
    # cv2_imshow(img)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.Canny(img, 150, 200, apertureSize=3)
    edges = cv2.dilate(edges, kernel, iterations=6)
    edges = cv2.erode(edges, kernel, iterations=4)
    in_filename = img_path.split('/')[-1]
    cv2.imwrite(f"./output/{in_filename.split('.')[0]}/canny.png", edges)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


def line_detect(args, img):
    """
    This function is not being used so far.
    """

    # line_img = src_img.copy()
    # line_img = np.zeros(src_img.shape)
    img = cv2.medianBlur(img, 3)
    img = cv2.medianBlur(img, 3)
    img = cv2.medianBlur(img, 3)
    print("blur image")
    # cv2_imshow(img)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.Canny(img, 150, 200, apertureSize=3)
    edges = cv2.dilate(edges, kernel, iterations=6)
    # edges = cv2.erode(edges, kernel, iterations=6)
    line_img = edges.copy()
    line_img = cv2.cvtColor(line_img, cv2.COLOR_GRAY2BGR)
    print("canny result")
    # cv2_imshow(edges)
    lines = cv2.HoughLinesP(edges, 10.0, np.pi/180, 100,
                            minLineLength=200, maxLineGap=100)
    empty_img = np.zeros(img.shape)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(empty_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    # cv2_imshow(line_img)
    # cv2_imshow(empty_img)

    in_filename = args.img.split('/')[-1]
    cv2.imwrite(f"./output/{in_filename.split('.')[0]}/line.png", line_img)
    return empty_img


def get_harris(img, kernel=5):
    """
    Use harris algorithm to detect corners.

    Args:
        @img (numpy.ndarray): Input image.
        @kernel (int): The kernel size of opencv Harris corner detect function.

    Returns:
        @ret (list): List of corners.
    """
    dst = cv2.cornerHarris(img, kernel, 3, 0.04)
    # result is dilated for marking the corners, not important
    dst = cv2.dilate(dst, None)
    # Threshold for an optimal value, it may vary depending on the image.
    # img[dst>0.01*dst.max()]=[0,0,255]
    dst_max = dst.max()
    ret = [(i, j) for i in range(dst.shape[0])
           for j in range(dst.shape[1]) if dst[i][j] > dst_max*0.05]
    return ret


def get_corner(img):
    """
    Use Harris function to detect corners with different kernel size.

    Args:
        @img (numpy.ndarray): Input image.

    Returns:
        @all_ret (list): List of all corners detect by Harris.
    """
    print(img.shape)
    cv2.imwrite("tmp.jpg", img)
    img = np.float32(img)
    all_ret = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    all_ret += get_harris(gray, kernel=5)
    all_ret += get_harris(gray, kernel=15)
    all_ret = list(set(all_ret))
    # dst = cv2.cornerHarris(gray, 5, 3, 0.04)
    # # result is dilated for marking the corners, not important
    # dst = cv2.dilate(dst, None)
    # # Threshold for an optimal value, it may vary depending on the image.
    # # img[dst>0.01*dst.max()]=[0,0,255]
    # dst_max = dst.max()
    # ret = [(i, j) for i in range(dst.shape[0])
    #        for j in range(dst.shape[1]) if dst[i][j] > dst_max*0.05]
    print("number of candidate corner: ", len(all_ret))
    # dstt = dst>0.01*dst_max
    # print(dstt)
    # print(np.count_nonzero(dstt==True))
    
    return all_ret


def segmentation(img, args):
    """
    Run indoor segmentation and store the output image.
    You should customize the return value to fit your application. For example,
    return directly the labels of every pixel.
    Reference: https://github.com/hellochick/Indoor-segmentation

    Args:
        @img (numpy.ndarray): The input image.
        @args (argparse): The input arguments.

    Returns:
        Path to the output image. (str)
    """

    tf.reset_default_graph()
    SAVE_DIR = './output/'
    in_filename = img.split('/')[-1]
    filename = f"./output/{in_filename.split('.')[0]}/blur.png"
    print(filename)
    file_type = in_filename.split('.')[-1]

    if os.path.isfile(img):
        print('successful load img: {0}'.format(img))
    else:
        print('not found file: {0}'.format(img))
        sys.exit(0)

    # Prepare image.
    if file_type.lower() == 'png':
        img = tf.image.decode_png(tf.read_file(img), channels=3)
    elif file_type.lower() == 'jpg':
        img = tf.image.decode_jpeg(tf.read_file(img), channels=3)
    else:
        print('cannot process {0} file.'.format(file_type))

    # Convert RGB to BGR.
    img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
    img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]),
                  dtype=tf.float32)
    # Extract mean.
    img -= IMG_MEAN

    # Create network.
    net = DeepLabResNetModel({'data': tf.expand_dims(img, dim=0)},
                             is_training=False, num_classes=NUM_CLASSES)

    # Which variables to load.
    restore_var = tf.global_variables()

    # Predictions.
    raw_output = net.layers['fc_out']
    raw_output_up = tf.image.resize_bilinear(raw_output, tf.shape(img)[0:2, ])
    raw_output_up = tf.argmax(raw_output_up, dimension=3)
    pred = tf.expand_dims(raw_output_up, dim=3)

    # Set up TF session and initialize variables.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()

    sess.run(init)

    # Load weights.
    ckpt = tf.train.get_checkpoint_state(args.model)

    if ckpt and ckpt.model_checkpoint_path:
        loader = tf.train.Saver(var_list=restore_var)
        load_step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
        load(loader, sess, ckpt.model_checkpoint_path)
    else:
        print('No checkpoint file found.')
        load_step = 0

    # Perform inference.
    preds = sess.run(pred)

    msk = decode_labels(preds, num_classes=NUM_CLASSES)
    im = Image.fromarray(msk[0])
    if not os.path.exists(f"{SAVE_DIR}/{in_filename.split('.')[0]}"):
        os.makedirs(f"{SAVE_DIR}/{in_filename.split('.')[0]}")
    # im.save(SAVE_DIR + in_filename)
    im.save(f"{SAVE_DIR}/{in_filename.split('.')[0]}/seg.{in_filename.split('.')[1]}")
    print('The output file has been saved.')
    return f"{SAVE_DIR}/{in_filename.split('.')[0]}/seg.{in_filename.split('.')[1]}"


def filter_corners(corners, seg_path, img, edge_img):
    """
    Filter out impossible candidate points with some rules.
    1. To deduplicate the corners that are too close to others, use
       @corner_dedup().
    2. Corners should only appear between walls and floor. 
       Use @between_wall_and_floor() to remove ridiculous candidate points.

    Args:
        @corners (list): List of all candidate corner points.
        @seg_path (str): Path to the segmentation image.
        @img (numpy.ndarray): Input image.
        @edge_img (numpy.ndarray): Edge image generate by Canny edge detector.

    Returns:
        @filter_corner (list): List of corners that seem to be legit.
    """
    all_corner_img = edge_img.copy()
    corner_img = img.copy()
    dedup_img = img.copy()
    path = os.path.split(seg_path)[0]
    img_type = os.path.split(seg_path)[1].split('.')[1]

    for (i, j) in corners:
        all_corner_img[i][j] = [0, 0, 255]
        cv2.circle(all_corner_img, (j, i), 20, (0, 0, 255))
    cv2.imwrite(f'{path}/all_corner.{img_type}', all_corner_img)

    dedup_corner = corner_dedup(corners, method=1, img=img)

    for (i, j) in dedup_corner:
        dedup_img[i][j] = [0, 0, 255]
        cv2.circle(dedup_img, (j, i), 17, (0, 0, 255))

    cv2.imwrite(f'{path}/dedup_corner.{img_type}', dedup_img)

    seg = cv2.imread(seg_path)
    filter_corner = []
    for (i, j) in dedup_corner:
        if between_wall_and_floor(seg, i, j):
            filter_corner.append((i, j))
            corner_img[i][j] = [0, 0, 255]
            cv2.circle(corner_img, (j, i), 2, (0, 0, 255), -1)
            # cv2.circle(corner_img, (j, i), 1, (0, 0, 255))
    cv2.imwrite(f'{path}/filter_corner.{img_type}', corner_img)
    return filter_corner


def corner_dedup(corners, method, img):
    """
    Deduplicate the corner points which are gather together.

    Args:
        @corners (list): List of corners.
        @method (int): Method.
            1: Discretize with block size @k_size.
        @img (numpy.ndarray): Input image.

    Returns:
        @boundary_corner (list): Corners after deduplication.
    """
    dedup_corner = []
    # ret_img = img.copy()
    if method == 1:
        k_size = 3
        dedup_corner = [((i//k_size)*k_size+k_size//2,
                        (j//k_size)*k_size+k_size//2) for (i, j) in corners]
        dedup_corner = list(set(dedup_corner))

        k_size = 10
        dedup_corner = [((i//k_size)*k_size+k_size//2,
                        (j//k_size)*k_size+k_size//2)
                        for (i, j) in dedup_corner]
        dedup_corner = list(set(dedup_corner))

        k_size = 11
        dedup_corner = [((i//k_size)*k_size+k_size//2,
                        (j//k_size)*k_size+k_size//2)
                        for (i, j) in dedup_corner]
        dedup_corner = list(set(dedup_corner))

        boundary_corner = [(i, j) for (i, j) in dedup_corner
                           if (i < img.shape[0] and j < img.shape[1])]
        # print(boundary_corner)
    else:
        print("deduplicate method not exsit")
    return boundary_corner

    # for (i, j) in boundary_corner:
    #     ret_img[i][j] = [0, 0, 255]
    #     cv2.circle(ret_img, (j, i), 17, (0, 0, 255))

    # in_filename = args.img.split('/')[-1]
    # cv2.imwrite(f"./output/{in_filename.split('.')[0]}/result.png", ret_img)


def between_wall_and_floor(seg, i, j):
    """
    To determine whether a corner is between walls and floor or not.

    Args:
        @seg (numpy.ndarray): Segmentation image.
        @i (int): pixel coordinate y.
        @j (int): pixel coordinate x.

    Return:
        Whether the corner is between walls and floor or not. (bool)
    """
    kernal_size = 71
    WALL_COLOR = (120, 120, 180)
    FLOOR_COLOR = (4, 200, 4)
    FLOOR_COLOR_2 = (3, 200, 4)
    crop_seg = seg[max(0, i-kernal_size//2):min(i+kernal_size//2+1,
                                                seg.shape[0]),
                   max(0, j-kernal_size//2):min(j+kernal_size//2+1,
                                                seg.shape[1])]
    crop_seg = crop_seg.reshape(-1, 3)
    return WALL_COLOR in [tuple(i) for i in crop_seg] and\
        (FLOOR_COLOR in [tuple(i) for i in crop_seg] or
         FLOOR_COLOR_2 in [tuple(i) for i in crop_seg])
    

def sharpen(img):
    """
    Sharpen an image with a kernel.
    
    Args:
        @img (numpy.ndarray): Input image.

    Return:
        Sharpen image. (numpy.ndarray)
    """
    f = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return cv2.filter2D(img, -1, f)


def preprocess(img_path, img):
    """
    Image preprocessing.

    Args:
        @img_path (str): Path to the input image.
        @img (numpy.ndarray): Input image.

    Returns:
        @ret (numpy.ndarray): Image after preprocess.
    """
    in_filename = img_path.split('/')[-1]
    filename = f"./output/{in_filename.split('.')[0]}/blur.png"

    ret = img.copy()
    ret = cv2.medianBlur(ret, 3)
    ret = cv2.medianBlur(ret, 3)
    ret = cv2.medianBlur(ret, 3)
    # ret = cv2.medianBlur(ret, 7)
    # ret = sharpen(ret)
    cv2.imwrite(filename, ret)
    return ret


def arrange_folder(img_path):
    """
    Remove output folder if exist and create new one.

    Args:
        @img_path (str): Path to input image.

    Returns:
        None.
    """
    in_filename = img_path.split('/')[-1]
    folder = f"./output/{in_filename.split('.')[0]}"
    shutil.rmtree(folder, ignore_errors=True)
    os.makedirs(folder)


def generate_gt(img_path, gt_list, src_img, filter_corner):
    in_filename = img_path.split('/')[-1]
    in_filename, image_type = in_filename.split('.')
    result_img = cv2.imread(f'./output/{in_filename}/filter_corner.{image_type}')
    gt_img = src_img.copy()
    tp_count = 0
    dist_threshold = 8     # distance threshold, used to count true positive
    for i in range(len(gt_list)):
        # find image ground truth
        if gt_list[i]['image_name'] == in_filename:
            # draw ground truth on to image
            for k, j in gt_list[i]['corner']:
                cv2.circle(gt_img, (k, j), dist_threshold, (0, 0, 255), 2)
                gt_img[j][k] = [0, 0, 255]
                # count true positive
                for a, b in filter_corner:
                    dist = np.linalg.norm(np.array([a, b])-np.array([j, k]))
                    if dist < dist_threshold:
                        tp_count += 1
                        cv2.circle(result_img, (k, j), dist_threshold, (0, 255, 0), 2)
                        break
                else:
                    cv2.circle(result_img, (k, j), dist_threshold, (0, 0, 255), 2)
            cv2.imwrite(f'./output/{in_filename}/gt.{image_type}', gt_img)
            cv2.imwrite(f'./output/{in_filename}/filter_corner.{image_type}', result_img)
            break
    else:
        print(f"no ground truth for image: {in_filename}")
    return tp_count


def parse_gt_json(json_path):
    js_file = open(json_path)
    gt = json.load(js_file)
    gt_list = gt['gt']
    total_gt_count = 0
    for img in gt_list:
        total_gt_count += len(img['corner'])
    return gt_list, total_gt_count


def main():
    args = get_arguments()
    files = glob.glob('input/*')
    time_rec = []
    net = ""
    shutil.rmtree('./output')
    gt_list, total_gt_count = parse_gt_json('gt.json')
    total_tp_count = 0
    total_filter_corner = 0
    for img in files:
        ts_now = time.time()
        src_img = cv2.imread(img)
        arrange_folder(img)
        processed_img = preprocess(img, src_img)
        # corners_map = get_corner(src_img)
        # line_img = line_detect(args, processed_img)
        edge_img = canny(img, processed_img)
        corners = get_corner(edge_img)
        seg_path = segmentation(img, args)
        # seg_path = 'output/out_C31.jpg'
        filter_corner = filter_corners(corners, seg_path, src_img, edge_img)
        time_rec.append(time.time()-ts_now)
        # corner_dedup(args, filter_corner, src_img, method=1)
        total_tp_count += generate_gt(img, gt_list, src_img, filter_corner)
        total_filter_corner += len(filter_corner)
        print(f"finish image {img}")
    print(f"avg time: {sum(time_rec)/len(time_rec)}")
    print(f"TP: {total_tp_count}, GT: {total_gt_count}")
    print(f"Total number of candidate corner: {total_filter_corner}")
    print(f"Recall: {total_tp_count/total_gt_count}")


if __name__ == "__main__":
    main()
