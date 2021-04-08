import logging
logging.basicConfig(level=logging.WARNING, format='%(message)s')
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import numpy as np

from icecream import ic
from tqdm import tqdm
from utils_HBLab import *


DEBUG = False
SHOW_IMG = False

SAVE_IMG = True
SAVE_DOUBLE_CHECK = False

PNG  = 'data_light/images'
JSON = 'data_light/images_mask'
SAVED_GROUND_TRUTH_DIR = 'data_light/images_mask'
SAVE_STACKED_DIR = '/home/maihai/0_PROJECT_heads_airbags_cubebox/1_gui_Tung/1_phan_4_tai/double_check_img'


def list_Path_of_filetype_in_folder(dir, str_extention):
    paths = Path(dir)
    paths = paths.glob(f'*.{str_extention}')
    paths = list(paths)
    return paths


def draw_mask(x):
    height, width = x.shape
    img = np.zeros((height, width))
    for label, pts in retreive_polygons_and_labels_from_json(json_path):
        color = select_color_for_label_exclude_ear(label)
        img = cv2.fillPoly(img, pts, color=color)
    return img



def obtain_filenames_without_extension(dirname, extension):
    filenames = [i.stem for i in Path(dirname).glob(f'**/*.{extension}')]
    return filenames

def obtain_filenames_with_extension(dirname, extension):
    filenames = [i for i in Path(dirname).glob(f'**/*.{extension}')]
    return filenames


def plot_img_with_text(img, text):
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    ax.axis("off")
    padding = 5
    ax.annotate(
        text,
        fontsize=6,
        xy=(0, 0),
        xytext=(padding - 1, -(padding - 1)),
        textcoords='offset pixels',
        bbox=dict(facecolor='white', alpha=1, pad=padding),
        va='top', ha='left')
    plt.show()

def read_then_resize_images(img_path, after_width, after_height):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (after_width, after_height), interpolation=cv2.INTER_NEAREST)
    return img

def obtain_original_filenames(png_dir):
    p = Path(png_dir)
    list_paths = p.glob('*.png')
    list_paths = [str(i) for i in list_paths]
    return list_paths

def select_color_for_labels_include_ear(label):
    HEAD, AB, AB_R  = 1, 2, 3
    EAR             = 4
    BACKGROUND      = 0

    if   label == 'head':
        return HEAD
    elif label == 'ab':
        return AB
    elif label == 'ab_r':
        return AB_R
    elif label == 'ear':
        return EAR
    else:
        return BACKGROUND

def select_color_for_labels_only_ear(label):
    HEAD, AB, AB_R  = 0, 0, 0
    EAR             = 1
    BACKGROUND      = 0

    if   label == 'head':
        return HEAD
    elif label == 'ab':
        return AB
    elif label == 'ab_r':
        return AB_R
    elif label == 'b1':
        return EAR
    else:
        return BACKGROUND

def select_color_for_label_exclude_ear(label):
    HEAD, AB, AB_R = 1, 2, 3
    EAR            = HEAD
    BACKGROUND     = 0

    if   label == 'head':
        return HEAD
    elif label == 'ab':
        return AB
    elif label == 'ab_r':
        return AB_R
    elif label == 'ear':
        return EAR
    else:
        return BACKGROUND

def draw_fillPoly_only_ears(json_path):
    height, width = retreive_img_shape_from_json(json_path)
    img = np.zeros((height, width))
    for label, pts in retreive_polygons_and_labels_from_json(json_path):
        print(label)
        color = select_color_for_labels_only_ear(label)
        img = cv2.fillPoly(img, pts, color=color)
    return img

def draw_fillPoly_with_ears(json_path):
    height, width = retreive_img_shape_from_json(json_path)
    img = np.zeros((height, width))
    for label, pts in retreive_polygons_and_labels_from_json(json_path):
        color = select_color_for_labels_include_ear(label)
        img = cv2.fillPoly(img, pts, color=color)
    return img

def draw_fillPoly_without_ears(json_path):
    height, width = retreive_img_shape_from_json(json_path)
    img = np.zeros((height, width))
    for label, pts in retreive_polygons_and_labels_from_json(json_path):
        color = select_color_for_label_exclude_ear(label)
        img = cv2.fillPoly(img, pts, color=color)
    return img


def retreive_img_shape_from_json(json_path):
    data = parse_data_from_json(json_path)
    height = data['imageHeight']
    width = data['imageWidth']
    return height, width


def compose_path_for_imwrite(save_dir, json_path):
    name = os.path.basename(json_path)
    name = os.path.splitext(name)[0]
    path = save_dir + '/' + name + '.png'
    return path


def write_img(json_path, image):
    filename = compose_path_for_imwrite(json_path)
    cv2.imwrite(filename, image)

def obtain_img_shape(img_path):
    img = cv2.imread(img_path, 0)
    height, width = img.shape[:2]
    return height, width

def retreive_polygons_and_labels_from_json(json_path):
    data = parse_data_from_json(json_path)
    labels = []
    points = []
    for segmentation in data['shapes']:
        label = segmentation['label']
        labels.append(label)

        pts = segmentation['points']
        pts = np.array(pts, np.int32)
        pts = [pts]
        points.append(pts)

    pairs = zip(labels, points)

    return [(item[0], item[1]) for item in zip(labels, points)]

def obtain_sorted_png_paths(parent_dir):
    parent_dir = Path(parent_dir)
    png_paths = parent_dir.glob('*.png')
    png_paths = [str(p) for p in png_paths]
    png_paths.sort()
    return png_paths

def retreive_polygons_from_data(data):
    pts = data['shapes'][0]['points']
    pts = np.array(pts, dtype=np.int32)
    pts = [pts]
    return pts

def parse_data_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def obtain_sorted_json_paths(parent_dir):
    parent_dir = Path(parent_dir)
    json_paths = parent_dir.glob('*.json')
    json_paths = [str(p) for p in json_paths]
    json_paths.sort()
    return json_paths


def is_cover_name(predicted_mask, target_mask):
    intersection = target_mask[predicted_mask == target_mask].sum().sum()
    ground_truth = target_mask.sum()
    IoU = intersection / ground_truth
    THRESHOLD = 0.95
    if IoU >= THRESHOLD:
        return 1
    else:
        return 0


def calc_pixel_accuracy(mask, target_mask):
    return np.asarray(mask == target_mask, dtype=np.uint8).sum() / (mask.shape[0] * mask.shape[1])


def calculate_iou(mask, target_mask):
    intersection = mask[mask == target_mask].sum()
    _mask = mask + target_mask
    _mask[_mask != 0] = 1
    union = _mask.sum()
    return intersection / union


def cal_iou_box(box_0, box_1):
    x_max = max(box_0[0], box_1[0])
    y_max = max(box_0[1], box_1[1])
    x_min = min(box_0[2], box_1[2])
    y_min = min(box_0[3], box_1[3])

    inter_area = (x_min - x_max + 1) * (y_min - y_max + 1)

    box_0_area = (box_0[2] - box_0[0] + 1) * (box_0[3] - box_0[1] + 1)
    box_1_area = (box_1[2] - box_1[0] + 1) * (box_1[3] - box_1[1] + 1)

    iou = inter_area / float(box_0_area + box_1_area - inter_area)

    return iou


def post_process_mask(mask_img):
    _, contours, _ = cv2.findContours(mask_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

    if len(contours) == 0:
        return mask_img

    temp_mask = np.zeros(mask_img.shape, dtype=mask_img.dtype)
    temp_mask = cv2.drawContours(temp_mask, [contours[0]], -1, 1, -1)
    return temp_mask


def automatic_brightness_and_contrast(image, clip_hist_percent=1):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Calculate grayscale histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = [float(hist[0])]
    for index in range(1, hist_size):
        accumulator.append(accumulator[index - 1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum / 100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size - 1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return auto_result, alpha, beta


def cal_len(line):
    return int(np.sqrt((line[1][0] - line[0][0]) ** 2 + (line[1][1] - line[0][1]) ** 2))


def warp_affine(fc_ori_img, fc_mask):
    fc_ori_img = np.asarray(fc_ori_img, dtype=np.uint8)
    if np.max(fc_mask) == 0:
        return fc_ori_img, fc_mask

    kernel = np.array([[1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1]], dtype=np.uint8)
    fc_mask = cv2.morphologyEx(fc_mask, cv2.MORPH_CLOSE, kernel)

    _, temp_contours, _ = cv2.findContours(fc_mask, 1, 2)
    temp_contours = sorted(temp_contours, key=lambda x: cv2.contourArea(x), reverse=True)
    temp_contours = temp_contours[0]

    temp_rect = cv2.minAreaRect(temp_contours)
    temp_box = cv2.boxPoints(temp_rect)
    temp_box = np.int0(temp_box)
    fc_mask = cv2.fillPoly(fc_mask, [temp_box], 1)

    # cv2.namedWindow('test', cv2.WINDOW_NORMAL)
    # cv2.imshow('test', fc_mask)
    # cv2.waitKey()

    leng_0 = cal_len([temp_box[0], temp_box[1]])
    leng_1 = cal_len([temp_box[0], temp_box[-1]])

    if leng_0 > leng_1:
        pts1 = np.float32([temp_box[0], temp_box[1],
                           temp_box[2], temp_box[3]])
    else:
        pts1 = np.float32([temp_box[1], temp_box[2],
                           temp_box[3], temp_box[0]])
        leng_0, leng_1 = leng_1, leng_0

    pts2 = np.float32([[0, 0], [leng_0, 0], [leng_0, leng_1], [0, leng_1]])
    matrix_transfrom = cv2.getPerspectiveTransform(pts1, pts2)

    result = cv2.warpPerspective(fc_ori_img, matrix_transfrom, (leng_0, leng_1))

    return result, fc_mask



if __name__ == '__main__':
    json_filenames = obtain_filenames_without_extension(JSON, 'json')

    for filenames in tqdm(json_filenames):
        json_path = JSON + '/' + filenames + '.json'
        png_path = PNG + '/' + filenames + '.png'

        # Since not every png has corresponding json
        if os.path.exists(json_path):
            original_img = cv2.imread(png_path)
            segmented_img = draw_fillPoly_only_ears(json_path)

        if DEBUG:
            ic(f'--- {filenames} ---')
            ic(np.unique(segmented_img))
            ic(png_path)
            ic(json_path)
            ic('------')

        if SHOW_IMG:
            segmented_img_with_color = original_img[:, :, 0] * segmented_img * 255
            stacks = np.hstack((original_img[:, :, 0], segmented_img_with_color,))
            plt.imshow(stacks)
            plt.show()

        if SAVE_IMG:
            path = compose_path_for_imwrite(SAVED_GROUND_TRUTH_DIR, json_path)
            cv2.imwrite(path, segmented_img)
            logging.warning(f'path: {path}')
            print('------')

        if SAVE_DOUBLE_CHECK:
            segmented_img_with_color = original_img[:, :, 0] * segmented_img
            stacks = np.hstack((original_img[:, :, 0], segmented_img_with_color,))
            path = compose_path_for_imwrite(SAVE_STACKED_DIR, json_path)
            cv2.imwrite(path, stacks*255)
            logging.warning(f'path: {path}')
            print('------')
