"""
extract canny edges from the infrared images.
"""

import os
import cv2 as cv
from tqdm import tqdm
import argparse


def get_canny_edge(img_path, low_threshold, high_threshold):
    img = cv.imread(img_path, 0)
    edge = cv.Canny(img, low_threshold, high_threshold)

    return edge


def preprocess(imgfile_path, save_path, low_threshold, high_threshold):
    os.makedirs(save_path, exist_ok=True)

    for root, dirs, files in os.walk(imgfile_path):
        for file in tqdm(files):
            img_path = os.path.join(root, file)
            edge = get_canny_edge(img_path, low_threshold, high_threshold)
            cv.imwrite(os.path.join(save_path, file), edge)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--imgfile_path", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--low_threshold", type=int, default=60, help="low_threshold of the canny edge")
    parser.add_argument("--high_threshold", type=int, default=120, help="high_threshold of the canny edge")

    args = parser.parse_args()

    imgfile_path = args.imgfile_path
    save_path = args.save_path
    low_threshold = args.low_threshold
    high_threshold = args.high_threshold

    preprocess(imgfile_path, save_path, low_threshold, high_threshold)