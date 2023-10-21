import sys
import os
import time
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from test import copyStateDict
from PIL import Image
import cv2
from skimage import io
import numpy as np
import craft_utils
import test
import imgproc
import file_utils
import json
import zipfile
import pandas as pd
from craft import CRAFT
from collections import OrderedDict

import matplotlib.pyplot as plt


def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")


'''

'''
# CRAFT
parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--trained_model', default='D:\Luan Van\Project\med-scan-backend\CRAFT\weights\craft_mlt_25k.pth',
                    type=str, help='pretrained model')
parser.add_argument('--text_threshold', default=0.7,
                    type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.4, type=float,
                    help='text low-bound score')
parser.add_argument('--link_threshold', default=0.4,
                    type=float, help='link confidence threshold')
parser.add_argument('--cpu', default=True, type=str2bool,
                    help='Use cpu for inference')
parser.add_argument('--canvas_size', default=1280,
                    type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1.5, type=float,
                    help='image magnification ratio')
parser.add_argument('--poly', default=False,
                    action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=False,
                    action='store_true', help='show processing time')
parser.add_argument('--test_folder', default='../data',
                    type=str, help='đường dẫn tới ảnh đầu vào')
parser.add_argument('--refine', default=True,
                    action='store_true', help='enable link refiner')
parser.add_argument('--refiner_model', default='D:\Luan Van\Project\med-scan-backend\CRAFT\weights\craft_refiner_CTW1500.pth',
                    type=str, help='pretrained refiner model')

args = parser.parse_args()


""" Lấy hết các ảnh trong floder Test """
image_list, _, _ = file_utils.get_files(args.test_folder)

image_names = []
image_paths = []

# CUSTOMISE START
start = args.test_folder

for num in range(len(image_list)):
    image_names.append(os.path.relpath(image_list[num], start))

# mở folder output nếu chưa có thì tạo ra
result_folder = 'Results'
if not os.path.isdir(result_folder):
    os.mkdir(result_folder)

if __name__ == '__main__':
    first = pd.DataFrame(columns=['0', '1', '2', '3', '4', '5', '6', '7'])
    first.to_csv(
        'D:\Luan Van\Project\med-scan-backend\TextDetection\TextRecognition\data.csv', index=False)
    csv_columns = ['x_top_left', 'y_top_left', 'x_top_right', 'y_top_right',
                   'x_bot_right', 'y_bot_right', 'x_bot_left', 'y_bot_left']
    # load net
    net = CRAFT()     # initialize
    print('Đang thực hiện load weight (' + args.trained_model + ')')
    '''
    nhảy sang file test, đưa vào train model
    '''
    if args.cpu:
        net.load_state_dict(copyStateDict(torch.load(
            args.trained_model, map_location='cpu')))
    else:
        net.load_state_dict(copyStateDict(torch.load(
            args.trained_model, map_location='cpu')))

    if args.cpu:
        net = net.cpu()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()
    # LinkRefiner Đoạn này code không chạy qua nên không cần đọc vì weight đã load ở cái bên trên
    # còn refine để mặc định bên trên là False nên sẽ bị bỏ qua
    # ------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------
    refine_net = None
    if args.refine:
        from refinenet import RefineNet
        refine_net = RefineNet()
        print('Đang thực hiện load weight (' + args.refiner_model + ')')
        if args.cpu:
            refine_net.load_state_dict(copyStateDict(
                torch.load(args.refiner_model, map_location='cpu')))
            refine_net = refine_net.cpu()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(copyStateDict(
                torch.load(args.refiner_model, map_location='cpu')))

        refine_net.eval()
        args.poly = True
    # ------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------

    t = time.time()

    # load data
    for k, image_path in enumerate(image_list):
        print("Test image {:d}/{:d}: {:s}".format(k+1,
              len(image_list), image_path), end='\r')
        # ở đây đã dùng skimage.io.imread thay vì cv2.imread
        # chủ yếu đầu ra như thế sẽ làm cho ảnh định dạng với dạng RGB thay vì BGR, chỉ khác chút màu
        # mục đích để load với kênh màu khác thì chưa rõ
        image = imgproc.loadImage(image_path)

        '''nhảy qua folder test và đọc lện tiếp
        4 tham số trả về bao gồm
        bbxes trả về tọa độ của từng từ một ví dụ công viên thống nhất có 4 từ
        lưu ý là bbxes trả về 8 tọa độ bao gồm 4 đỉnh của hình chữ nhật
        polys có vẻ giống với box nhưng mà với việc load theo weight hoặc model khác, ở đây polys đang ko cần
        score_text trả về bản đồ nhiệt và đồng thời lưu bản đồ vào file kết quả
        '''
        bboxes, polys, score_text, det_scores = test.test_net(
            net, image, args.text_threshold, args.link_threshold, args.low_text, args.cpu, args.poly, args, refine_net)

        bbox_score = {}

        for box_num in range(len(bboxes)):
            item = bboxes[box_num]
            data = np.array([[int(item[0][0]), int(item[0][1]), int(item[1][0]), int(
                item[1][1]), int(item[2][0]), int(item[2][1]), int(item[3][0]), int(item[3][1])]])
            csvdata = pd.DataFrame(data, columns=csv_columns)
            csvdata.to_csv('D:\Luan Van\Project\med-scan-backend\TextDetection\TextRecognition\data.csv',
                           index=False, mode='a', header=False)
        '''
        như vậy là đã phát hiện chữ cái và tọa độ 4 đỉnh của hình chữ nhật
        từ đây ta có thể để dàng tính đượng width height nếu cần
        việc tiếp theo là tìm kiếm label ta sẽ tìm ở repo deep-text
        '''

        # save score text
        filename, file_ext = os.path.splitext(os.path.basename(image_path))
        mask_file = result_folder + "/res_" + filename + \
            '_mask.jpg'  # tạo đường dẫn file bản đồ nhiệt

        cv2.imwrite(mask_file, score_text)  # in ra bản đồ nhiệt

        file_utils.saveResult(
            image_path, image[:, :, ::-1], polys, dirname=result_folder)

    print("elapsed time : {}s".format(time.time() - t))
