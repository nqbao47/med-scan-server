from PIL import Image
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from label_medicines import label_medicines
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import cv2
import time
import csv
import glob

# Load the VietOCR model
config = Cfg.load_config_from_name('vgg_transformer')
config['export'] = 'transformerocr_checkpoint.pth'
config['device'] = 'cpu'
config['predictor']['beamsearch'] = False
detector = Predictor(config)

# Open the input CSV file
with open('data.csv', 'r') as csvinput:
    with open('../../results/output.csv', 'w', encoding='utf-8') as csvoutput:
        writer = csv.writer(csvoutput, lineterminator='\n')
        reader = csv.reader(csvinput)
        all = []
        # Đọc dòng tiêu đề và thêm cột mới vào dòng tiêu đề
        header = next(reader)
        header.append('NewColumn')  # Đặt tên cho cột mới
        all.append(header)
        i = 0
        path = glob.glob("cutEachWord/*.jpg")
        cv_img = []
        for i in range(len(path)):
            print('cutEachWord/1'+str(i)+'.jpg')
            n = Image.open('cutEachWord/1'+str(i)+'.jpg')
            print(str(detector.predict(n)))
            cv_img.append(str(detector.predict(n)))
        i = 0
        for row in reader:
            # Thêm giá trị từ cv_img vào cột mới
            row.append(str(cv_img[i]))
            # Gán nhãn cho thuốc và thêm vào cột mới
            # Gán nhãn dựa vào cột cuối cùng của dữ liệu đã trích xuất
            medicine_label = label_medicines(row[-1])
            row.append(medicine_label)

            i = i + 1
            all.append(row)

        writer.writerows(all)
