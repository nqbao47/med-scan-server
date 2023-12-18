# -*- coding: utf-8 -*-
import os
import shutil

import crop_images
import cv2
import matplotlib.pyplot as plt
import numpy as np

# borrowed from https://github.com/lengstrom/fast-style-transfer/blob/master/src/utils.py


def get_files(img_dir):
    imgs, masks, xmls = list_files(img_dir)
    return imgs, masks, xmls


def list_files(in_path):
    img_files = []
    mask_files = []
    gt_files = []
    for dirpath, dirnames, filenames in os.walk(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if (
                ext == ".jpg"
                or ext == ".jpeg"
                or ext == ".gif"
                or ext == ".png"
                or ext == ".pgm"
            ):
                img_files.append(os.path.join(dirpath, file))
            elif ext == ".bmp":
                mask_files.append(os.path.join(dirpath, file))
            elif ext == ".xml" or ext == ".gt" or ext == ".txt":
                gt_files.append(os.path.join(dirpath, file))
            elif ext == ".zip":
                continue
    # img_files.sort()
    # mask_files.sort()
    # gt_files.sort()
    return img_files, mask_files, gt_files


def saveResult(img_file, img, boxes, dirname="Results", verticals=None, texts=None):
    """save text detection result one by one
    Args:
        img_file (str): image file name
        img (array): raw image context
        boxes (array): array of result file
            Shape: [num_detections, 4] for BB output / [num_detections, 4] for QUAD output
    Return:
        None
    """

    img = np.array(img)

    # make result file list: tên ảnh và đuôi ảnh
    filename, file_ext = os.path.splitext(os.path.basename(img_file))
    # result directory
    res_file = dirname + "res_" + filename + ".txt"
    res_img_file = dirname + "res_" + filename + ".jpg"

    if not os.path.isdir(dirname):
        os.mkdir(dirname)

    # Tạo đường dẫn đến thư mục "cut_images"
    cut_images_dir = (
        "D:/Luan Van/Project/med-scan-backend/processing/Craft_model/cut_images/"
    )

    # Kiểm tra nếu thư mục tồn tại thì xoá tất cả các tệp trong thư mục
    if os.path.exists(cut_images_dir):
        shutil.rmtree(cut_images_dir)

    # Tạo lại thư mục
    os.makedirs(cut_images_dir)

    with open(res_file, "w") as f:
        for i, box in enumerate(boxes):
            poly = np.array(box).astype(np.int32).reshape((-1))
            strResult = ",".join([str(p) for p in poly]) + "\r\n"
            f.write(strResult)

            poly = poly.reshape(-1, 2)
            # cv2.polylines(
            #     img, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=1
            # )
            ptColor = (0, 255, 255)
            xmin = min(poly[:, 0])
            xmax = max(poly[:, 0])
            ymin = min(poly[:, 1])
            ymax = max(poly[:, 1])
            width = xmax - xmin
            height = ymax - ymin
            # các điểm này từ file txt
            pts = np.array([[xmin, ymax], [xmax, ymax], [xmax, ymin], [xmin, ymin]])

            word = crop_images.crop(pts, img)

            folder = "/".join(filename.split("/")[:-1])

            # folder cut_images
            # dir = "D:/Luan Van/Project/med-scan-backend/processing/Craft_model/cut_images/"
            dir = cut_images_dir

            if os.path.isdir(os.path.join(dir + folder)) == False:
                os.makedirs(os.path.join(dir + folder))
            try:
                file_name = os.path.join(dir + filename)
                cv2.imwrite(file_name + str(i) + ".jpg", word)
            except:
                continue

            if verticals is not None:
                if verticals[i]:
                    ptColor = (255, 0, 0)

            if texts is not None:
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                cv2.putText(
                    img,
                    "{}".format(texts[i]),
                    (poly[0][0] + 1, poly[0][1] + 1),
                    font,
                    font_scale,
                    (0, 0, 0),
                    thickness=1,
                )
                cv2.putText(
                    img,
                    "{:.2f}".format(texts[i]),
                    tuple(poly[0]),
                    font,
                    font_scale,
                    (0, 255, 255),
                    thickness=1,
                )

    # Save result image
    cv2.imwrite(res_img_file, img)


# # -*- coding: utf-8 -*-
# import os
# import shutil

# import crop_images
# import cv2
# import matplotlib.pyplot as plt
# import numpy as np


# def get_files(img_dir):
#     imgs, masks, xmls = list_files(img_dir)
#     return imgs, masks, xmls


# def list_files(in_path):
#     img_files = []
#     mask_files = []
#     gt_files = []
#     for dirpath, dirnames, filenames in os.walk(in_path):
#         for file in filenames:
#             filename, ext = os.path.splitext(file)
#             ext = str.lower(ext)
#             if (
#                 ext == ".jpg"
#                 or ext == ".jpeg"
#                 or ext == ".gif"
#                 or ext == ".png"
#                 or ext == ".pgm"
#             ):
#                 img_files.append(os.path.join(dirpath, file))
#             elif ext == ".bmp":
#                 mask_files.append(os.path.join(dirpath, file))
#             elif ext == ".xml" or ext == ".gt" or ext == ".txt":
#                 gt_files.append(os.path.join(dirpath, file))
#             elif ext == ".zip":
#                 continue
#     return img_files, mask_files, gt_files


# def saveResult(
#     img_file,
#     img,
#     boxes,
#     dirname="D:/Luan Van/Project/med-scan-backend/Training/Results/mask/",
#     verticals=None,
#     texts=None,
# ):
#     img = np.array(img)

#     # Lấy tên tệp gốc của ảnh đầu vào
#     filename, file_ext = os.path.splitext(os.path.basename(img_file))
#     # Tạo đường dẫn đến tệp kết quả và hình ảnh kết quả
#     res_file = os.path.join(dirname, "res_" + filename + ".txt")
#     res_img_file = os.path.join(dirname, "res_" + filename + ".jpg")

#     # Tạo thư mục cắt hình ảnh dựa trên tên ảnh gốc
#     folder_path = "D:/Luan Van/Project/med-scan-backend/Training/Results/cut_images"

#     os.makedirs(folder_path, exist_ok=True)  # Tạo thư mục nếu chưa tồn tại

#     with open(res_file, "w") as f:
#         for i, box in enumerate(boxes):
#             poly = np.array(box).astype(np.int32).reshape((-1))
#             strResult = ",".join([str(p) for p in poly]) + "\r\n"
#             f.write(strResult)

#             poly = poly.reshape(-1, 2)
#             cv2.polylines(
#                 img, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=1
#             )
#             ptColor = (0, 255, 255)
#             xmin = min(poly[:, 0])
#             xmax = max(poly[:, 0])
#             ymin = min(poly[:, 1])
#             ymax = max(poly[:, 1])
#             width = xmax - xmin
#             height = ymax - ymin
#             pts = np.array([[xmin, ymax], [xmax, ymax], [xmax, ymin], [xmin, ymin]])

#             word = crop_images.crop(pts, img)

#             dir = os.path.join(
#                 "D:/Luan Van/Project/med-scan-backend/Training/Results/cut_images"
#             )
#             if not os.path.exists(dir):
#                 os.makedirs(dir)

#             try:
#                 file_name = os.path.join(
#                     "D:/Luan Van/Project/med-scan-backend/Training/Results/cut_images",
#                     filename + "_" + str(i) + ".jpg",
#                 )
#                 cv2.imwrite(file_name, word)
#             except:
#                 continue

#             if verticals is not None:
#                 if verticals[i]:
#                     ptColor = (255, 0, 0)

#             if texts is not None:
#                 font = cv2.FONT_HERSHEY_SIMPLEX
#                 font_scale = 0.5
#                 cv2.putText(
#                     img,
#                     "{}".format(texts[i]),
#                     (poly[0][0] + 1, poly[0][1] + 1),
#                     font,
#                     font_scale,
#                     (0, 0, 0),
#                     thickness=1,
#                 )
#                 cv2.putText(
#                     img,
#                     "{:.2f}".format(texts[i]),
#                     tuple(poly[0]),
#                     font,
#                     font_scale,
#                     (0, 255, 255),
#                     thickness=1,
#                 )

#     # Lưu hình ảnh kết quả
#     cv2.imwrite(res_img_file, img)
