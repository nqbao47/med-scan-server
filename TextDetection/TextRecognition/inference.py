import csv
import os
from pathlib import Path

from PIL import Image
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor

# Load the VietOCR model
config = Cfg.load_config_from_name("vgg_transformer")
config["export"] = "transformerocr_checkpoint.pth"
config["device"] = "cpu"
config["predictor"]["beamsearch"] = False

# Exclude specific characters and only allow alphanumeric characters
# config = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

detector = Predictor(config)

# Đường dẫn đến thư mục chứa hình ảnh cắt nhỏ
img_dir = "D:/Luan Van/Project/med-scan-backend/processing/Craft_model/cut_images/"

# Đường dẫn chứa box
box_file = "D:/Luan Van/Project/med-scan-backend/processing/OCR_model/box"

# Tệp CSV cần xử lý (đường dẫn đầy đủ)
csv_file_to_process = (
    "D:/Luan Van/Project/med-scan-backend/processing/Craft_model/coordinates/1.csv"
)

# Kiểm tra xem tệp CSV tồn tại hay không
if os.path.isfile(csv_file_to_process):
    with open(csv_file_to_process, "r", encoding="utf-8") as csvinput:
        # Read the content of the CSV file into a list of rows
        rows = list(csv.reader(csvinput))

    # Tách tên tệp CSV để lấy tiền tố
    csv_filename = Path(csv_file_to_process).stem
    # prefix = csv_filename + "_"
    # print(f"CSV File: {csv_file_to_process}, Prefix: {prefix}")
    # Thư mục để lưu tệp kết quả CSV
    output_dir = "D:/Luan Van/Project/med-scan-backend/processing/OCR_model/"
    output_csv_file = os.path.join(output_dir, f"1.csv")

    # Initialize an index variable to keep track of the current row
    i = 0

    for img_file in sorted(
        os.listdir(img_dir), key=lambda x: int(os.path.splitext(x)[0])
    ):
        try:
            if img_file.endswith(".jpg") and img_file[:-4].isdigit():
                img_path = os.path.join(img_dir, img_file)
                n = Image.open(img_path)
                cv_img = str(detector.predict(n))
                print(f"Processing Image: {img_file}, CV Result: {cv_img}")

                # Append the cv_img value to the current row
                rows[i].append(cv_img)

                i += 1
        except (ValueError, IndexError):
            # Handle invalid file names (skip or log, depending on your needs)
            print(f"Skipping invalid file: {img_file}")

    # Write the modified rows to the new CSV file
    with open(output_csv_file, "w", encoding="utf-8", newline="") as csvoutput:
        writer = csv.writer(csvoutput)
        writer.writerows(rows)

else:
    print(f"CSV File not found: {csv_file_to_process}")


# import csv
# import glob
# import os
# from pathlib import Path

# from PIL import Image
# from vietocr.tool.config import Cfg
# from vietocr.tool.predictor import Predictor

# # Load the VietOCR model
# config = Cfg.load_config_from_name("vgg_transformer")
# config["export"] = "transformerocr_checkpoint.pth"
# config["device"] = "cpu"
# config["predictor"]["beamsearch"] = False
# detector = Predictor(config)

# # Đường dẫn đến thư mục chứa tệp CSV và hình ảnh cắt nhỏ
# csv_dir = "D:/Luan Van/Project/med-scan-backend/Training/Results/coordinates/"
# img_dir = "D:/Luan Van/Project/med-scan-backend/Training/Results/cut_images/"

# # Đường dẫn chứa box
# box_file = "D:/Luan Van/Project/med-scan-backend/Training/Results/box/"

# # Lặp qua tất cả các tệp CSV trong thư mục csv_files
# for csv_file in glob.glob(f"{csv_dir}*.csv"):
#     with open(csv_file, "r", encoding="utf-8") as csvinput:
#         # Read the content of the CSV file into a list of rows
#         rows = list(csv.reader(csvinput))

#     # Tách tên tệp CSV để lấy tiền tố
#     csv_filename = Path(csv_file).stem
#     prefix = csv_filename + "_"
#     print(f"CSV File: {csv_file}, Prefix: {prefix}")

#     # Initialize an index variable to keep track of the current row
#     i = 0

#     # Lặp qua các tệp ảnh cắt nhỏ trong thư mục img_dir
#     for img_file in sorted(
#         os.listdir(img_dir), key=lambda x: int(x.split("_")[1].split(".")[0])
#     ):
#         if img_file.startswith(prefix) and img_file.endswith(".jpg"):
#             img_path = os.path.join(img_dir, img_file)
#             n = Image.open(img_path)
#             cv_img = str(detector.predict(n))
#             print(f"Processing Image: {img_file}, CV Result: {cv_img}")

#             # Append the cv_img value to the current row
#             rows[i].append(cv_img)
#             # print(f"Row {i + 1}: {rows[i]}")

#             i = i + 1

#     # Write the modified rows to the new CSV file
#     with open(
#         f"{box_file}{csv_filename}.csv", "w", encoding="utf-8", newline=""
#     ) as csvoutput:
#         writer = csv.writer(csvoutput)
#         writer.writerows(rows)
