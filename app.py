# from craft import CRAFT
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from data import guide_line
import subprocess
import csv


app = Flask(__name__)
# CORS(app, origins=["http://localhost:3000"])
CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})


# Thư mục để lưu ảnh
UPLOAD_FOLDER = 'data'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Xử lý ảnh request và reply
@app.route('/api/upload_image', methods=['POST'])
def upload_image():
    # Đảm bảo rằng thư mục UPLOAD_FOLDER (data) tồn tại
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    # Xoá tất cả các tệp trong thư mục UPLOAD_FOLDER
    for file_name in os.listdir(app.config['UPLOAD_FOLDER']):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Không thể xoá tệp {file_path}: {e}")

    uploaded_image = request.files['image']

    if uploaded_image:
        # Đổi tên cho hình ảnh thành "1.jpg" mặc định
        uploaded_image.filename = '1.jpg'
        # Lưu hình ảnh vào thư mục ""data""
        image_path = os.path.join(
            app.config['UPLOAD_FOLDER'], uploaded_image.filename)
        uploaded_image.save(image_path)

        # Gọi CRAFT và inference.py bằng subprocess
        craft_command = ['python', 'CRAFT/pipeline.py',
                         '--test_folder', app.config['UPLOAD_FOLDER']]
        inference_command = [
            'python', 'TextDetection/TextRecognition/inference.py']

        # Thực hiện lệnh CRAFT
        subprocess.run(craft_command)

        # Thực hiện lệnh inference.py
        subprocess.run(inference_command)

        # Đọc kết quả từ output.csv
        with open('D:/Luan Van/Project/med-scan-backend/results/output.csv', 'r', encoding='utf-8') as csv_file:
            csv_reader = csv.reader(csv_file)

            # Lọc ra các thông tin có nhãn "Medicine_Name"
            medicine_names = [
                row for row in csv_reader if row[-1] == "Medicine_Name"]

            # Chuyển danh sách kết quả thành một chuỗi
            result = "\n".join(",".join(row) for row in medicine_names)

            if result:
                return jsonify({'result': result})
            else:
                return jsonify({'error': 'Không có thông tin thuốc được tìm thấy'})


# Enpoint để lấy Guided_line
@app.route('/api/guideline', methods=['GET'])
def get_guideline():
    return jsonify(guide_line)


if __name__ == '__main__':
    app.run(debug=True)
