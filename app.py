# from craft import CRAFT
from pymongo import MongoClient
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from data import guide_line
import subprocess
import csv
from bson import json_util


app = Flask(__name__)
# CORS(app, origins=["http://localhost:3000"])
CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})


# Kết nối đến MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['Med_Scan']
collection = db["Medicines"]

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

        subprocess.run(craft_command)

        subprocess.run(inference_command)

        # Đọc kết quả từ output.csv
        with open('D:/Luan Van/Project/med-scan-backend/results/output.csv', 'r', encoding='utf-8') as csv_file:
            csv_reader = csv.reader(csv_file)

            # Lọc ra các thông tin có nhãn "Medicine_Name"
            medicine_names = [
                row for row in csv_reader if row[-1] == "Medicine_Name"]

        if not medicine_names:
            return jsonify({'error': 'Không có thông tin thuốc được tìm thấy'})

        # Trích xuất các thông tin thuốc thành một danh sách
        medicine_info = []
        for row in medicine_names:
            # Định dạng thông tin thuốc
            medicine = {
                "ID": row[7],
                "Medicine_Name": row[8]
            }
            medicine_info.append(medicine)

        return jsonify({'medicine_info': medicine_info})


# Enpoint để tìm kiếm tên thuốc từ db
@app.route('/api/search_medicine', methods=['GET'])
def search_medicine():
    search_query = request.args.get('query')

    if not search_query:
        return jsonify({'message': 'Vui lòng cung cấp thông tin tìm kiếm.'}), 400

    # Sử dụng pymongo để truy vấn cơ sở dữ liệu và chỉ lấy trường "name"
    results = list(collection.find(
        {"name": {"$regex": search_query, "$options": 'i'}},
        {"id": 1, "name": 1, "longDescription": 1, "_id": 0}
    ))
    if not results:
        return jsonify({'message': 'Không tìm thấy kết quả phù hợp.'}), 404
    else:
        # Chuyển đổi kết quả thành JSON sử dụng json_util
        return jsonify(json_util.dumps(results))


# Enpoint để trả về "name" và "longDescription" dựa trên "id" của loại thuốc
@app.route('/api/medicine_details/<id>', methods=['GET'])
def get_medicine_details(id):
    # Tìm kiếm loại thuốc theo "id"
    medicine = collection.find_one(
        {"id": id}, {"name": 1, "longDescription": 1, "_id": 0})

    if medicine:
        return jsonify(medicine)
    else:
        return jsonify({'message': 'Không tìm thấy loại thuốc với ID cung cấp.'}), 404


# Enpoint để lấy Guided_line
@app.route('/api/guideline', methods=['GET'])
def get_guideline():
    return jsonify(guide_line)


if __name__ == '__main__':
    app.run(debug=True)
