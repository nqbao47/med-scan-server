# from craft import CRAFT
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from data import guide_line
# from CRAFT.pipeline import run_craft
# from TextDetection.TextRecognition.inference import run_text_recognition

app = Flask(__name__)
# CORS(app, origins=["http://localhost:3000"])
CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})

# # Đường dẫn đến môi trường ảo Conda
# conda_env_path = "C:/Users/baong/miniconda3/envs/baoenv"


# def activate_conda_environment():
#     os.system(f"conda activate {conda_env_path}")


# Thư mục để lưu ảnh
UPLOAD_FOLDER = 'data'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Xử lý ảnh request và reply


@app.route('/api/upload_image', methods=['POST'])
def upload_image():
    # Đảm bảo rằng thư mục UPLOAD_FOLDER tồn tại
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

    #     # Bước 1: Kích hoạt môi trường Conda
    #     activate_conda_environment()

    #     # Bước 2: Chạy CRAFT trên ảnh
    #     run_craft(image_path)

    #     # Bước 3: Chạy Text-Detection
    #     run_text_recognition()

    #     # # Bước 4: Chạy GCN trên kết quả
    #     # run_gcn()

        return jsonify({'message': 'Xử lý hoàn tất'})

    return jsonify({'error': 'Không có hình ảnh được tải lên'})

# Enpoint để lấy Guided_line


@app.route('/api/guideline', methods=['GET'])
def get_guideline():
    return jsonify(guide_line)


if __name__ == '__main__':
    app.run(debug=True)
