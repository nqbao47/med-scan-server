# from craft import CRAFT
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from data import guide_line
import subprocess

app = Flask(__name__)
# CORS(app, origins=["http://localhost:3000"])
CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})

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
        # Lưu hình ảnh vào thư mục ""data""
        image_path = os.path.join(
            app.config['UPLOAD_FOLDER'], uploaded_image.filename)
        uploaded_image.save(image_path)

        # Gọi tệp pipeline.py từ thư mục CRAFT để xử lý ảnh
        command_craft = ['python', 'CRAFT/pipeline.py',
                         '--image_path', image_path]
        process_craft = subprocess.Popen(
            command_craft, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout_craft, stderr_craft = process_craft.communicate()

        if process_craft.returncode == 0:
            # Xử lý thành công, stdout_craft chứa kết quả
            craft_result = stdout_craft.decode('utf-8')
        else:
            # Xử lý không thành công, stderr_craft chứa thông báo lỗi
            return jsonify({'message': 'Lỗi trong quá trình xử lý ảnh từ CRAFT: ' + stderr_craft.decode('utf-8')})

        # Gọi tệp inference.py từ thư mục Text-Detection/TextRecognition để trích xuất văn bản
        command_text_detection = [
            'python', 'Text-Detection/TextRecognition/inference.py', '--input', craft_result]
        process_text_detection = subprocess.Popen(
            command_text_detection, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout_text_detection, stderr_text_detection = process_text_detection.communicate()

        if process_text_detection.returncode == 0:
            # Xử lý thành công, stdout_text_detection chứa kết quả trích xuất văn bản
            text_result = stdout_text_detection.decode('utf-8')
        else:
            # Xử lý không thành công, stderr_text_detection chứa thông báo lỗi
            return jsonify({'message': 'Lỗi trong quá trình trích xuất văn bản từ Text-Detection: ' + stderr_text_detection.decode('utf-8')})

        # Trả về kết quả cho ứng dụng React
        response_data = {
            'message': 'Hình ảnh đã được xử lý và văn bản đã được trích xuất thành công.',
            'image_path': image_path,
            'text_result': text_result
        }
        return jsonify(response_data)
    else:
        return jsonify({'message': 'Không có hình ảnh được gửi lên.'})


# Enpoint để lấy Guided_line


@app.route('/api/guideline', methods=['GET'])
def get_guideline():
    return jsonify(guide_line)


if __name__ == '__main__':
    app.run(debug=True)
