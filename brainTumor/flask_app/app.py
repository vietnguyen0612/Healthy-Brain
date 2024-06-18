
from flask import Flask, render_template, request, redirect, url_for, jsonify, session, flash

import json
import ultralytics
from ultralytics import YOLO
import os
import cv2
import matplotlib.pyplot as plt
from flask_mysqldb import MySQL
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
from PIL import Image
import io
import base64
import shutil
import uuid
from MySQLdb import IntegrityError
import MySQLdb.cursors


app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Đặt secret key cho session

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'braintumor'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'



mysql = MySQL(app)

# Đường dẫn đến folder chứa model đã huấn luyện
model_path = "../runs/detect/train16/weights/best.pt"
trained_model = YOLO(model_path)


@app.route('/')
def index():
    if 'username' in session:
        role = session.get('role')
        if role =='doctor':
            return render_template('index.html')
        elif role == 'admin':
            return redirect(url_for('doctors'))
        else:
            return render_template('error.html', message='Unknown role')
    else:
        return redirect(url_for('login'))

@app.route('/predict')
def predict():
    if 'username' in session:
        role = session.get('role')
        if role =='doctor':
            return render_template('predict.html')
        elif role == 'admin':
            return redirect(url_for('doctors'))
        else:
            return render_template('error.html', message='Unknown role')
    else:
        return redirect(url_for('login'))
#login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        cursor = mysql.connection.cursor()
        cursor.execute('SELECT * FROM users WHERE username = %s AND password = %s', (username, password))
        user = cursor.fetchone()
        cursor.close()
        if user:
            # Lưu thông tin người dùng vào session, bao gồm cả full_name
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['full_name'] = user['full_name']
            session['role'] = user['role']

            return redirect(url_for('index'))
        else:
            return render_template('login.html', error='Invalid username or password')
    return render_template('login.html')

#Logout
@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    session.pop('full_name', None)
    session.pop('role', None)
    return redirect(url_for('login'))

#hiển thị danh sách bác sĩ
@app.route('/doctors')
def doctors():
    # Kiểm tra vai trò của người dùng, chỉ admin mới có quyền xem danh sách bác sĩ
    if 'username' in session and session['role'] == 'admin':
        cursor = mysql.connection.cursor()
        cursor.execute('SELECT * FROM users')
        doctors = cursor.fetchall()
        cursor.close()
        return render_template('admin.html', doctors=doctors)
    else:
        return render_template('error.html', message='Permission denied')

@app.route('/doctors_manage')
def doctors_manage():
    # Kiểm tra vai trò của người dùng, chỉ admin mới có quyền xem danh sách bác sĩ
    if 'username' in session and session['role'] == 'admin':
        cursor = mysql.connection.cursor()
        cursor.execute('SELECT * FROM users')
        doctors = cursor.fetchall()
        cursor.close()
        return render_template('doctor_manage.html', doctors=doctors)
    else:
        return render_template('error.html', message='Permission denied')

@app.route('/patients_manage')
def patients_manage():
    # Kiểm tra vai trò của người dùng, chỉ admin mới có quyền xem danh sách bác sĩ
    if 'username' in session and session['role'] == 'admin':
        cursor = mysql.connection.cursor()
        cursor.execute('SELECT * FROM patient')
        patients = cursor.fetchall()
        cursor.close()
        return render_template('patient_manage.html', patients=patients)
    else:
        return render_template('error.html', message='Permission denied')

# thêm bác sĩ mới
@app.route('/add_doctor', methods=['GET', 'POST'])
def add_doctor():
    # Kiểm tra vai trò của người dùng, chỉ admin mới có quyền thêm bác sĩ
    if 'username' in session and session['role'] == 'admin':
        if request.method == 'POST':
            # Lấy thông tin từ form
            username = request.form['username']
            password = request.form['password']
            role = 'doctor'  # Mặc định role là doctor
            full_name = request.form['full_name']
            email = request.form['email']

            cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
            
            try:
                # Kiểm tra xem username đã tồn tại hay chưa
                cursor.execute('SELECT COUNT(*) as count FROM users WHERE username = %s', (username,))
                result = cursor.fetchone()
                if result['count'] > 0:
                    return render_template('add_doctor.html', error="Username already exists.")
                
                # Thêm bác sĩ vào cơ sở dữ liệu
                cursor.execute(
                    'INSERT INTO users (username, password, role, full_name, email) VALUES (%s, %s, %s, %s, %s)',
                    (username, password, role, full_name, email)
                )
                mysql.connection.commit()
                return redirect(url_for('doctors_manage'))

            except IntegrityError as e:
                if e.args[0] == 1062:  # Duplicate entry error code
                    return render_template('add_doctor.html', error="Username already exists.")
                else:
                    return render_template('error.html', message="An error occurred. Please try again.")
            finally:
                cursor.close()
        return render_template('add_doctor.html')
    else:
        return render_template('error.html', message='Từ chối quyền truy cập')
# sửa thông tin bác sĩ
@app.route('/edit_doctor/<int:doctor_id>', methods=['GET', 'POST'])
def edit_doctor(doctor_id):
    # Kiểm tra vai trò của người dùng, chỉ admin mới có quyền sửa thông tin bác sĩ
    if 'username' in session and session['role'] == 'admin':
        cursor = mysql.connection.cursor()
        cursor.execute('SELECT * FROM users WHERE id = %s', (doctor_id,))
        doctor = cursor.fetchone()
        cursor.close()

        if doctor:
            if request.method == 'POST':
                # Lấy thông tin từ form
                username = request.form['username']
                full_name = request.form['full_name']
                email = request.form['email']
                role = request.form['role']
                # Cập nhật thông tin bác sĩ trong cơ sở dữ liệu
                cursor = mysql.connection.cursor()
                cursor.execute('UPDATE users SET username = %s, full_name = %s, email = %s, role = %s WHERE id = %s',
                               (username, full_name, email, role, doctor_id))
                mysql.connection.commit()
                cursor.close()

                return redirect(url_for('doctors'))

            return render_template('edit_doctor.html', doctor=doctor)
        else:
            return render_template('error.html', message='Doctor not found')
    else:
        return render_template('error.html', message='Permission denied')
# xóa bác sĩ
@app.route('/delete_doctor/<int:doctor_id>')
#Danh sách bệnh nhân
@app.route('/patients_manage_by_doctor')
def patients_manage_by_doctor():
    # Kiểm tra vai trò của người dùng, chỉ bác sĩ mới có quyền xem danh sách bệnh nhân
    if 'username' in session and session['role'] == 'doctor':
        doctor_id = session.get('user_id')  # Lấy id của user từ session
        cursor = mysql.connection.cursor()
        cursor.execute('SELECT * FROM patient WHERE id_doctor_ex = %s', (doctor_id,))
        patients = cursor.fetchall()
        cursor.close()
        return render_template('patient_manage_by_doctor.html', patients=patients)
    else:
        return render_template('error.html', message='Permission denied')
       
def delete_doctor(doctor_id):
    # Kiểm tra vai trò của người dùng, chỉ admin mới có quyền xóa bác sĩ
    if 'username' in session and session['role'] == 'admin':
        cursor = mysql.connection.cursor()
        cursor.execute('SELECT * FROM users WHERE id = %s', (doctor_id,))
        doctor = cursor.fetchone()

        if doctor:
            # Xóa bác sĩ từ cơ sở dữ liệu
            cursor.execute('DELETE FROM users WHERE id = %s', (doctor_id,))
            mysql.connection.commit()
            cursor.close()

            return redirect(url_for('doctors_manage'))
        else:
            return render_template('error.html', message='Doctor not found')
    else:
        return render_template('error.html', message='Permission denied')
#xóa bệnh nhân
@app.route('/delete_patient/<int:patient_id>')
def delete_patient(patient_id):
    # Kiểm tra vai trò của người dùng, chỉ admin và bác sĩ mới có quyền xóa bệnh nhân
    if 'username' in session and (session['role'] == 'admin' or session['role'] == 'doctor'):
        cursor = mysql.connection.cursor()
        try:
            cursor.execute('SELECT * FROM patient WHERE id = %s', (patient_id,))
            patient = cursor.fetchone()

            if patient:
                # Xóa bệnh nhân từ cơ sở dữ liệu
                cursor.execute('DELETE FROM patient WHERE id = %s', (patient_id,))
                mysql.connection.commit()
                flash('Patient deleted successfully', 'success')
                if session['role'] == 'admin':
                    return redirect(url_for('patients_manage'))
                elif session['role'] =="doctor":
                    return redirect(url_for('patients_manage_by_doctor'))
            else:
                flash('Patient not found', 'error')
                if session['role'] == 'admin':
                    return redirect(url_for('patients_manage'))
                elif session['role'] =="doctor":
                    return redirect(url_for('patients_manage_by_doctor'))
        except Exception as e:
            print(f"Error: {str(e)}")
            flash('An error occurred while trying to delete the patient', 'error')
            return redirect(url_for('patients_manage'))
        finally:
            cursor.close()
    else:
        return render_template('error.html', message='Permission denied')
# Hàm tính diện tích của bounding box
def calculate_bbox_area(x, y, w, h, img_shape):
    x_pt1 = int((x - w/2) * img_shape[1])
    y_pt1 = int((y - h/2) * img_shape[0])
    x_pt2 = int((x + w/2) * img_shape[1])
    y_pt2 = int((y + h/2) * img_shape[0])

    bbox_area = (x_pt2 - x_pt1) * (y_pt2 - y_pt1)
    return bbox_area

@app.route('/', methods=['POST'])
def upload_file():
    # Kiểm tra xem có file được chọn hay không
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')

    file = request.files['file']

    # Kiểm tra xem có file nào được chọn hay không
    if file.filename == '':
        return render_template('index.html', error='No selected file')

    # Lưu file và lấy đường dẫn
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    # Kiểm tra hành động của người dùng
    action = request.form.get('action')

    if action == 'annotate':
        # Nếu người dùng chọn vẽ, thì chuyển hướng đến trang vẽ bounding box
        return redirect(url_for('annotate_page'))

    # Dự đoán trên ảnh cụ thể
    predictions = trained_model.predict(source=file_path, conf=0.4, save_txt=True, save_conf=True)
    
    # Lấy đường dẫn lưu trữ dự đoán
    predictions_save_dir = predictions[0].save_dir + '/labels'
    # Tính diện tích của bounding box
    filename = os.path.splitext(os.path.basename(file_path))[0]
    with open(os.path.join(predictions_save_dir, f'{filename}.txt'), 'r') as f:
        labels = f.readlines()
        labels = labels[0].split(' ')
        f.close()
    tumor_class, x, y, w, h = int(labels[0]), float(labels[1]), float(labels[2]), float(labels[3]), float(labels[4])
    img_pred = cv2.imread(file_path, 1)
    tumor_area = calculate_bbox_area(x, y, w, h, img_pred.shape)
    # Tính vị trí bounding box
    center_x_bbox = int(x * img_pred.shape[1] + (w * img_pred.shape[1]) / 2)
    center_x_image = img_pred.shape[1] / 2

    if center_x_bbox > center_x_image:
        bounding_box_position = 'Bên phải'
    elif center_x_bbox < center_x_image:
        bounding_box_position = 'Bên trái'
    else:
        bounding_box_position = 'Giữa'
    print(f"center_x_bbox: {center_x_bbox}")
    print(f"center_x_image: {center_x_image}")
    print(f"Bounding box position: {bounding_box_position}")

    # Vẽ bounding box lên ảnh
    def draw_bbox(file_path, filename, img):
        with open(os.path.join(file_path, f'{filename}.txt'), 'r') as f:
            labels = f.readlines()
            labels = labels[0].split(' ')
            f.close()

        tumor_class, x, y, w, h = int(labels[0]), float(labels[1]), float(labels[2]), float(labels[3]), float(labels[4])
        x_pt1 = int((x - w/2) * img.shape[1])
        y_pt1 = int((y - h/2) * img.shape[0])
        x_pt2 = int((x + w/2) * img.shape[1])
        y_pt2 = int((y + h/2) * img.shape[0])

        if tumor_class == 0:
            colour = (255, 0, 0)
            label = 'Negative'
        else:
            colour = (0, 255, 0)
            label = 'Positive'
        if len(labels) > 5:
            prob = float(labels[5])
            prob = round(prob, 1)
            prob = str(prob)
            label = label + ' ' + prob

        cv2.rectangle(img, (x_pt1, y_pt1), (x_pt2, y_pt2), colour, 1)
        cv2.putText(img, label, (x_pt1, y_pt1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1)

    # Đọc và vẽ bounding box trên ảnh dự đoán
    img_pred = cv2.imread(file_path, 1)
    img_pred = cv2.cvtColor(img_pred, cv2.COLOR_BGR2RGB)
    filename = os.path.splitext(os.path.basename(file_path))[0]
    draw_bbox(predictions_save_dir, filename, img_pred)

    # Lưu ảnh với bounding box đã vẽ
    output_path_original = os.path.join('static', 'original_image.jpg')
    plt.imsave(output_path_original, cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB))

    output_path_result = os.path.join('static', 'result_image.jpg')
    plt.imsave(output_path_result, img_pred)

    # Trả về trang hiển thị kết quả
    return render_template('result.html', original_image_path=output_path_original, result_image_path=output_path_result,bounding_box_size=tumor_area, bounding_box_position=bounding_box_position)
# Route cho trang vẽ bounding box
#_____________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
@app.route('/save_diagnosis', methods=['GET', 'POST'])
def save_diagnosis():
    if request.method == 'POST':
        # Lấy thông tin từ form
        patient_name = request.form['patient_name']
        original_image_path = request.form['original_image_path']
        result_image_path = request.form['result_image_path']
        bounding_box_size = request.form['bounding_box_size']
        bounding_box_position = request.form['bounding_box_position']

        # Lưu thông tin vào cơ sở dữ liệu hoặc thực hiện các hành động khác theo nhu cầu của bạn

        return render_template('info_patient.html', patient_name=patient_name,
                               original_image_path=original_image_path,
                               result_image_path=result_image_path,
                               bounding_box_size=bounding_box_size,
                               bounding_box_position=bounding_box_position)
    elif request.method == 'GET':
        # Xử lý logic khi có yêu cầu GET
        original_image_path = request.args.get('original_image_path')
        result_image_path = request.args.get('result_image_path')
        bounding_box_size = request.args.get('bounding_box_size')
        bounding_box_position = request.args.get('bounding_box_position')

        # Hiển thị trang nhập thông tin với thông tin đã nhận được
        return render_template('info_patient.html', original_image_path=original_image_path,
                               result_image_path=result_image_path,
                               bounding_box_size=bounding_box_size,
                               bounding_box_position=bounding_box_position)
    

    return render_template('error.html', message='Invalid request')



@app.route('/save_patient',methods=['GET', 'POST'])
def save_patient():
    if 'username' in session and session['role'] == 'doctor':
        if request.method == 'POST':

            image_path = './static/result_image.jpg'
            
            # Đọc dữ liệu ảnh gốc
            with open(image_path, 'rb') as f:
                image_data = f.read()
            
            # Chuyển đổi ảnh sang base64
            base64_image = base64.b64encode(image_data).decode('utf-8')

            # Tạo UID mới
            uid = str(uuid.uuid4())

            # Đường dẫn tới thư mục đích
            destination_folder = './static/result'
            
            # Đảm bảo rằng thư mục đích tồn tại
            os.makedirs(destination_folder, exist_ok=True)
            
            # Đường dẫn tới ảnh mới
            destination_image_path = os.path.join(destination_folder, f'{uid}.jpg')

            # Copy và đổi tên ảnh
            shutil.copy(image_path, destination_image_path)

            # Thu thập dữ liệu từ form
            patient_name = request.form['patient_name']
            patient_age = request.form['patient_age']
            patient_address = request.form['patient_address']
            tumor_size = request.form['bounding_box_size']
            tumor_position = request.form['bounding_box_position']
            additional_info = request.form['additional_info']
            result_image_data = destination_image_path.replace("b'.", ".").replace("jpg'", "jpg")
            
            # Lấy id bác sĩ từ session  
            id_doctor_ex = session.get('user_id')

            # Lưu thông tin vào cơ sở dữ liệu hoặc thực hiện các hành động khác theo nhu cầu của bạn
            cursor = mysql.connection.cursor()
            insert_query = """INSERT INTO patient (name, age, address, image, tumor_size, tumor_position, info, id_doctor_ex) 
                              VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"""
            cursor.execute(insert_query, (patient_name, patient_age, patient_address, result_image_data, tumor_size, tumor_position, additional_info, id_doctor_ex))
            mysql.connection.commit()
            cursor.close()
            
        return redirect(url_for('patients_manage_by_doctor'))
    else:
        return render_template('error.html', message='Từ chối quyền truy cập')
    
       
@app.route('/annotate_page')
def annotate_page():
    return render_template('annotate.html')

# Route cho việc vẽ bounding box
@app.route('/annotate', methods=['POST'])
def annotate_image():
    # Lấy thông tin bounding box từ request
    bounding_boxes_json = request.form.get('bounding_boxes')
    bounding_boxes = json.loads(bounding_boxes_json)

    # Kiểm tra xem trường image_path có trong request không
    if 'image_path' not in request.form:
        return jsonify({'status': 'error', 'message': 'Missing image_path in the request'})

    # Lấy tên file ảnh để tạo tên tương ứng cho tệp văn bản
    image_path = request.form.get('image_path')
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    # Lưu thông tin bounding box vào tệp văn bản trong thư mục "labels"
    labels_dir = './annotate/labels'
    label_file_path = os.path.join(labels_dir, image_name + '.txt')
    with open(label_file_path, 'w') as label_file:
        for box in bounding_boxes:
            label_file.write(f"{box[0]} {box[1]} {box[2]} {box[3]}\n")

    # Trả về phản hồi JSON (nếu cần)
    return jsonify({'status': 'success'})
if __name__ == '__main__':
    app.run(debug=True)
