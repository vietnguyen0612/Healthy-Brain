import ultralytics
from ultralytics import YOLO
import os
import cv2
import matplotlib.pyplot as plt
from tkinter import filedialog
from tkinter import Tk, Button, Label, PhotoImage, Canvas

# Hàm để chọn ảnh từ ổ đĩa
def choose_image():
    file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    return file_path

# Hàm khi nút dự đoán được nhấn
def predict_image():
    # Sử dụng hàm để chọn ảnh
    input_image_path = choose_image()

    # Kiểm tra xem người dùng đã chọn ảnh hay chưa
    if not input_image_path:
        print("Không có ảnh được chọn.")
    else:
        # Đường dẫn đến folder chứa model đã huấn luyện
        model_path = "./runs/detect/train16/weights/best.pt"
        trained_model = YOLO(model_path)

        # Dự đoán trên ảnh cụ thể
        predictions = trained_model.predict(source=input_image_path, conf=0.4, save_txt=True, save_conf=True)

        # Lấy đường dẫn lưu trữ dự đoán
        predictions_save_dir = predictions[0].save_dir + '/labels'

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
        img_pred = cv2.imread(input_image_path, 1)
        img_pred = cv2.cvtColor(img_pred, cv2.COLOR_BGR2RGB)
        filename = os.path.splitext(os.path.basename(input_image_path))[0]
        draw_bbox(predictions_save_dir, filename, img_pred)

        # Đọc và vẽ bounding box trên ảnh thực tế (nếu có)
        label_path = os.path.join('dataset/labels/test', f'{filename}.txt')
        if os.path.exists(label_path):
            img_real = cv2.imread(input_image_path, 1)
            img_real = cv2.cvtColor(img_real, cv2.COLOR_BGR2RGB)
            draw_bbox('dataset/labels/test', filename, img_real)
        else:
            img_real = None

        # Hiển thị ảnh dự đoán và ảnh thực tế (nếu có)
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].imshow(img_pred)
        axes[0].set_title('Predicted Image')
        axes[0].axis('off')

        if img_real is not None:
            axes[1].imshow(img_real)
            axes[1].set_title('Real Image')
            axes[1].axis('off')

        plt.tight_layout()
        plt.show()

# Tạo cửa sổ giao diện đầu tiên
root = Tk()
root.title("Image Prediction")

# Label và Canvas để hiển thị hình ảnh được chọn
image_label = Label(root, text="Selected Image:")
image_label.pack()

canvas = Canvas(root, width=300, height=300)
canvas.pack()

# Nút để chọn ảnh và thực hiện dự đoán
choose_button = Button(root, text="Choose Image", command=predict_image)
choose_button.pack()

root.mainloop()




##---------------------------------------------------------------------------------
# import ultralytics
# from ultralytics import YOLO
# import os
# import cv2
# import matplotlib.pyplot as plt

# # Đường dẫn đến folder chứa model đã huấn luyện
# model_path = "./runs/detect/train16/weights/best.pt"
# trained_model = YOLO(model_path)

# # Đường dẫn đến ảnh bạn muốn dự đoán
# input_image_path = "./axial_t1wce_2_class/images/test/00018_101.jpg"

# # Dự đoán trên ảnh cụ thể
# predictions = trained_model.predict(source=input_image_path, conf=0.4, save_txt=True, save_conf=True)

# # Lấy đường dẫn lưu trữ dự đoán
# predictions_save_dir = predictions[0].save_dir + '/labels'

# # Vẽ bounding box lên ảnh
# def draw_bbox(file_path, filename, img):
#     with open(os.path.join(file_path, f'{filename}.txt'), 'r') as f:
#         labels = f.readlines()
#         labels = labels[0].split(' ')
#         f.close()

#     tumor_class, x, y, w, h = int(labels[0]), float(labels[1]), float(labels[2]), float(labels[3]), float(labels[4])
#     x_pt1 = int((x - w/2) * img.shape[1])
#     y_pt1 = int((y - h/2) * img.shape[0])
#     x_pt2 = int((x + w/2) * img.shape[1])
#     y_pt2 = int((y + h/2) * img.shape[0])

#     if tumor_class == 0:
#         colour = (255, 0, 0)
#         label = 'Negative'
#     else:
#         colour = (0, 255, 0)
#         label = 'Positive'
#     if len(labels) > 5:
#         prob = float(labels[5])
#         prob = round(prob, 1)
#         prob = str(prob)
#         label = label + ' ' + prob

#     cv2.rectangle(img, (x_pt1, y_pt1), (x_pt2, y_pt2), colour, 1)
#     cv2.putText(img, label, (x_pt1, y_pt1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1)

# # Đọc và vẽ bounding box trên ảnh dự đoán
# img_pred = cv2.imread(input_image_path, 1)
# img_pred = cv2.cvtColor(img_pred, cv2.COLOR_BGR2RGB)
# filename = os.path.splitext(os.path.basename(input_image_path))[0]
# draw_bbox(predictions_save_dir, filename, img_pred)

# # Đọc và vẽ bounding box trên ảnh thực tế (nếu có)
# label_path = os.path.join('dataset/labels/test', f'{filename}.txt')
# if os.path.exists(label_path):
#     img_real = cv2.imread(input_image_path, 1)
#     img_real = cv2.cvtColor(img_real, cv2.COLOR_BGR2RGB)
#     draw_bbox('dataset/labels/test', filename, img_real)
# else:
#     img_real = None

# # Hiển thị ảnh dự đoán và ảnh thực tế (nếu có)
# fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# axes[0].imshow(img_pred)
# axes[0].set_title('Predicted Image')
# axes[0].axis('off')

# if img_real is not None:
#     axes[1].imshow(img_real)
#     axes[1].set_title('Real Image')
#     axes[1].axis('off')

# plt.tight_layout()
# plt.show()



###-----------------------------------------------------------------------------------------------------------------

# import ultralytics
# from ultralytics import YOLO
# import os
# import random
# import cv2
# import matplotlib.pyplot as plt
# trained_model = YOLO("./runs/detect/train16" + '/weights/best.pt')
# predictions = trained_model.predict(
#     source="./axial_t1wce_2_class/images/test",
#     conf=0.4, save_txt=True, save_conf=True)

# predictions_save_dir = predictions[0].save_dir + '/labels'

# def draw_bbox(file_path, filename, img):
#     with open(os.path.join(file_path, f'{filename}.txt'),'r') as f:
#         labels = f.readlines()
#         labels = labels[0].split(' ')
#         print(labels)
#         f.close()

#     tumor_class, x, y, w, h = int(labels[0]), float(labels[1]), float(labels[2]), float(labels[3]), float(labels[4])
#     x_pt1 = int((x - w/2) * img.shape[1])
#     y_pt1 = int((y - h/2) * img.shape[0])
#     x_pt2 = int((x + w/2) * img.shape[1])
#     y_pt2 = int((y + h/2) * img.shape[0])

#     if tumor_class == 0:
#         colour = (255, 0, 0)
#         label = 'Negative'
#     else:
#         colour = (0, 255, 0)
#         label = 'Positive'
#     if len(labels) > 5:
#         prob = float(labels[5])
#         prob = round(prob, 1)
#         prob = str(prob)
#         label = label + ' ' + prob

#     cv2.rectangle(img, (x_pt1, y_pt1), (x_pt2, y_pt2), colour, 1)
#     cv2.putText(img, label, (x_pt1, y_pt1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1)


# files = os.listdir(predictions_save_dir)
# random_file = random.choice(files)
# random_file = os.path.splitext(random_file)[0]

# img_pred = cv2.imread(os.path.join('dataset/images/test', f'{random_file}.jpg'), 1)
# img_pred = cv2.cvtColor(img_pred, cv2.COLOR_BGR2RGB)
# draw_bbox(predictions_save_dir, random_file, img_pred)

# img_real = cv2.imread(os.path.join('dataset/images/test', f'{random_file}.jpg'), 1)
# img_real = cv2.cvtColor(img_real, cv2.COLOR_BGR2RGB)
# draw_bbox('dataset/labels/test', random_file, img_real)

# fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# axes[0].imshow(img_pred)
# axes[0].set_title('Predicted Image')
# axes[0].axis('off')

# axes[1].imshow(img_real)
# axes[1].set_title('Real Image')
# axes[1].axis('off')

# plt.tight_layout()
# plt.show()