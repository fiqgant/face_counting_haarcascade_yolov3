import cv2
import numpy as np

# Inisialisasi detektor Haarcascade dan YOLO-Face
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load YOLO-Face
net = cv2.dnn.readNet("models/yolov3-wider_16000.weights", "models/yolov3-face.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Webcam
# cap = cv2.VideoCapture(0)

# Video
cap = cv2.VideoCapture('video.mp4')

def is_overlap(box1, box2):
    (x1, y1, w1, h1) = box1
    (x2, y2, w2, h2) = box2

    # Hitung batas bounding box
    x1_min, y1_min = x1, y1
    x1_max, y1_max = x1 + w1, y1 + h1
    x2_min, y2_min = x2, y2
    x2_max, y2_max = x2 + w2, y2 + h2

    # Cek apakah bounding box tumpang tindih
    if (x1_max < x2_min or x1_min > x2_max or y1_max < y2_min or y1_min > y2_max):
        return False
    return True

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame untuk mempercepat deteksi
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    # Konversi frame ke grayscale untuk Haarcascade
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

    # Deteksi wajah menggunakan Haarcascade
    faces_haar = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Deteksi wajah menggunakan YOLO-Face
    blob = cv2.dnn.blobFromImage(small_frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    faces_yolo = []
    confidences = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            confidence = scores[0]  # YOLO-Face hanya memiliki satu kelas yaitu 'face'
            if confidence > 0.5:
                center_x = int(detection[0] * small_frame.shape[1])
                center_y = int(detection[1] * small_frame.shape[0])
                w = int(detection[2] * small_frame.shape[1])
                h = int(detection[3] * small_frame.shape[0])
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                faces_yolo.append((x, y, w, h))
                confidences.append(float(confidence))

    # Menggunakan Non-Maximum Suppression untuk mengurangi kotak yang berlebihan
    indices = cv2.dnn.NMSBoxes(faces_yolo, confidences, 0.5, 0.4)
    faces_yolo = [faces_yolo[i] for i in indices.flatten()]

    # Periksa tumpang tindih antara bounding box Haarcascade dan YOLO
    counted_faces = []
    for haar_face in faces_haar:
        (x, y, w, h) = haar_face
        # Sesuaikan bounding box ke ukuran frame asli
        haar_face = (x*2, y*2, w*2, h*2)
        for yolo_face in faces_yolo:
            # Sesuaikan bounding box ke ukuran frame asli
            yolo_face = (yolo_face[0]*2, yolo_face[1]*2, yolo_face[2]*2, yolo_face[3]*2)
            if is_overlap(haar_face, yolo_face):
                counted_faces.append(haar_face)
                break

    # Hitung jumlah wajah yang terdeteksi
    total_faces = len(counted_faces)

    # Tampilkan jumlah wajah yang terdeteksi pada frame
    cv2.putText(frame, f'Jumlah wajah yang terdeteksi: {total_faces}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Gambarkan kotak di sekitar wajah yang terdeteksi oleh Haarcascade dan YOLO-Face
    for (x, y, w, h) in faces_haar:
        # Sesuaikan bounding box ke ukuran frame asli
        x, y, w, h = x*2, y*2, w*2, h*2
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    for (x, y, w, h) in faces_yolo:
        # Sesuaikan bounding box ke ukuran frame asli
        x, y, w, h = x*2, y*2, w*2, h*2
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Tampilkan frame
    cv2.imshow('Face Detection', frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Rilis video capture dan tutup semua jendela OpenCV
cap.release()
cv2.destroyAllWindows()
