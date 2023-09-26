import os
from collections import defaultdict
import cv2
from ultralytics import YOLO
import face_recognition
import pandas as pd
import random
import csv
import time
from sklearn.cluster import KMeans
import shutil
import numpy as np
import pickle
import re
import sys

start_time = time.time()
####################################여기 영상 경로만 맞춰주세요.
# video_path = './source_videos/ys_theking.mp4'  
video_path = sys.argv[1]
##################################################################
cap = cv2.VideoCapture(video_path)

# 영상 제목 추출
video_name = os.path.splitext(os.path.basename(video_path))[0]

output_folders = './output_folders'
os.makedirs(output_folders, exist_ok=True)

# YOLO 모델 로드
model = YOLO('./ptfiles/v8_m_100_batch30.pt')




# 이미 존재하는 폴더를 삭제하고 새로운 폴더를 생성
new_output_dir = os.path.join(output_folders, video_name)

if os.path.exists(new_output_dir):
    # 기존 폴더가 존재하는 경우 삭제
    shutil.rmtree(new_output_dir)

os.makedirs(new_output_dir)

# faces 폴더 생성
faces_folder = os.path.join(new_output_dir, 'faces')
os.makedirs(faces_folder, exist_ok=True)

# faces/before_cluster 폴더 생성
before_cluster_folder = os.path.join(faces_folder, 'before_cluster')
os.makedirs(before_cluster_folder, exist_ok=True)

after_cluster_folder = os.path.join(faces_folder, 'after_cluster')
os.makedirs(after_cluster_folder, exist_ok=True)

# faces/before_cluster/images
images = os.path.join(before_cluster_folder, 'images')
os.makedirs(images, exist_ok=True)

# faces/before_cluster/encodings
encodings_folders = os.path.join(before_cluster_folder, 'encodings')
os.makedirs(encodings_folders, exist_ok=True)

# faces/suggestion_faces
suggestion_faces_folder = os.path.join(faces_folder, 'suggestion_faces')
os.makedirs(suggestion_faces_folder, exist_ok=True)

################################## 여기까지 기본 폴더 구성

frame_number = 0  # 현재 프레임 번호 초기화
track_face_data = defaultdict(list)  # 각 추적 ID별로 얼굴 데이터 저장하기 위한 딕셔너리
high_quality_image_selected = defaultdict(list)
high_quality_image_not_selected = defaultdict(list)


# # dlib 얼굴 랜드마크 모델 로드
# shape_predictor_path = './modelfiles/shape_predictor_68_face_landmarks.dat'  # 적절한 경로로 변경
# face_predictor = dlib.shape_predictor(shape_predictor_path)

def extract_face(frame, box):
    x1, y1, x2, y2 = map(int, box[:4])
    # 박스 크기를 유지하되, 프레임 내부로 조정
    x1 = max(0, x1 - 10)
    y1 = max(0, y1 - 10)
    x2 = min(frame.shape[1], x2 + 10)
    y2 = min(frame.shape[0], y2 + 10)

    # 이미지 추출
    face_img = frame[y1:y2, x1:x2]
    return face_img


def evaluate_image_quality(face_img):
    gray_image = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    brightness = cv2.mean(gray_image)[0]
    # 밝기에 따라 이미지 판단
    threshold_brightness = 220  # 임계값을 조정하여 밝기 판단을 바꿀 수 있습니다.
    threshold_darkness = 50

    if conf_value > 0.85:  # confidence 값이 0.85 이상인 경우에만 작업 수행
        # 이미지의 밝기 측정
        if brightness < threshold_brightness and brightness > threshold_darkness:
            return True
        else:
            return False


# 이미지 크기 조건 (너비 또는 높이가 최소 크기보다 작으면 작은 이미지로 판단)
MIN_FACE_WIDTH = 50
MIN_FACE_HEIGHT = 50

while cap.isOpened():
    success, frame = cap.read()
    if success:
        results = model.track(frame, persist=True, conf=0.5)
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().numpy()
            conf_values = results[0].boxes.conf.cpu()

            for box, track_id, conf_value in zip(boxes, track_ids, conf_values):
                face_img = extract_face(frame, box)

                # 이미지 크기 평가
                if face_img.shape[1] >= MIN_FACE_WIDTH and face_img.shape[0] >= MIN_FACE_HEIGHT:
                    # 객체 정보 저장
                    obj_data = {
                        'track_id': track_id,
                        'box_coordinates': box.tolist(),
                        'face_image': face_img,
                        'frame_number': frame_number,
                        'conf_value': conf_value,
                    }
                    track_face_data[track_id].append(obj_data)
                    if evaluate_image_quality(face_img):
                        high_quality_image_selected[track_id].append(obj_data)
                    else:
                        high_quality_image_not_selected[track_id].append(obj_data)

                # 각 추적 ID별로 데이터 저장
            frame_number += 1  # 다음 프레임 번호로 업데이트
    else:
        break
cap.release()

data_frames = []
for track_id, data_list in track_face_data.items():
    frame_count = len(data_list)
    data_frames.append({'track_id': track_id, 'frame_count': frame_count})
data_frame = pd.DataFrame(data_frames)

selected_data_frame = data_frame[data_frame['frame_count'] >= 60]

# 상위 5개 리스트에 저장. 밑에꺼는 그 밑에 5개 만드는 데이터 프레임인데 일단 보류.
top_5_ids = selected_data_frame.nlargest(5, 'frame_count')['track_id'].tolist()


# next_selected_data_frame = selected_data_frame.nlargest(5, 'frame_count').tail(5)

def save_images_for_id(track_id):
    selected_images = []
    # high_quality_image_selected에서 이미지 선택
    if track_id in high_quality_image_selected:
        images_list = high_quality_image_selected[track_id]

        # 이미지가 50장 이상인 경우, 랜덤하게 이미지를 선택
        if len(images_list) >= 50:
            selected_images = random.sample(images_list, 50)  # 이미지 리스트에서 50장을 랜덤 선택
            selected_images = [img_data['face_image'] for img_data in selected_images]
        else:
            for obj_data in images_list:
                selected_images.append(obj_data['face_image'])

    # high_quality_image_not_selected에서 이미지를 보충 선택
    if len(selected_images) < 50 and track_id in high_quality_image_not_selected:
        # conf_value를 기반으로 정렬
        sorted_images = sorted(high_quality_image_not_selected[track_id], key=lambda x: x['conf_value'], reverse=True)
        for obj_data in sorted_images:
            if len(selected_images) < 50:
                selected_images.append(obj_data['face_image'])

    # ID별 디렉토리 생성 및 이미지 저장
    id_dir = os.path.join(images, str(track_id))
    os.makedirs(id_dir, exist_ok=True)
    for idx, img in enumerate(selected_images):
        img_path = os.path.join(id_dir, f'id_{track_id}_conf{conf_value}_img{idx + 1}.jpg')
        cv2.imwrite(img_path, img)


# 상위 5개 ID에 대해 함수를 실행
for track_id in top_5_ids:
    save_images_for_id(track_id)
print("Images have been saved successfully!")

images_save_endpoint = time.time()
execution_time = images_save_endpoint - start_time
print(f"사진저장까지 걸린 시간: {execution_time:.2f} seconds")

############################################################################################################### 임베딩
track_id_minimum_dir = images
os.makedirs(track_id_minimum_dir , exist_ok=True)  # 디렉토리가 없으면 생성합니다.

#임베딩 파일 생성
encodings_dir = encodings_folders
os.makedirs(encodings_dir, exist_ok=True)

base_encodings_name = 'face_encodings'
extension = ".pickle"
counter = 0
encodings_path = os.path.join(encodings_dir, f"{base_encodings_name}{extension}")

# 파일 이름이 이미 존재하는 경우, 숫자를 증가시키며 새로운 파일 이름을 생성합니다.
while os.path.exists(encodings_path):
    counter += 1
    encodings_path = os.path.join(encodings_dir, f"{base_encodings_name}_{counter}{extension}")

track_id_dirs = [d for d in os.listdir(track_id_minimum_dir) if os.path.isdir(os.path.join(track_id_minimum_dir, d))]
all_image_paths_in_track_id_minimum = []

for track_id_dir in track_id_dirs:
    images_in_track_id_dir = [os.path.join(track_id_minimum_dir, track_id_dir, f) for f in os.listdir(os.path.join(track_id_minimum_dir, track_id_dir)) if f.endswith(('.jpg', '.png'))]
    all_image_paths_in_track_id_minimum.extend(images_in_track_id_dir)

selected_image_paths = all_image_paths_in_track_id_minimum

data = []
start_time = time.time()
last_print_time = time.time()
#

# for imagePath in selected_image_paths:
for index, imagePath in enumerate(selected_image_paths):
    # 현재 시간을 가져오고 마지막 출력 시간으로부터 30초가 지났는지 확인합니다.
    current_time = time.time()
    if current_time - last_print_time >= 30:
        print(f"Processing image {index + 1}/{len(selected_image_paths)}: {imagePath}")
        last_print_time = current_time

    # 이미지를 로드하고 RGB로 변환
    image = face_recognition.load_image_file(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 얼굴 찾기
    face_locations = face_recognition.face_locations(image)
    if len(face_locations) == 0:
        continue

    encodings = face_recognition.face_encodings(rgb, face_locations)

    # 이미지 경로, bounding box 위치, 얼굴 임베딩 포함하는 딕셔너리 생성
    d = [{"imagePath": imagePath, "loc": loc, "encoding": enc}
         for (loc, enc) in zip(face_locations, encodings)]
    data.extend(d)

# 얼굴 임베딩 값을 피클로 직렬화하여 저장합니다.
print("[INFO] serializing encodings...")
with open(encodings_path, "wb") as f:
    f.write(pickle.dumps(data))

# # 종료 시간 체크
# end_time = time.time()

# # 걸린 시간 계산
# elapsed_time = end_time - start_time
# base_output_path = "embeddings_time"
# extension = ".txt"
# counter = 0
# output_path = f"{base_output_path}{extension}"

# 파일 이름이 이미 존재하는 경우, 숫자를 증가시키며 새로운 파일 이름을 생성합니다.
# while os.path.exists(output_path):
#     counter += 1
#     output_path = f"{base_output_path}_{counter}{extension}"

# # 생성된 파일 이름으로 파일을 작성합니다.
# with open(output_path, "w") as file:
#     file.write(f"Total elapsed time for embedding calculation: {elapsed_time} seconds.")

################################################################################################ 클러스터링
os.environ['OMP_NUM_THREADS'] = '1'

# # 인코딩 파일 경로 설정
encodings_path = os.path.join(encodings_folders, 'face_encodings.pickle')
# 데이터 로드
print("[INFO] loading encodings...")
# data = pickle.loads(open(args["encodings"], "rb").read())
data = pickle.loads(open(encodings_path, "rb").read())
data = np.array(data)
encodings = [d["encoding"] for d in data]

# 클러스터링
print("[INFO] clustering...")

n_clusters_value = 3  # 예상되는 클러스터 수
kmeans = KMeans(n_clusters=n_clusters_value, n_init=10)  # n_init을 명시적으로 설정
kmeans.fit(encodings)


# 라벨을 얻습니다.
labelIDs = kmeans.labels_
numUniqueFaces = len(np.where(labelIDs > -1)[0])
print("[INFO] # unique faces: {}".format(numUniqueFaces))

label_info = {} #track_ids and image count 저장

for labelID in labelIDs:
    print("[INFO] saving faces for face ID: {}".format(labelID))

    # 해당 라벨의 디렉터리 생성
    label_dir = os.path.join(after_cluster_folder, f"label_{labelID}")
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

    # 해당 라벨의 인덱스 가져오기
    idxs = np.where(kmeans.labels_ == labelID)[0]

    track_ids = set()

    # 인덱스에 해당하는 얼굴 이미지 저장
    for i in idxs:
        image = cv2.imread(data[i]["imagePath"])
        (top, right, bottom, left) = data[i]["loc"]
        face = image[top:bottom, left:right]

        # 이미지 파일명 설정
        filename = os.path.basename(data[i]["imagePath"])
        save_path = os.path.join(label_dir, filename)

        cv2.imwrite(save_path, face)

        #track id 추출
        track_id = re.search('id_(\d+)', filename)
        if track_id:
            track_ids.add(track_id.group(1))

    label_info[labelID] = {
        "track_ids": list(track_ids),
        "count": len(idxs)
    }

#txt 생성
base_output_path = "label_info_with_track_ids"
extension = ".txt"
counter = 0
output_path = os.path.join(after_cluster_folder, f"{base_output_path}{extension}")

# 파일 이름이 이미 존재하는 경우, 숫자를 증가시키며 새로운 파일 이름을 생성합니다.
while os.path.exists(output_path):
    counter += 1
    output_path = f"{base_output_path}_{counter}{extension}"

with open(output_path, "w") as f:
    for labelID, info in label_info.items():
        track_ids_str = ", ".join(info["track_ids"])
        f.write(f"label_{labelID}: track_ids {track_ids_str} image count {info['count']}\n")

#라벨별로 1개의 대표 이미지 썸네일 추출 suggestion_faces/label_{label}.jpg
#조건: 라벨 디렉토리에서 가장 큰 해상도, 눈코입 제대로 추출, 눈을 뜬 이미지

#썸네일 이미지 사이즈 통일 - 대표 이미지 선정 이후
def resize_image(image, target_size):
    h, w, _ = image.shape

    scale = target_size / max(h, w)
    resized_image = cv2.resize(image, (int(w * scale), int(h * scale)))

    final_image = np.ones((target_size, target_size, 3), dtype=np.uint8) * 255

    y_offset = (target_size - resized_image.shape[0]) // 2
    x_offset = (target_size - resized_image.shape[1]) // 2

    final_image[y_offset:y_offset + resized_image.shape[0], x_offset:x_offset + resized_image.shape[1]] = resized_image

    return final_image

#face, eyes 검출을 위해서 로드
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def detect_eyes(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
    return img, eyes

for labelID in np.unique(labelIDs):
    label_dir = os.path.join(after_cluster_folder, f"label_{labelID}")

    # Collect all images and sort them by resolution
    image_paths = [os.path.join(label_dir, f) for f in os.listdir(label_dir)]
    image_paths.sort(key=lambda x: -os.path.getsize(x))

    representative_image = None
    for image_path in image_paths:
        img, eyes = detect_eyes(image_path)
        if len(eyes) >= 2:
            representative_image = img
            break

    if representative_image is not None:
        save_path = os.path.join(suggestion_faces_folder, f"label_{labelID}.jpg")
        img_resized = resize_image(representative_image, 100)
        img_resized = resize_image(representative_image, 100)
        cv2.imwrite(save_path, img_resized)
    else:
        print(f"[WARNING] Eyes not detected for any image in label_{labelID}.")
