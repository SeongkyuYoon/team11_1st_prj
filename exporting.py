import numpy as np
from ultralytics import YOLO
import cv2
from collections import defaultdict
import os
import pandas as pd
import sys

####################################여기 영상 경로만 맞춰주세요.
# video_path = './source_videos/ys_theking.mp4'  
video_path = sys.argv[1]
##################################################################

cap = cv2.VideoCapture(video_path)

video_name = os.path.splitext(os.path.basename(video_path))[0]

output_folder = './output_folders'
# 이미 존재하는 폴더를 삭제하고 새로운 폴더를 생성
folder_path = os.path.join(output_folder, video_name)

# 폴더 내의 파일 처리
label_info_path = os.path.join(folder_path, 'faces/after_cluster/label_info_with_track_ids.txt')

# 파일에서 데이터 읽기
with open(label_info_path, 'r') as file:
    text_data = file.readlines()

# 데이터를 저장할 빈 리스트 생성
data = []

# 텍스트 데이터를 처리하여 데이터 프레임에 추가
for line in text_data:
    parts = line.split(':')
    label = parts[0].strip()  # 레이블 추출
    track_ids_str = parts[1].split('track_ids')[1].split('image count')[0].strip()  # track_ids 부분 추출
    track_ids = [int(track_id.strip()) for track_id in track_ids_str.split(',')]  # track_ids를 리스트로 변환
    data.append((label, track_ids))

# 데이터 프레임 생성
df = pd.DataFrame(data, columns=['label', 'ids'])

selected_images_txt = os.path.join(folder_path, 'faces/suggestion_faces/selected_images.txt')

with open(selected_images_txt, 'r') as file:
    text_data_selected = file.readlines()  

labels_to_search = [line.strip() for line in text_data_selected]    

matching_items = df[df['label'].isin(labels_to_search)]
# matching_items에서 'ids' 열을 추출하고 모든 값들을 하나의 리스트로 저장
exclude_ids = matching_items['ids'].explode().tolist()

# print(df)
# print(matching_items)
# print(exclude_ids)

# 모델 불러오기
model = YOLO('./ptfiles/v8_m_100_batch30.pt')

# 영상 정보 추출
video_name = os.path.splitext(os.path.basename(video_path))[0]
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# 경로 지정
# output_dir = './output_folders'

# output_dir_videos = folder_path

# 이 부분이 경로입니다. 현재는 'output_videos' 라는 폴더에 '(원래 영상 이름)_blurred' 라는 이름으로 저장되게 되어 있습니다.
output_path = f"./output_folders/{video_name}/{video_name}_blurred.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# 모자이크 처리하지 않을 아이디들의 리스트
# exclude_ids = [1]  # 예시 아이디, 필요에 따라 수정

while cap.isOpened():
    success, frame = cap.read()

    if success:
        # 태래킹 정보 추출
        results = model.track(frame, persist=True, conf=0.5)
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().numpy()

            # 반복문을 통해 프레임에 추가
            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box[:4])  # 박스 좌표 변수에 할당
                
                # 특정 아이디가 리스트에 없는 경우에만 모자이크 처리
                if track_id not in exclude_ids:
                    # 얼굴 영역 추출
                    face_region = frame[y1:y2, x1:x2]
                    # 얼굴에 모자이크 입히기
                    mosaic_factor = 15
                    small_face = cv2.resize(face_region, (face_region.shape[1] // mosaic_factor, face_region.shape[0] // mosaic_factor))
                    mosaic_face = cv2.resize(small_face, (face_region.shape[1], face_region.shape[0]), interpolation=cv2.INTER_NEAREST)

                    # 모자이크 얼굴로 원래 얼굴 대체하기
                    frame[y1:y2, x1:x2] = mosaic_face
            
            annotated_frame = frame
            # 프레임을 영상으로 저장
            out.write(annotated_frame)
        else:
            annotated_frame = frame   
            out.write(annotated_frame) 

    else:
        break
        
print('완료')    
cap.release()
out.release()  # VideoWriter 객체를 닫아주어야 영상이 저장됩니다.

