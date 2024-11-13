import cv2
import os
import logging
import json

class PhotoGeneratorConfig:
    VIDEO_PATH = r'保存先フォルダのパス\2019 全日本シングルスソフトテニス選手権 男子決勝.mp4'
    OUTPUT_PATH = 'photos\photos'
    

cap = cv2.VideoCapture()

output_f = 'photos\photos_5'
os.makedirs(output_f, exist_ok=True)

total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print(total_frames)

if 1:
    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if (frame_number % 100) == 0: 
            frame_filename = os.path.join(output_f, f'frame_{frame_number}.png')
            cv2.imwrite(frame_filename, frame)
            
            print(f'{frame_filename}を保存しました。')
        frame_number += 1
        
    cap.release()
    cv2.destroyAllWindows()
