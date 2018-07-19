import numpy as np
import pickle
import face_recognition
import pygame
from pygame.locals import *
import sys
import cv2


def main():

    with open('./data/model.pkl', 'rb') as f:
        clf = pickle.load(f)

    happy_face = pygame.image.load('./data/baby_image/happiness.png')
    neutral_face = pygame.image.load('./data/baby_image/neutral.png')
    cry_face = pygame.image.load('./data/baby_image/cry.png')
    image = neutral_face

    video_capture = cv2.VideoCapture(0)
    process_this_frame = True
    face_landmarks = {}
    fe_class = None

    baby_tension = 0

    pygame.init()
    screen = pygame.display.set_mode((600, 300))

    while True:

        ret, frame = video_capture.read()

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        if process_this_frame:
            face_landmarks_list = face_recognition.face_landmarks(rgb_small_frame)
            if face_landmarks_list:
                face_landmarks = face_landmarks_list[0]

                feature = get_feature(face_landmarks)
                fe_class = clf.predict([feature])[0]

                baby_tension += (fe_class * 0.5)
                if baby_tension >= 5:
                    baby_tension = 5

        process_this_frame = not process_this_frame

        # if fe_class is not None:
        #     if fe_class == 0:
        #         image = neutral_face
        #     elif fe_class == 1:
        #         image = happy_face
        #     else:
        #         image = cry_face

        if baby_tension >= 2:
            image = happy_face
            screen.fill((255, 204, 255))
        elif baby_tension < 2 and baby_tension >= -3:
            image = neutral_face
            screen.fill((255, 255, 255))
        else:
            image = cry_face
            screen.fill((0, 153, 204))

        screen.blit(image, (300 - (image.get_rect()[2] / 2), 150 - (image.get_rect()[3] / 2)))

        pygame.display.update()

        baby_tension -= 0.1
        if baby_tension <= -10:
            baby_tension = -10
        print(baby_tension)

        for event in pygame.event.get():  # 終了処理
            if event.type == QUIT:
                pygame.quit()
                sys.exit()


def points_min_max_norm(points):
    return (points - points.min(axis=0)) / (points.max(axis=0) - points.min(axis=0))


def get_feature(landmarks):
    features = [landmarks[parts_name] for parts_name in ['left_eyebrow', 'right_eyebrow', 'left_eye', 'right_eye', 'top_lip', 'bottom_lip']]
    features = [np.array(feature) for feature in features]
    features_norm = list(map(points_min_max_norm, features))
    features_flatten = [value for feature in features_norm for value in feature.flatten()]
    return np.array(features_flatten)


if __name__ == '__main__':
    main()
