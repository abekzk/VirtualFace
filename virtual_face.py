import numpy as np
import pickle
import pygame
from pygame.locals import *
import sys
import face_recognition
import cv2

fe_text = ['Neutral', 'Happiness', 'Anger']


def main():

    video_capture = cv2.VideoCapture(0)
    ret, frame = video_capture.read()

    process_this_frame = True
    face_landmarks = {}

    pygame.init()
    screen = pygame.display.set_mode((frame.shape[1], frame.shape[0]))

    while True:
        screen.fill((255, 255, 255))

        ret, frame = video_capture.read()

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        if process_this_frame:
            face_landmarks_list = face_recognition.face_landmarks(rgb_small_frame)
            if face_landmarks_list:
                face_landmarks = face_landmarks_list[0]

        process_this_frame = not process_this_frame

        if len(face_landmarks):
            c_center = chin_center(face_landmarks['chin'])
            pygame.draw.circle(screen, (255, 255, 0), (c_center[0] * 4, c_center[1] * 4), 250)

            el_center = parts_center(face_landmarks['left_eye'])
            dist = parts_dist(face_landmarks['left_eye'])
            pygame.draw.ellipse(screen, (255, 255, 255), (el_center[0] * 4 - 60, el_center[1] * 4 - (dist[1] * 15), 120, dist[1] * 15 * 2))
            eye_center = eye_point(rgb_small_frame, face_landmarks['left_eye'])
            if eye_center:
                pygame.draw.circle(screen, (0, 0, 0), (eye_center[0] * 4, eye_center[1] * 4), 30)

            er_center = parts_center(face_landmarks['right_eye'])
            dist = parts_dist(face_landmarks['right_eye'])
            pygame.draw.ellipse(screen, (255, 255, 255), (er_center[0] * 4 - 60, er_center[1] * 4 - (dist[1] * 15), 120, dist[1] * 15 * 2))
            eye_center = eye_point(rgb_small_frame, face_landmarks['right_eye'])
            if eye_center:
                pygame.draw.circle(screen, (0, 0, 0), (eye_center[0] * 4, eye_center[1] * 4), 30)

            polylist = face_landmarks['top_lip'][6:] + face_landmarks['bottom_lip'][6:]
            polylist = list(map(lambda x: (x[0] * 4, x[1] * 4), polylist))
            pygame.draw.polygon(screen, (0, 0, 0), polylist)

            linelist = list(map(lambda x: (x[0] * 4, x[1] * 4 - 50), face_landmarks['left_eyebrow']))
            pygame.draw.lines(screen, (0, 0, 0), False, linelist, 7)

            linelist = list(map(lambda x: (x[0] * 4, x[1] * 4 - 50), face_landmarks['right_eyebrow']))
            pygame.draw.lines(screen, (0, 0, 0), False, linelist, 7)

        # for parts_name, parts_pos in face_landmarks.items():

        #     for pos in parts_pos:
        #         pygame.draw.circle(screen, (0, 0, 0), (pos[0] * 4, pos[1] * 4), 5)

        #     if parts_name == 'left_eye' or parts_name == 'right_eye':
        #         eye_center = eye_point(rgb_small_frame, parts_pos)
        #         if eye_center:
        #             pygame.draw.circle(screen, (255, 0, 0), (eye_center[0] * 4, eye_center[1] * 4), 5)

        pygame.display.update()

        for event in pygame.event.get():  # 終了処理
            if event.type == QUIT:
                pygame.quit()
                sys.exit()


def parts_dist(pos):
    points = np.array([[point[0], point[1]] for point in pos])
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()

    dist_x = x_max - x_min
    dist_y = y_max - y_min

    return dist_x, dist_y


def parts_center(pos):
    points = np.array([[point[0], point[1]] for point in pos])
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()

    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2

    return int(center_x), int(center_y)


def chin_center(chin_pos):
    points = np.array([[point[0], point[1]] for point in chin_pos])
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()

    center_x = (x_min + x_max) / 2
    center_y = y_min * 4 / 5 + y_max * 1 / 5

    return int(center_x), int(center_y)


def eye_point(img, parts):
    points = np.array([[point[0], point[1]] for point in parts])
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    eye = img[y_min:y_max, x_min:x_max]
    _, eye = cv2.threshold(cv2.cvtColor(eye, cv2.COLOR_RGB2GRAY), 80, 255, cv2.THRESH_BINARY_INV)
    center = get_center(eye)
    if is_close(y_min, y_max):
        return None

    if center:
        return center[0] + x_min, center[1] + y_min
    return center


def get_center(gray_img):
    moments = cv2.moments(gray_img, False)
    try:
        return int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00'])
    except:
        return None


def is_close(y0, y1):
    if abs(y0 - y1) < 1:
        return True
    return False


if __name__ == '__main__':
    main()
