import numpy as np
import pickle
import face_recognition
import cv2

fe_text = ['Neutral', 'Happiness', 'Anger']


def main():

    with open('./data/model.pkl', 'rb') as f:
        clf = pickle.load(f)

    video_capture = cv2.VideoCapture(0)
    process_this_frame = True
    face_landmarks = {}
    fe_class = None

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

        process_this_frame = not process_this_frame

        for parts_name, parts_pos in face_landmarks.items():

            for pos in parts_pos:
                cv2.circle(frame, (pos[0] * 4, pos[1] * 4), 10, (0, 255, 0), -1)

            if parts_name == 'left_eye' or parts_name == 'right_eye':
                eye_center = eye_point(rgb_small_frame, parts_pos)
                if eye_center:
                    cv2.circle(frame, (eye_center[0] * 4, eye_center[1] * 4), 5, (0, 0, 255), -1)

        if fe_class is not None:
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, fe_text[fe_class], (100, 100), font, 1, (255, 0, 0))

        cv2.imshow('test capture', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


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
