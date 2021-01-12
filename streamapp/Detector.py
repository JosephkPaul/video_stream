# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 01:04:44 2020

@author: hp
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import math

from face_detector import get_face_detector, find_faces
from face_landmarks import get_landmark_model, detect_marks, draw_marks

face_model = get_face_detector()
landmark_model = get_landmark_model()
outer_points = [[49, 59], [50, 58], [51, 57], [52, 56], [53, 55]]
d_outer = [0] * 5
inner_points = [[61, 67], [62, 66], [63, 65]]
d_inner = [0] * 3
font = cv2.FONT_HERSHEY_SIMPLEX
cap = cv2.VideoCapture(0)

ret, img = cap.read()
size = img.shape
# 3D model points.
model_points = np.array([
    (0.0, 0.0, 0.0),  # Nose tip
    (0.0, -330.0, -65.0),  # Chin
    (-225.0, 170.0, -135.0),  # Left eye left corner
    (225.0, 170.0, -135.0),  # Right eye right corne
    (-150.0, -150.0, -125.0),  # Left Mouth corner
    (150.0, -150.0, -125.0)  # Right mouth corner
])

# Camera internals
focal_length = size[1]
center = (size[1] / 2, size[0] / 2)
camera_matrix = np.array(
    [[focal_length, 0, center[0]],
     [0, focal_length, center[1]],
     [0, 0, 1]], dtype="double"
)


#######################head pose################
def get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val):
    """Return the 3D points present as 2D for making annotation box"""
    point_3d = []
    dist_coeffs = np.zeros((4, 1))
    rear_size = val[0]
    rear_depth = val[1]
    point_3d.append((-rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, -rear_size, rear_depth))

    front_size = val[2]
    front_depth = val[3]
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d.append((-front_size, front_size, front_depth))
    point_3d.append((front_size, front_size, front_depth))
    point_3d.append((front_size, -front_size, front_depth))
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)

    # Map to 2d img points
    (point_2d, _) = cv2.projectPoints(point_3d,
                                      rotation_vector,
                                      translation_vector,
                                      camera_matrix,
                                      dist_coeffs)
    point_2d = np.int32(point_2d.reshape(-1, 2))
    return point_2d


def draw_annotation_box(img, rotation_vector, translation_vector, camera_matrix,
                        rear_size=300, rear_depth=0, front_size=500, front_depth=400,
                        color=(255, 255, 0), line_width=2):

    rear_size = 1
    rear_depth = 0
    front_size = img.shape[1]
    front_depth = front_size * 2
    val = [rear_size, rear_depth, front_size, front_depth]
    point_2d = get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val)
    # # Draw all the lines
    cv2.polylines(img, [point_2d], True, color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[1]), tuple(
        point_2d[6]), color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[2]), tuple(
        point_2d[7]), color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[3]), tuple(
        point_2d[8]), color, line_width, cv2.LINE_AA)


def head_pose_points(img, rotation_vector, translation_vector, camera_matrix):
    rear_size = 1
    rear_depth = 0
    front_size = img.shape[1]
    front_depth = front_size * 2
    val = [rear_size, rear_depth, front_size, front_depth]
    point_2d = get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val)
    y = (point_2d[5] + point_2d[8]) // 2
    x = point_2d[2]

    return (x, y)


#######################mouth################################################################

while (True):
    #print("initial")
    ret, img = cap.read()
    rects = find_faces(img, face_model)
    for rect in rects:
        shape = detect_marks(img, landmark_model, rect)
        draw_marks(img, shape)
        # cv2.putText(img, 'Press r to record Mouth distances', (30, 30), font,
        # 1, (0, 255, 255), 2)
        cv2.imshow("Output", img)
    if cv2.waitKey(1):
        for i in range(100):
            for i, (p1, p2) in enumerate(outer_points):
                d_outer[i] += shape[p2][1] - shape[p1][1]
            for i, (p1, p2) in enumerate(inner_points):
                d_inner[i] += shape[p2][1] - shape[p1][1]
        break
cv2.destroyAllWindows()
d_outer[:] = [x / 100 for x in d_outer]
d_inner[:] = [x / 100 for x in d_inner]

hu=0
hd=0
hl=0
hr=0
M=0
toll=100

while (True):
    ret, img = cap.read()
    if ret == True:
        faces = find_faces(img, face_model)

        for face in faces:
            marks = detect_marks(img, landmark_model, face)
            #######################

            shape=marks
            cnt_outer = 0
            cnt_inner = 0
            draw_marks(img, shape[48:])
            for i, (p1, p2) in enumerate(outer_points):
                if d_outer[i] + 3 < shape[p2][1] - shape[p1][1]:
                    cnt_outer += 1
            for i, (p1, p2) in enumerate(inner_points):
                if d_inner[i] + 2 < shape[p2][1] - shape[p1][1]:
                    cnt_inner += 1
            if cnt_outer > 3 and cnt_inner > 2:
                M= M + 1
                if M >= 8:
                    toll = toll - 1
                    M = 0
                    f = open("mem.txt", "a")
                    f.write(str("MOUTH OPEN") + " "+str(toll)+"\n")
                    f.close()
                    #print(toll)


                cv2.putText(img, 'Mouth open', (30, 30), font,
                            1, (0, 255, 255), 2)

            ###################################
            # mark_detector.draw_marks(img, marks, color=(0, 255, 0))
            image_points = np.array([
                marks[30],  # Nose tip
                marks[8],  # Chin
                marks[36],  # Left eye left corner
                marks[45],  # Right eye right corne
                marks[48],  # Left Mouth corner
                marks[54]  # Right mouth corner
            ], dtype="double")
            dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                          dist_coeffs, flags=cv2.SOLVEPNP_UPNP)

            # Project a 3D point (0, 0, 1000.0) onto the image plane.
            # We use this to draw a line sticking out of the nose

            (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector,
                                                             translation_vector, camera_matrix, dist_coeffs)

            for p in image_points:
                cv2.circle(img, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

            p1 = (int(image_points[0][0]), int(image_points[0][1]))
            p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
            x1, x2 = head_pose_points(img, rotation_vector, translation_vector, camera_matrix)

            cv2.line(img, p1, p2, (0, 255, 255), 2)
            cv2.line(img, tuple(x1), tuple(x2), (255, 255, 0), 2)
            # for (x, y) in marks:
            #     cv2.circle(img, (x, y), 4, (255, 255, 0), -1)
            # cv2.putText(img, str(p1), p1, font, 1, (0, 255, 255), 1)
            try:
                m = (p2[1] - p1[1]) / (p2[0] - p1[0])
                ang1 = int(math.degrees(math.atan(m)))
            except:
                ang1 = 90

            try:
                m = (x2[1] - x1[1]) / (x2[0] - x1[0])
                ang2 = int(math.degrees(math.atan(-1 / m)))
            except:
                ang2 = 90

                # print('div by zero error')
            if ang1 >= 40:

                hd=hd+1
                if hd>=5:
                    toll=toll-1
                    hd=0
                    f = open("mem.txt", "a")
                    f.write("Head down" + " " + str(toll)+"\n")
                    f.close()
                    #print(toll)

                cv2.putText(img, 'Head down', (30, 30), font, 2, (255, 255, 128), 3)

            elif ang1 <= -35:

                hu = hu + 1
                if hu >= 5:
                    toll = toll - 1
                    hu = 0
                    f = open("mem.txt", "a")
                    f.write("Head Up" + " " + str(toll)+"\n")
                    f.close()
                    #print(toll)

                cv2.putText(img, 'Head up', (30, 30), font, 2, (255, 255, 128), 3)

            if ang2 >= 40:
                hr = hr + 1
                if hr >= 5:
                    toll = toll - 1
                    hr = 0
                    f = open("mem.txt", "a")
                    f.write("Head Right" + " " + str(toll)+"\n")
                    f.close()
                    print(toll)

                cv2.putText(img, 'Head right', (90, 30), font, 2, (255, 255, 128), 3)
            elif ang2 <= -40:
                hl = hl + 1
                if hl >= 5:
                    toll = toll - 1
                    hl = 0
                    f = open("mem.txt", "a")
                    f.write("Head Left" + " " + str(toll)+"\n")
                    f.close()
                    #print(toll)

                cv2.putText(img, 'Head left', (90, 30), font, 2, (255, 255, 128), 3)

            cv2.putText(img, str(ang1), tuple(p1), font, 2, (128, 255, 255), 3)
            cv2.putText(img, str(ang2), tuple(x1), font, 2, (255, 255, 128), 3)
        #cv2.imshow('img', img)

    ######################mouth###############
    #ret, img = cap.read()
    #rects = find_faces(img, face_model)
    """
    for rect in rects:
        shape = detect_marks(img, landmark_model, rect)
               # show the output image with the face detections + facial landmarks
    """
    cv2.imshow("Output", img)
    if cv2.waitKey(1)& 0xFF == ord('q'):
        with open("mem.txt","r") as f:
            d=f.readlines()

        break



cap.release()
cv2.destroyAllWindows()
