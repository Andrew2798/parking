from imageai.Detection import ObjectDetection
import os
import cv2

cap = cv2.VideoCapture('parking.mp4')
#cap = cv2.VideoCapture(0)
execution_path = os.getcwd()
detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(os.path.join(execution_path, "yolo.h5"))
detector.loadModel(detection_speed='fast')
custom = detector.CustomObjects(car=True, motorcycle=True, bus=True, truck=True)

detect = []

point_pos_x = [50, 165, 345, 490, 660, 840, 990, 1150]
point_pos_y = 550

color_list = [(0, 255, 0)] * 8


frame_number = 0
while True:
    ret, frame = cap.read()
    if ret:
        frame_number += 1
        if frame_number % 5 == 0:
            returned_image, detect = detector.detectCustomObjectsFromImage(custom_objects=custom, input_image=frame,
                                                                           minimum_percentage_probability=20,
                                                                           output_type='array', input_type='array')
            color_list_tmp = [(0, 255, 0)] * 8
            for i in range(len(detect)):
                x1, y1, x2, y2 = tuple(detect[i]['box_points'])
                if y1 > 420 and y2 < 700:
                    # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    for j in range(len(point_pos_x)):
                        if (point_pos_x[j] > x1) and (point_pos_x[j] < x2) and (point_pos_y > y1) and (point_pos_y < y2):
                            color_list_tmp[j] = (0, 0, 255)
            color_list = color_list_tmp
            for k in range(len(point_pos_x)):
                cv2.circle(frame, (point_pos_x[k], point_pos_y), 15, color_list[k], -1)
            # cv2.line(frame, (0, 420), (1280, 420), (255, 0, 0), 3)
            # cv2.line(frame, (0, 700), (1280, 700), (255, 0, 0), 3)
            cv2.imshow("Video Original", frame)
            frame_number = 0
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        break

cv2.destroyAllWindows()
cap.release()
