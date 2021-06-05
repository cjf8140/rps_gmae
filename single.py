import cv2
import mediapipe as mp
import numpy as np
import time
max_num_hands = 1
gesture = {
    0:'fist', 1:'one', 2:'two', 3:'three', 4:'four', 5:'five',
    6:'six', 7:'rock', 8:'spiderman', 9:'yeah', 10:'ok',
}
rps_gesture = {0:'Rock', 5:'Paper', 9:'Scissors'}

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Gesture recognition model
file = np.genfromtxt('data/gesture_train.csv', delimiter=',')
angle = file[:,:-1].astype(np.float32)
label = file[:, -1].astype(np.float32)
knn = cv2.ml.KNearest_create()
knn.train(angle, cv2.ml.ROW_SAMPLE, label)

cap = cv2.VideoCapture(0)

initial_time = 0

timeon = 0

txt = 0

dec = 0

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        continue

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = hands.process(img)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 3))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z]

            # 조인트간 벡터 구하기
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # 부모
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # 자식
            v = v2 - v1 # [20,3]
            # Normalize v
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # Get angle using arcos of dot product
            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

            angle = np.degrees(angle) # Convert radian to degree

            # Inference gesture
            data = np.array([angle], dtype=np.float32)
            ret, results, neighbours, dist = knn.findNearest(data, 3)
            idx = int(results[0][0])
                
            if cv2.waitKey(1) != ord(' ') and timeon == 0:
                cv2.putText(img, text="Space!", org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 50)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2)
            if cv2.waitKey(1) == ord(' ') and timeon == 0:
                timeon = 1
                initial_time = time.time()
            if cv2.waitKey(1) == ord('z'):
                timeon = 0
                dec = 0
            if time.time() - initial_time < 1:
                cv2.putText(img, text="3", org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 50)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2)    
            elif time.time() - initial_time < 2:
                cv2.putText(img, text="2", org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 50)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2)    
            elif time.time() - initial_time < 3:
                cv2.putText(img, text="1", org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 50)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2)    
            if time.time() - initial_time > 3 and dec == 0 and timeon == 1:
                if idx == 0:
                    txt="Scissors..z"
                if idx == 5:
                    txt="Rock..z"
                if idx == 9:
                    txt="Paper..z"
                dec = 1
            if timeon == 1 and dec == 1:
                cv2.putText(img, text=txt, org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] +50)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2)    
            
            # Draw gesture result
            if idx in rps_gesture.keys():
                cv2.putText(img, text=rps_gesture[idx].upper(), org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0), thickness=2)
            
            

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('GBB', img)
    if cv2.waitKey(1) == ord('q'):
        break
