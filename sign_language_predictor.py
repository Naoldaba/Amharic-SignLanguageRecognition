import pickle
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

model_dict = pickle.load(open('./AmharicSignLangRecognition.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)  

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)
labels_dict = {0: 'ሀ', 1: 'ለ', 2: 'ሐ', 3: 'መ', 4: 'ሠ', 5: 'ረ', 6: 'ሰ', 7: 'ሸ'}

amharic_font_path = "./fonts/washrab.ttf"  
font = ImageFont.truetype(amharic_font_path, 32) 

while True:
    data_aux = []  
    x_ = []
    y_ = []

    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(21): 
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            min_x, min_y = min(x_), min(y_)
            for i in range(21):  
                data_aux.append(x_[i] - min_x)
                data_aux.append(y_[i] - min_y)

        if len(data_aux) == 42:
            data_input = np.asarray(data_aux).reshape(1, -1)  # Ensure shape is (1, 42)
            prediction = model.predict(data_input)

            predicted_character = labels_dict[int(prediction[0])]

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)

            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            draw.text((x1, y1 - 40), predicted_character, font=font, fill=(0, 0, 0))
            frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
