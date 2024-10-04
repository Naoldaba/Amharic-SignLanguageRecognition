import os
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np

import sys
sys.stdout.reconfigure(encoding='utf-8')

data_directory = './data'
if not os.path.exists(data_directory):
    os.makedirs(data_directory)

labels = 33
label_size = 100

def put_amharic_text(frame, text, position, font_size, color=(0, 0, 0)):
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    font_path = "./fonts/washrab.ttf"  
    font = ImageFont.truetype(font_path, font_size)

    draw = ImageDraw.Draw(pil_image)
    draw.text(position, text, font=font, fill=color)

    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

cap = cv2.VideoCapture(0)
for j in range(labels):
    if not os.path.exists(os.path.join(data_directory, str(j))):
        os.makedirs(os.path.join(data_directory, str(j)))

    print('የክፍል {} መረጃን እየሰበሰብን ነው'.format(j))

    done = False
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = put_amharic_text(frame, 'እራስህን አዘጋጅ! "Y" ን ጠቅ ያድርጉ! :)', (100, 50), font_size=25, color=(0, 0, 0))
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('Y'):
            break

    counter = 0
    while counter < label_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(data_directory, str(j), '{}.jpg'.format(counter)), frame)

        counter += 1

cap.release()
cv2.destroyAllWindows()