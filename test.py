import easyocr
import cv2
from matplotlib import pyplot as plt
import numpy as np

image_path = 'numbers3.png'

reader = easyocr.Reader(['en'], gpu=True)
results = reader.readtext(image_path)

img = cv2.imread(image_path)

with open('detected_text.txt', 'w') as f:
    for (bbox, text, prob) in results:
        print(f"Detected text: {text} (Confidence: {prob:.2f})")

        f.write(text + '\n')
        
        top_left = tuple(np.array(bbox[0]).astype(int))
        bottom_right = tuple(np.array(bbox[2]).astype(int))
    
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        img = cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 3)
        
        img = cv2.putText(img, text, top_left, font, 1, (255, 0, 0), 2, cv2.LINE_AA)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

print("\nResults have been saved to detected_text.txt")