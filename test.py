import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)  # Detect up to 2 hands
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20
imgSize = 300

label = ["Angry", "Hello","Help", "I Love You", "No", "Ready", "Sorry", "Thank You", "What", "Who", "Yes"]

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)  # Detect hands

    if hands:
        # Initialize bounding box coordinates
        x_min, y_min = float('inf'), float('inf')
        x_max, y_max = 0, 0

        # Calculate combined bounding box
        for hand in hands:
            x, y, w, h = hand['bbox']
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)

        # Define the combined bounding box dimensions
        combined_width = x_max - x_min
        combined_height = y_max - y_min

        # Crop and resize the image
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y_min - offset:y_max + offset, x_min - offset:x_max + offset]

        aspectRatio = combined_height / combined_width

        if aspectRatio > 1:
            k = imgSize / combined_height
            wcal = math.ceil(k * combined_width)
            imgResize = cv2.resize(imgCrop, (wcal, imgSize))
            wGap = math.ceil((imgSize - wcal) / 2)
            imgWhite[:, wGap:wcal + wGap] = imgResize
        else:
            k = imgSize / combined_width
            hcal = math.ceil(k * combined_height)
            imgResize = cv2.resize(imgCrop, (imgSize, hcal))
            hGap = math.ceil((imgSize - hcal) / 2)
            imgWhite[hGap:hcal + hGap, :] = imgResize

        # Classification
        prediction, index = classifier.getPrediction(imgWhite, draw=False)
        confidence = prediction[index] * 100
        accuracy_text = f"{confidence:.2f}%"
        text = f"{label[index]} : {accuracy_text}"

        # Draw the combined bounding box and label
        cv2.rectangle(imgOutput, (x_min - offset, y_min - offset - 50), (x_min + 200, y_min - offset), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, text, (x_min, y_min - 20), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x_min - offset, y_min - offset), (x_max + offset, y_max + offset), (255, 0, 255), 4)

        # Show intermediate results
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    # Display the final output
    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)
