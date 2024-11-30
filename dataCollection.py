import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math 
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)

offset = 20
imgSize = 300

folder = "DataSet/What"
counter = 0

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        if len(hands) == 1:  # Case for one hand
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            imgCropShape = imgCrop.shape
            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

            # Draw the bounding box for the single hand
            cv2.rectangle(img, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 0), 2)

        elif len(hands) == 2:  # Case for two hands
            hand1 = hands[0]
            hand2 = hands[1]

            x1, y1, w1, h1 = hand1['bbox']
            x2, y2, w2, h2 = hand2['bbox']

            # Calculate the bounding box that covers both hands
            x_min = min(x1, x2) - offset
            y_min = min(y1, y2) - offset
            x_max = max(x1 + w1, x2 + w2) + offset
            y_max = max(y1 + h1, y2 + h2) + offset

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y_min:y_max, x_min:x_max]

            imgCropShape = imgCrop.shape
            aspectRatio = imgCropShape[0] / imgCropShape[1]

            if aspectRatio > 1:
                k = imgSize / imgCropShape[0]
                wCal = math.ceil(k * imgCropShape[1])
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / imgCropShape[1]
                hCal = math.ceil(k * imgCropShape[0])
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

            # Draw the bounding box for the combined area of both hands
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

    # Display the original image
    cv2.imshow("Image", img)

    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        if len(hands) == 1:
            # Save the cropped image for one hand
            cv2.imwrite(f'{folder}/OneHand_{time.time()}.jpg', imgWhite)
            print(f"One-hand image saved: {counter}")
        elif len(hands) == 2:
            # Save the cropped image for two hands
            cv2.imwrite(f'{folder}/TwoHands_{time.time()}.jpg', imgWhite)
            print(f"Two-hands image saved: {counter}")

