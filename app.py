import cv2
import numpy as np
import math
import streamlit as st
from PIL import Image
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

# Initialize Hand Detector and Classifier
detector = HandDetector(maxHands=2)  # Detect up to 2 hands
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20
imgSize = 300

# Labels for the classification
labels = ["Angry", "Hello", "Help", "I Love You", "No", "Ready", "Sorry", "Thank You", "What", "Who", "Yes"]

# Streamlit app setup
st.set_page_config(page_title="Sign Language Detection", layout="centered", page_icon="✋")

st.markdown(
    """
    <h1 style='text-align: center; color: #c23a22;'>SignConnect</h1>
    <h2 style='text-align: center; color: 'black';'>Real-Time & Image-Based Sign Language Detection</h2>
    """,
    unsafe_allow_html=True,
)

st.write(
    "Detect sign language using your **webcam** or by uploading an **image**. Works seamlessly on both desktop and mobile."
)

# Tabs for Webcam and Image Upload
tab1, tab2 = st.tabs(["📷 Webcam", "🖼️ Upload Image"])

# Webcam Tab
with tab1:
    st.markdown("### Real-Time Detection")
    run = st.checkbox("Enable Webcam")

    if run:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()  # Placeholder for video frames

        while True:
            success, img = cap.read()
            if not success:
                st.error("Unable to access the camera.")
                break

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

                # Crop and resize the image
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                imgCrop = img[y_min - offset:y_max + offset, x_min - offset:x_max + offset]

                aspectRatio = (y_max - y_min) / (x_max - x_min)

                if aspectRatio > 1:
                    k = imgSize / (y_max - y_min)
                    wcal = math.ceil(k * (x_max - x_min))
                    imgResize = cv2.resize(imgCrop, (wcal, imgSize))
                    wGap = math.ceil((imgSize - wcal) / 2)
                    imgWhite[:, wGap:wcal + wGap] = imgResize
                else:
                    k = imgSize / (x_max - x_min)
                    hcal = math.ceil(k * (y_max - y_min))
                    imgResize = cv2.resize(imgCrop, (imgSize, hcal))
                    hGap = math.ceil((imgSize - hcal) / 2)
                    imgWhite[hGap:hcal + hGap, :] = imgResize

                # Classification
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                confidence = prediction[index] * 100
                accuracy_text = f"{confidence:.2f}%"
                text = f"{labels[index]} : {accuracy_text}"

                # Draw the combined bounding box and label
                cv2.rectangle(imgOutput, (x_min - offset, y_min - offset - 50), (x_min + 200, y_min - offset), (255, 0, 255), cv2.FILLED)
                cv2.putText(imgOutput, text, (x_min, y_min - 20), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 2)
                cv2.rectangle(imgOutput, (x_min - offset, y_min - offset), (x_max + offset, y_max + offset), (255, 0, 255), 4)

            # Convert the BGR image to RGB for Streamlit
            imgOutput = cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGB)

            # Display the output in Streamlit
            stframe.image(imgOutput, channels="RGB", use_container_width=True)

        cap.release()
    else:
        st.info("Click the checkbox to start the webcam.")

# Image Upload Tab
with tab2:
    st.markdown("### Upload and Detect")
    uploaded_file = st.file_uploader("Upload an Image (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Read the uploaded image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Detect hands in the image
        hands, img = detector.findHands(img)
        imgOutput = img.copy()

        if hands:
            x_min, y_min = float('inf'), float('inf')
            x_max, y_max = 0, 0

            for hand in hands:
                x, y, w, h = hand['bbox']
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x + w)
                y_max = max(y_max, y + h)

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y_min - offset:y_max + offset, x_min - offset:x_max + offset]

            aspectRatio = (y_max - y_min) / (x_max - x_min)

            if aspectRatio > 1:
                k = imgSize / (y_max - y_min)
                wcal = math.ceil(k * (x_max - x_min))
                imgResize = cv2.resize(imgCrop, (wcal, imgSize))
                wGap = math.ceil((imgSize - wcal) / 2)
                imgWhite[:, wGap:wcal + wGap] = imgResize
            else:
                k = imgSize / (x_max - x_min)
                hcal = math.ceil(k * (y_max - y_min))
                imgResize = cv2.resize(imgCrop, (imgSize, hcal))
                hGap = math.ceil((imgSize - hcal) / 2)
                imgWhite[hGap:hcal + hGap, :] = imgResize

            # Classification
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            confidence = prediction[index] * 100
            text = f"Prediction: {labels[index]} ({confidence:.2f}%)"
            st.success(text)

            # Show the processed and original images
            st.image(imgOutput, caption="Processed Image", use_container_width=True)
        else:
            st.warning("No hands detected in the image.")

