import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

# Load the pre-trained Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Get the absolute paths to the sample images
sample_image1_path = os.path.abspath('/Users/bandaruvinaykumar/Desktop/Projects/Projects/Opencv/Face_recognition Using Opencv & Python/Faces/Vinay.jpg')
sample_image2_path = os.path.abspath('/Users/bandaruvinaykumar/Desktop/Projects/Projects/Opencv/Face_recognition Using Opencv & Python/Faces/Abdul_Kalam.jpg')

# Load the sample images
sample_image1 = cv2.imread(sample_image1_path)
sample_image2 = cv2.imread(sample_image2_path)

# Convert sample images to grayscale
sample_image1_gray = cv2.cvtColor(sample_image1, cv2.COLOR_BGR2GRAY)
sample_image2_gray = cv2.cvtColor(sample_image2, cv2.COLOR_BGR2GRAY)

# Detect faces in sample images
faces1 = face_cascade.detectMultiScale(sample_image1_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
faces2 = face_cascade.detectMultiScale(sample_image2_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Assuming you have only one face in each sample image, extract the face ROI
if len(faces1) == 1:
    (x1, y1, w1, h1) = faces1[0]
    face_roi1 = sample_image1_gray[y1:y1 + h1, x1:x1 + w1]

if len(faces2) == 1:
    (x2, y2, w2, h2) = faces2[0]
    face_roi2 = sample_image2_gray[y2:y2 + h2, x2:x2 + w2]

# Initialize empty lists to store ROI data
roi_data = []

# Open a video capture object for the default camera (change the index if needed)
cap = cv2.VideoCapture(0)

frame_count = 0  # Initialize frame count for plotting

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Perform face recognition by comparing the detected face with the sample faces
        detected_face = gray[y:y + h, x:x + w]
        match_result1 = cv2.matchTemplate(detected_face, face_roi1, cv2.TM_CCOEFF_NORMED)
        match_result2 = cv2.matchTemplate(detected_face, face_roi2, cv2.TM_CCOEFF_NORMED)

        # Append ROI data to the list
        roi_data.append({
            'Match Result 1': match_result1[0][0],
            'Match Result 2': match_result2[0][0],
        })

    # Display the frame with detected faces and recognition results
    cv2.imshow('Face Recognition', frame)

    # Check if any key is pressed (wait for a key press for 1 millisecond)
    key = cv2.waitKey(1)

    # If any key is pressed, break out of the loop
    if key != -1:
        break

    frame_count += 1  # Increment frame count

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Create a DataFrame from the ROI data
df = pd.DataFrame(roi_data)

# Export the DataFrame to an Excel file
df.to_excel('/Users/bandaruvinaykumar/Desktop/Projects/Projects/Opencv/Face_recognition Using Opencv & Python/Graphs/roi_data.xlsx', index=False)

# Create a graph from the ROI data
frame_count = min(frame_count, len(df['Match Result 1']))  # Ensure frame_count and DataFrame lengths match
plt.figure(figsize=(10, 6))
plt.plot(range(frame_count), df['Match Result 1'][:frame_count], label='Match Result 1')
plt.plot(range(frame_count), df['Match Result 2'][:frame_count], label='Match Result 2')
plt.xlabel('Frame')
plt.ylabel('Match Result')
plt.title('Face Recognition Match Results Over Time')
plt.legend()
plt.grid(True)

# Specify the path where you want to save the graph
graph_save_path = '/Users/bandaruvinaykumar/Desktop/Projects/Projects/Opencv/Face_recognition Using Opencv & Python/Graphs/face_recognition_graph.png'

# Save the graph to the specified path
plt.savefig(graph_save_path)

# Display the graph
plt.show()
