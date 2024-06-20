import cv2
import numpy as np
import os
import time
import pandas as pd

# Define the duration to run the face detection window (in seconds)
RUN_DURATION = 5  # Duration to run the face detection

size = 10
datasets = "C:\\Users\\KIIT\\OneDrive\\Desktop\\Python\\datasets"
print('Training...')

(images, labels, names, id) = ([], [], {}, 0)

# Add paths to Haar cascade classifier files
haarcascade_default_path = "C:\\Users\\KIIT\\OneDrive\\Desktop\\Python\\Project\\haarcascade_frontalface_default.xml"
haarcascade_alt_path = "C:\\Users\\KIIT\\OneDrive\\Desktop\\Python\\Project\\haarcascade_frontalface_alt.xml"
haarcascade_alt2_path = "C:\\Users\\KIIT\\OneDrive\\Desktop\\Python\\Project\\haarcascade_frontalface_alt2.xml"

# Load Haar cascade classifiers
face_cascade_default = cv2.CascadeClassifier(haarcascade_default_path)
face_cascade_alt = cv2.CascadeClassifier(haarcascade_alt_path)
face_cascade_alt2 = cv2.CascadeClassifier(haarcascade_alt2_path)

# Load dataset images
for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(datasets, subdir)
        for filename in os.listdir(subjectpath):
            path = os.path.join(subjectpath, filename)
            label = id
            img = cv2.imread(path, 0)
            # Resize the image to ensure all images have the same dimensions
            img = cv2.resize(img, (130, 100))
            images.append(img)
            labels.append(int(label))
        id += 1

(images, labels) = [np.array(lis) for lis in [images, labels]]

# Train the model
model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, labels)

webcam = cv2.VideoCapture(0)
start_time = time.time()

output_data = []

while time.time() - start_time < RUN_DURATION:
    # Capture frames from the webcam
    ret, im = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    # Detect faces using the classifiers
    faces_default = face_cascade_default.detectMultiScale(gray, 1.3, 5)
    faces_alt = face_cascade_alt.detectMultiScale(gray, 1.3, 5)
    faces_alt2 = face_cascade_alt2.detectMultiScale(gray, 1.3, 5)
    
    # Combine all detected faces
    faces_list = [faces_default, faces_alt, faces_alt2]
    
    # Filter out empty arrays and concatenate the rest
    non_empty_faces = [face for face in faces_list if len(face) > 0]
    
    if non_empty_faces:
        faces = np.concatenate(non_empty_faces)
        # Process the first detected face
        (x, y, w, h) = faces[0]
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (130, 100))
        prediction = model.predict(face_resize)
        
        # Draw rectangle and display confidence level
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)
        
        if prediction[1] < 800:
            name = names[prediction[0]]
            confidence = prediction[1]
            cv2.putText(im, f'{name} - {confidence:.0f}', (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 2)
            output_data.append([time.time(), name, confidence])
        else:
            cv2.putText(im, "Unknown", (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
    
    # Show the live video with face detection
    cv2.imshow('Face Recognition', im)
    
    # Break the loop if 'c' is pressed
    if cv2.waitKey(10) == ord('c'):
        break


webcam.release()
cv2.destroyAllWindows()

# Convert output data to DataFrame
df = pd.DataFrame(output_data, columns=['Timestamp', 'Name', 'Confidence'])

# Group by 'Name' and select the row with the maximum 'Confidence'
max_confidence_df = df.loc[df.groupby('Name')['Confidence'].idxmax()]

# Output file path
output_file = 'face_recognition_output.xlsx'

# Use ExcelWriter to append data to existing file
with pd.ExcelWriter(output_file, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
    # Append max_confidence_df to the existing 'Sheet1'
    max_confidence_df.to_excel(writer, index=False, header=False, sheet_name='Sheet1')


print(f"Output saved to '{output_file}'")
