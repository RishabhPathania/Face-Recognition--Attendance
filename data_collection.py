import cv2
import os

# Function to create a directory if it doesn't exist
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Function to collect face data from webcam
def collect_face_data(dataset_name):
    dataset_path = os.path.join('datasets', dataset_name)
    create_directory(dataset_path)

    # Load Haar cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Open webcam
    webcam = cv2.VideoCapture(0)
    if not webcam.isOpened():
        print("Error: Unable to access webcam.")
        return

    count = 0
    max_images = 30

    # Capture images for face dataset
    while count < max_images:
        print("Image", count + 1)
        ret, frame = webcam.read()
        if not ret:
            print("Error: Unable to capture frame.")
            break

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=4)

        if len(faces) == 0:
            # If no faces are detected, continue to the next frame
            print("No faces detected.")
            continue

        for (x, y, w, h) in faces:
            face = frame[y:y + h, x:x + w]
            # Resize face image to a standard size
            face_resize = cv2.resize(face, (130, 100))
            # Save face image to dataset directory
            cv2.imwrite(os.path.join(dataset_path, f'{dataset_name}_{count}.png'), face_resize)
            count += 1

            # Draw rectangle around the detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Display captured frame with detected faces
        cv2.imshow('Collecting Face Data', frame)

        # Wait for a key press to capture the next image
        key = cv2.waitKey(1)
        if key == 27 or count >= max_images:
            break

    # Release webcam and close OpenCV windows
    webcam.release()
    cv2.destroyAllWindows()

# Main function to prompt user for dataset name and roll number, and initiate data collection
def main():
    dataset_name = input("Enter the name for data collection: ")
    collect_face_data(dataset_name)

# Entry point of the script
if __name__ == "__main__":
    main()
