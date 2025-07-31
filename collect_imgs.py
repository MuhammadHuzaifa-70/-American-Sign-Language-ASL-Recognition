import os
import cv2

# Constants
DATA_DIR = './data'
NUM_CLASSES = 24
DATASET_SIZE = 200

# Create data directory if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)

# Initialize video capture
cap = cv2.VideoCapture(0)  # Change the index if needed (0 for built-in camera)

# Loop through each class
for class_index in range(NUM_CLASSES):
    class_dir = os.path.join(DATA_DIR, str(class_index))
    os.makedirs(class_dir, exist_ok=True)

    print(f'Collecting data for class {class_index}')

    # Display prompt to start capturing
    input("Press Enter when ready to start capturing...")

    # Capture and save images
    for img_index in range(DATASET_SIZE):
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image!")
            break

        cv2.imshow('frame', frame)
        cv2.waitKey(25)

        # Save image
        img_path = os.path.join(class_dir, f'00{img_index}.jpg')
        cv2.imwrite(img_path, frame)

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()
