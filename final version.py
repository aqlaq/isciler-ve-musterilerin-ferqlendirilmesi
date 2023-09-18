import cv2
import numpy as np

# Load YOLO model and labels
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = ["person"]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load video
cap = cv2.VideoCapture(r"C:\Users\User\OneDrive\Desktop\data\first video.mp4")

# Initialize a dictionary to store names/IDs for each person
person_names = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape

    # Detect objects using YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Information to draw bounding boxes
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:  # Check if it's a person
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # Apply non-max suppression to eliminate redundant overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Define the threshold value for blue tie detection
    threshold = 100  # Adjust this value based on experimentation

    # Loop through the detected people
    for i in indices:
        box = boxes[i]
        x, y, w, h = box
        label = str(classes[class_ids[i]])

        tie = 1

        # Check if person is wearing white uniform and distinguish based on position
        if y + h < height // 2:  # Assuming working people are mostly on the top half of the frame
            color = (255, 255, 255)  # White uniform

            # Further narrow down workers to those on the top third of the frame
            if y + h < height // 3:
                # Draw blue tie workers in white
                # Convert the region of interest to HSV color space
                roi = frame[y:y + h, x:x + w]
                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

                # Define the range of blue color in HSV
                lower_blue = np.array([100, 100, 50])
                upper_blue = np.array([130, 255, 255])

                # Create a mask to detect blue color
                mask = cv2.inRange(hsv_roi, lower_blue, upper_blue)

                # Count the number of blue pixels in the mask
                blue_pixel_count = cv2.countNonZero(mask)

                # If there are a significant number of blue pixels, consider it a blue tie
                if blue_pixel_count > threshold:  # You'll need to define the threshold
                    color = (255, 255, 255)  # White color
                else:
                    tie = 0


            if tie == 0:
                color = (0, 0, 255)  # Red color for workers without blue ties
                       #Assign a unique name/ID to each person
                worker_label = "QANUNA ZIDD ISCI"
                cv2.putText(frame, worker_label, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)
            else:
                worker_label = "ISCI"
                cv2.putText(frame, worker_label, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)
        else:
            color = (0, 255, 0)  # Green color for customers

            # Rename customers as "Customer"
            customer_label = "MUSTERI"
            cv2.putText(frame, customer_label, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
