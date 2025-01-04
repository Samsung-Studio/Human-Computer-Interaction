import cv2
import numpy as np

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame for natural interaction
    frame = cv2.flip(frame, 1)

    # Define a region of interest (ROI) for the hand
    roi = frame[100:400, 100:400]
    cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 2)

    # Convert ROI to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Apply thresholding
    _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour, assuming it's the hand
        max_contour = max(contours, key=cv2.contourArea)

        # Draw the contour
        cv2.drawContours(roi, [max_contour], -1, (255, 0, 0), 2)

        # Convex hull and defects
        hull = cv2.convexHull(max_contour, returnPoints=False)
        defects = cv2.convexityDefects(max_contour, hull)

        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(max_contour[s][0])
                end = tuple(max_contour[e][0])
                far = tuple(max_contour[f][0])

                # Draw points and lines for convexity defects
                cv2.line(roi, start, end, (0, 255, 0), 2)
                cv2.circle(roi, far, 5, (0, 0, 255), -1)

    # Display the original frame and the thresholded ROI
    cv2.imshow("Original Frame", frame)
    cv2.imshow("Hand Detection", thresh)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()