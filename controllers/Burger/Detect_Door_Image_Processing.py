import cv2
import numpy as np

def detect_door_white(image, tolerance=5):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Very narrow threshold around [0,0,207]
    lower = np.array([0, 0, 207 - tolerance])
    upper = np.array([0, 0, 207 + tolerance])

    mask = cv2.inRange(hsv, lower, upper)

    # Morphological cleanup
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("❌ No door detected")
        return None, mask, image

    # Largest contour = door
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    print(f"✅ Door detected at (x={x}, y={y}, w={w}, h={h})")

    return (x, y, w, h), mask, image

# image_path = r"C:\Robotics Final Project Git In Progress\controllers\Burger\qr_detection_10.png"
# image = cv2.imread(image_path)

# bbox, mask, output = detect_door_white(image, tolerance=5)

# # Show results
# cv2.imshow("Mask (thresholded)", mask)
# cv2.imshow("Detected Door", output)
# cv2.waitKey(0)
# cv2.destroyAllWindows()