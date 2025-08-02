import cv2
import numpy as np

# Global list to store clicked HSV values
clicked_hsv = []

def pick_color(event, x, y, flags, param):
    global clicked_hsv
    if event == cv2.EVENT_LBUTTONDOWN:
        hsv = param[y, x]  # HSV value at clicked point
        clicked_hsv.append(hsv)
        print(f"Clicked HSV: {hsv}")

def analyze_door_image(image_path):
    global clicked_hsv
    clicked_hsv = []

    # Load image
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    cv2.namedWindow("Door Image")
    cv2.setMouseCallback("Door Image", pick_color, hsv)

    print("👉 Click on the white part of the door to collect HSV values (press 'q' when done).")

    while True:
        cv2.imshow("Door Image", image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()

    if not clicked_hsv:
        print("No points clicked.")
        return None

    clicked_hsv = np.array(clicked_hsv)
    hsv_min = np.min(clicked_hsv, axis=0)
    hsv_max = np.max(clicked_hsv, axis=0)

    print(f"\nSuggested HSV Range for door white part:")
    print(f"Lower: {hsv_min}")
    print(f"Upper: {hsv_max}")

    return hsv_min, hsv_max

# Example usage:
hsv_min, hsv_max = analyze_door_image(r"C:\Robotics Final Project Git In Progress\controllers\Burger\qr_detection_10.png")