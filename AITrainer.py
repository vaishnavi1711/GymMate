import cv2
import numpy as np
import time
from PoseModule import poseDetector

def main():
    cap = cv2.VideoCapture(0)  # Use 0 for webcam

    detector = poseDetector()
    count = 0
    dir = 0  # Direction: 0 for extending, 1 for contracting
    pTime = 0

    while True:
        success, img = cap.read()
        if not success:
            break  # Exit loop if frame loading fails

        img = cv2.resize(img, (1280, 720))
        img = detector.findPose(img, False)
        lmList = detector.findPosition(img, False)

        if len(lmList) != 0:
            # Using right arm landmarks (11, 13, 15) for angle calculation
            angle = detector.findAngle(img, 11, 13, 15)

            # Adjusted interpolation ranges for smoother progress
            # Define the full range of motion for the curl
            full_extension_angle = 170  # Angle for fully extended arm
            full_contraction_angle = 45  # Angle for fully contracted curl

            # Interpolate the angle to a percentage for the progress bar
            # The bar should gradually fill as the arm moves from full extension to full contraction
            per = np.interp(angle, [full_contraction_angle, full_extension_angle], [100, 0])
            bar = np.interp(angle, [full_contraction_angle, full_extension_angle], [100, 650])

            # Update count and direction logic based on the angle
            if angle >= full_extension_angle - 10:
                if dir == 1:
                    count += 1  # Increment count only after the arm is extended again
                dir = 0  # Reset direction when arm is extended
            elif angle <= full_contraction_angle + 10:
                dir = 1  # Update direction when arm is contracted

            color = (0, 255, 0) if dir == 1 else (255, 0, 0)

            # Draw Bar for visualization
            cv2.rectangle(img, (1100, 100), (1175, 650), color, 3)
            cv2.rectangle(img, (1100, int(bar)), (1175, 650), color, cv2.FILLED)
            cv2.putText(img, f'{int(per)}%', (1100, 75), cv2.FONT_HERSHEY_PLAIN, 4, color, 4)

            # Draw Curl Count
            cv2.rectangle(img, (0, 450), (250, 720), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, str(count), (10, 670), cv2.FONT_HERSHEY_PLAIN, 15, (255, 0, 0), 25)

        # FPS calculation
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (50, 100), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)

        cv2.imshow("AI Trainer", img)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
