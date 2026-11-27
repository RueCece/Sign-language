import cv2
import mediapipe as mp
import csv

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return
    
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    ) as hands:
        while True:
            success, frame = cap.read()
            if not success:
                print("Failed to read from webcam.")
                break
            
            frame = cv2.flip(frame,1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            
            key = cv2.waitKey(1) & 0xFF
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )
                    if ord("a") <= key <= ord("z"):
                        label = chr(key).upper()
                        with open("data/landmarks.csv","a",newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow([label]+row)
                    row=[]
                    for lm in hand_landmarks.landmark:
                        row.append(lm.x)
                        row.append(lm.y)
                        row.append(lm.z)
                                                    
            cv2.putText(frame,"q to quit",(10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0),2)
            cv2.imshow("Hand Tracking", frame)
            
            if key == 27:
                break
            
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()