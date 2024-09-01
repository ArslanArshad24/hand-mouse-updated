import cv2
import mediapipe as mp
import util
import pyautogui
from pynput.mouse import Button, Controller

mouse = Controller()

mphands = mp.solutions.hands
hand = mphands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
) 

screen_w, screen_h = pyautogui.size()
def move_mouse(index_finger_tip):
    if index_finger_tip is not None:
        x = int(index_finger_tip.x * screen_w)
        y = int(index_finger_tip.y *screen_h)
        pyautogui.moveTo(x,y)
        
def left_click(landmark_list,thumb_index_dis):
    return(util.get_angle(landmark_list[5],landmark_list[6],landmark_list[8]) < 50 and
            util.get_angle(landmark_list[9],landmark_list[10],landmark_list[12]) >90 and
            thumb_index_dis >50
        )
def right_click(landmark_list,thumb_index_dis):
    return(util.get_angle(landmark_list[5],landmark_list[6],landmark_list[8]) > 90 and
            util.get_angle(landmark_list[9],landmark_list[10],landmark_list[12]) < 50 and
            thumb_index_dis >50
        )

def press_down(landmark_list,thumb_index_dis):
    return(util.get_angle(landmark_list[5],landmark_list[6],landmark_list[8]) > 90 and
        thumb_index_dis <50
    )
    
def find_finger_tip(processed):
    if processed.multi_hand_landmarks:
        hand_landmarks = processed.multi_hand_landmarks[0]
        return hand_landmarks.landmark[mphands.HandLandmark.INDEX_FINGER_TIP]
    return None

def detect_gesture(frame,landmark_list,processed):
    if len(landmark_list) >= 20:
        index_finger_tip = find_finger_tip(processed)
        # print(index_finger_tip)
        thumb_index_dis = util.get_distance((landmark_list[4],landmark_list[5]))
        if thumb_index_dis <50 and util.get_angle(landmark_list[5],landmark_list[6],landmark_list[8]) >90:
            move_mouse(index_finger_tip)
        elif left_click(landmark_list,thumb_index_dis):
            mouse.press(Button.left)
            mouse.release(Button.left)
            cv2.putText(frame,"Left Click",(50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        elif right_click(landmark_list,thumb_index_dis):
            mouse.press(Button.right)
            mouse.release(Button.right)
            cv2.putText(frame,"Right Click",(50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        # if press_down(landmark_list,thumb_index_dis):
        #     pyautogui.press('down')
        #     cv2.putText(frame,"Clicked",(50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

def main():
    cap = cv2.VideoCapture(0)
    draw = mp.solutions.drawing_utils
    try:
        while cap.isOpened():
            if cv2.waitKey(1)==ord('q'):
                break
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame,1)
            framergb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            processed = hand.process(framergb)
            landmark_list = []
            if processed.multi_hand_landmarks:
                hand_landmarks = processed.multi_hand_landmarks[0]
                draw.draw_landmarks(frame, hand_landmarks,mphands.HAND_CONNECTIONS)
                for ls in hand_landmarks.landmark:
                    landmark_list.append((ls.x, ls.y))
            detect_gesture(frame,landmark_list,processed)
            
            frame = cv2.imshow("Hand Mouse",frame)
    
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()