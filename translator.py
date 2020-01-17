
import pyttsx3
from predict import *


# Initialise Text to speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 105)
engine.setProperty('voice', 1)

window_name = "ASL"
frame_height, frame_width, roi_height, roi_width = 480, 900, 200, 200
cap = cv2.VideoCapture(0)
cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
x_start, y_start = 100, 100
sentence = ""
while True:
	ret, frame = cap.read()
	if ret is None:
		print("No Frame Captured")
		continue

	# bounding box which captures ASL sign to be detected by the system
	cv2.rectangle(frame, (x_start, y_start), (x_start + roi_width, y_start + roi_height), (255, 0, 0),3)  

	# Crop blue rectangular area(ROI)
	img1 = frame[y_start: y_start + roi_height, x_start: x_start + roi_width]
	img_ycrcb = cv2.cvtColor(img1, cv2.COLOR_BGR2YCR_CB)
	blur = cv2.GaussianBlur(img_ycrcb, (11, 11), 0)

	# lower  and upper skin color
	skin_ycrcb_min = np.array((0, 138, 67))
	skin_ycrcb_max = np.array((255, 173, 133))

	mask = cv2.inRange(blur, skin_ycrcb_min, skin_ycrcb_max)  # detecting the hand in the bounding box

	kernel = np.ones((2, 2), dtype=np.uint8)

	# Fixes holes in foreground
	mask = cv2.dilate(mask, kernel, iterations=1)

	naya = cv2.bitwise_and(img1, img1, mask=mask)
	cv2.imshow("mask", mask)
	cv2.imshow("naya", naya)
	hand_bg_rm = naya
	hand = img1

	# Control Key
	c = cv2.waitKey(1) & 0xff

	# Speak the sentence
	if len(sentence) > 0 and c == ord('s'):
		engine.say(sentence)
		engine.runAndWait()
	# Clear the sentence
	if c == ord('c') or c == ord('C'):
		sentence = ""
	# Delete the last character
	if c == ord('d') or c == ord('D'):
		sentence = sentence[:-1]

	# Put Space between words
	if c == ord('m') or c == ord('M'):
		sentence += " "

	# If  valid hand area is cropped
	if hand.shape[0] != 0 and hand.shape[1] != 0:
		conf, label = which(hand_bg_rm)
		if conf >= THRESHOLD:
			cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, .7, (0, 0, 255))
		if c == ord('n') or c == ord('N'):
			sentence += label
	
	cv2.putText(frame, sentence, (50, 70), cv2.FONT_HERSHEY_COMPLEX_SMALL, .7, (0, 0, 255))
	cv2.imshow(window_name, frame)
	# If pressed ESC break
	if c == 27:
		cap.release()
		cv2.destroyAllWindows()
		exit()
cap.release()
cv2.destroyAllWindows()




