import cv2
import threading

url = [0, 'rtsp://codonsoft:rohanchowdary@192.168.0.113:554/stream1']

class camThread(threading.Thread):
    def __init__(self, previewName, camID):
        threading.Thread.__init__(self)
        self.previewName = previewName
        self.camID = camID
    def run(self):
        print("Starting " + self.previewName)
        camPreview(self.previewName, self.camID)

def camPreview(previewName, camID):
    cv2.namedWindow(previewName)
    # if camID == 0:
    #     cam = cv2.VideoCapture(0)
    # else:
    #     cam = cv2.VideoCapture('videos/example_01.mp4')  
    cam=cv2.VideoCapture(camID)
    if cam.isOpened():
        rval, frame = cam.read()
    else:
        rval = False

    while rval:
        cv2.imshow(previewName, frame)
        rval, frame = cam.read()
        key = cv2.waitKey(20)
        if key == 27:  # exit on ESC
            break
    cv2.destroyWindow(previewName)

# Create threads as follows

# ?
thread3 = camThread("Camera 3", 'videos/example_01.mp4')

# thread1.start()
# thread2.start()
thread3.start()
print()
print("Active threads", threading.activeCount())