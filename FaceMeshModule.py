import cv2
import mediapipe as mp
import time
from mediapipe.python.solutions.drawing_utils import RED_COLOR


class FaceMeshDetector():

    def __init__(self, staticMode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5, refineLandmarks=False):
        # Initialization parameters for face mesh detection
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon
        self.refineLandmarks = refineLandmarks

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(static_image_mode=self.staticMode,
            max_num_faces=self.maxFaces,
            refine_landmarks=self.refineLandmarks,
            min_detection_confidence=self.minDetectionCon,
            min_tracking_confidence=self.minTrackCon)# MediaPipe FaceMesh configuration
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2, color=RED_COLOR)

    def findFaceMesh(self, img, draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)# MediaPipe requires RGB input
        self.results = self.faceMesh.process(self.imgRGB)# Process frame with FaceMesh
        faces = []
        if self.results.multi_face_landmarks:
            for numFace, faceLms in enumerate(self.results.multi_face_landmarks):
                if draw:                                   # Draw face mesh connections on the image
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS,
                                          self.drawSpec, self.drawSpec)
                # print(f'Face Number:{numFace}')
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    # Convert normalized coordinates to pixel values
                    # print(lm)
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    # cv2.putText(img, f'{str(id)}', (x, y), cv2.FONT_HERSHEY_PLAIN,
                    #             0.3, (0, 255, 0), 1)
                    # print(id, x, y)
                    face.append([x,y])# Store landmark coordinates
            faces.append(face)        # Add face landmarks to list

        return img , faces




def main():
    cap = cv2.VideoCapture(0)  # Initialize webcam
    pTime = 0
    detector = FaceMeshDetector()
    while True:
        success, img = cap.read()
        img, faces = detector.findFaceMesh(img)
        if len(faces) != 0:
            print(len(faces))# Output number of detected faces
        # FPS calculation and display
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS:{int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                    3, (0, 255, 0), 3)
        img_resized = cv2.resize(img, (600, 600)) # Resize for display window
        cv2.imshow("Image", img_resized)
        cv2.waitKey(1)



if __name__ =="__main__":
    main()