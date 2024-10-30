import cv2 
import mediapipe as mp 
import time 


class FaceDetector():
    def __init__(self, minDetection = 0.75):
        self.minDetection = minDetection
    
        self.mpFaceDetection = mp.solutions.face_detection

        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetection)

    # Tìm mặt 
    def findFaces(self, img, draw = True):
        # Chuyển ảnh từ BGR thành RGB 
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        
        self.results = self.faceDetection.process(imgRGB)
        self.bboxs = []
        # Hiển thị khung nhận dạng
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = (int(bboxC.xmin * iw), int(bboxC.ymin * ih),int(bboxC.width * iw), int(bboxC.height * ih))
                
                self.bboxs.append([id,bbox, detection.score])
                if draw:
                    cv2.rectangle(img, bbox, (255,0,255), 2)
                
                    cv2.putText(img, str(int((detection.score[0]*100))),(bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 1)
        
        return img, self.bboxs
    # Tìm mặt và chuyển ảnh mặt về ảnh xám có size 48x48 
    def facegrayDetection(self, img):
        ok = True
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        self.results = self.faceDetection.process(imgRGB)
        if self.results.detections:
            
            detection = self.results.detections[0]
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape
            bbox = (int(bboxC.xmin * iw), int(bboxC.ymin * ih),int(bboxC.width * iw), int(bboxC.height * ih))
            # Cắt khung chứa mặt 
            cropped_face = imgRGB[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
            
            if len(cropped_face[0])!=0:
                # Thay đổi kích thước 
                cropped_face = cv2.resize(cropped_face,(48,48))
                cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_RGB2GRAY)
                return cropped_face, ok, bbox
            else :
                ok = False
                return None, ok, None
        else:
            ok = False
            return None, ok, None
    
   

def main():
    cap = cv2.VideoCapture(0)

    pTime = 0
    detector = FaceDetector(0.4)
    bboxs = []
    while True:
        success, img = cap.read()
        cTime = time.time()

        fps = 1/(cTime-pTime)
        
        pTime = cTime
        img, bboxs = detector.findFaces(img)
        imgDetection, ok = detector.facegrayDetection(img)
        
        cv2.putText(img, f"FPS: {int(fps)}",(20,50), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)
        
        
        
        
        if (ok):
            cv2.imshow("Image2", imgDetection)
        cv2.imshow("Image1", img)
        
    
    
        if cv2.waitKey(2) == ord("q"):
            break
  
    
if __name__ == "__main__":
    main()
    
    