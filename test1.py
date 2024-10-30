import cv2 
import FaceDetectionModule as fdm 
import keras.models 
import numpy as np 
from PIL import ImageFont, ImageDraw, Image 
import time

font = ImageFont.truetype("./arial.ttf", 32)
font2 = ImageFont.truetype("./arial.ttf", 16)
cap = cv2.VideoCapture(0)
detector = fdm.FaceDetector()
model = keras.models.load_model("Model4.h5")

label = ["Bất ngờ", "Bình thường", "Buồn", "Tức giận", "Vui vẻ"]

detector = fdm.FaceDetector()
imgCanvas = np.zeros((480,300,3), np.uint8)
while True:
   
    success, img = cap.read()
    if success:
        
        hi, wi,_  = img.shape
        
        img = cv2.flip(img, cv2.ROTATE_180)
        
        imgnew, bboxs = detector.findFaces(img)
        
        imgDetection,ok,_ = detector.facegrayDetection(img)
        imgnew = np.append(img, imgCanvas, axis=1)
        if ok:
            
            imgDetection = np.array(imgDetection)
            imgDetection = imgDetection/255.0
            imgDetection = imgDetection.reshape(1,48,48,1)
            y = model.predict(imgDetection)
            result = np.argmax(y)
            
            listexp = np.interp(y[0], (0,1),(0, 100))
            listexp2  = np.interp(listexp, (0,100),(650, 870))
            img_pil = Image.fromarray(imgnew)
            draw = ImageDraw.Draw(img_pil)
            draw.text((50, 80), label[result], font = font, fill = (255,0,0))
            
            draw.text((650, 15), "Bất ngờ", font = font2, fill = (255,255,255))
            draw.text((880, 45), f'{int(listexp[0])}%', font = font2, fill = (255,255,255))
            draw.rectangle((650, 40, int(listexp2[0]), 70), fill=(34,34,178))
            draw.rectangle((650, 40, 870, 70), outline=(34,34,178))
            
            
            draw.text((650, 75), "Bình thường", font = font2, fill = (255,255,255))
            draw.text((880, 105), f'{int(listexp[1])}%', font = font2, fill = (255,255,255))
            draw.rectangle((650, 100, int(listexp2[1]), 130), fill=(35,142,107))
            draw.rectangle((650, 100, 870, 130), outline=(35,142,107))
            
            draw.text((650, 135), "Buồn", font = font2, fill = (255,255,255))
            draw.text((880, 165), f'{int(listexp[2])}%', font = font2, fill = (255,255,255))
            draw.rectangle((650, 160, int(listexp2[2]), 190), fill=(0,255,255))
            draw.rectangle((650, 160, 870, 190), outline=(0,255,255))
            
            draw.text((650, 195), "Tức giận", font = font2, fill = (255,255,255))
            draw.text((880, 225), f'{int(listexp[3])}%', font = font2, fill = (255,255,255))
            draw.rectangle((650, 220, int(listexp2[3]), 250), fill=(255,255,0))
            draw.rectangle((650, 220, 870, 250), outline=(255,255,0))
            
            draw.text((650, 255), "Vui vẻ", font = font2, fill = (255,255,255))
            draw.text((880, 285), f'{int(listexp[4])}%', font = font2, fill = (255,255,255))
            draw.rectangle((650, 280, int(listexp2[4]), 310), fill=(255,0,255))
            draw.rectangle((650, 280, 870, 310), outline=(255,0,255))
            
            # draw.text((650, 315), "Buồn", font = font2, fill = (255,255,255))
            # draw.text((880, 345), f'{int(listexp[5])}%', font = font2, fill = (255,255,255))
            # draw.rectangle((650, 340, int(listexp2[5]), 370), fill=(235,206,135))
            # draw.rectangle((650, 340, 870, 370), outline=(235,206,135))
            
            # draw.text((650, 375), "Bất ngờ", font = font2, fill = (255,255,255))
            # draw.text((880, 405), f'{int(listexp[6])}%', font = font2, fill = (255,255,255))
            # draw.rectangle((650, 400, int(listexp2[6]),430), fill=(147,20,255))
            # draw.rectangle((650, 400, 870, 430), outline=(147,20,255))
            
            imgnew = np.array(img_pil) #hiển thị ra window
            
        
        
    
    
        cv2.imshow("Img", imgnew)
    
    
    
    if cv2.waitKey(10) == ord("q"):
        break
    
    
