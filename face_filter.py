import numpy as np 
import cv2

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
eyes_cascade = cv2.CascadeClassifier('frontalEyes35x16.xml')
nose_cascade = cv2.CascadeClassifier('Nose18x15.xml')

glasses = cv2.imread("glasses.png",-1)
#mustache = cv2.imread("mustache.png",-1)
pignose = cv2.imread("pignose.png",-1)

while True:
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.5,5)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h,x:x+h]
        roi_color = frame[y:y+h,x:x+h]
        #cv2.rectangle(frame,(x,y),(x + w,y + h),(255,255,255),3)

        eyes = eyes_cascade.detectMultiScale(roi_gray,1.5,5)
        for(ex,ey,ew,eh) in eyes:
            #cv2.rectangle(roi_color,(ex,ey),(ex + ew,ey + eh),(0,255,0),3)
            roi_eyes = roi_gray[ey : ey + eh,ex : ex + ew]
            glasses2 = cv2.resize(glasses.copy(),(ew*int(1.2),eh))

            gw, gh, gc = glasses2.shape
            for i in range(0,gw):
                for j in range(0,gh):
                    if glasses2[i, j][3] != 0:
                        roi_color[ey + i, ex + j] = glasses2[i, j]

        nose = nose_cascade.detectMultiScale(roi_gray,1.5,5)
        for(nx,ny,nw,nh) in nose:
#            cv2.rectangle(roi_color,(nx,ny),(nx + nw,ny + nh),(0,0,255),3)
#            roi_nose = roi_gray[ny : ny+nh, nx : nx + nw]
#            mustache2 = cv2.resize(mustache.copy(),(nw,nh))
#            mw, mh, mc = mustache2.shape
#            for i in range(0,mw):
#               for j in range(0,mh):
#                    if mustache2[i, j][3] != 0:
#                        roi_color[ny + int(nh/2.0) + i, nx + j] = mustache2[i, j]
            
            
            pignose2 = cv2.resize(pignose.copy(),(nw,nh))
            pw, ph, pc = pignose2.shape
            for i in range(0,pw):
                for j in range(0,ph):
                    if pignose2[i, j][3] != 0:
                        roi_color[ny + i, nx + j] = pignose2[i, j] 

        frame = cv2.cvtColor(frame,cv2.COLOR_BGRA2BGR)
        cv2.imshow("Frame",frame)
        
        key_pressed = cv2.waitKey(20) & 0xFF
        if key_pressed == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()