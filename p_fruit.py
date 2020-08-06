import cv2
import numpy as np
import sys
hsv_l_r=(0,120,190)
hsv_h_r=(187,180,255)



def contains_red_lichee(img_path,win):
    img=cv2.imread(img_path)
    kernel = np.ones((15,15),np.float32)/225
    #img_work = cv2.filter2D(img,-1,kernel)
    img_work=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    img_work2=img.copy()
    
    '''
    static
    '''
    img_mask_r=cv2.inRange(img_work,hsv_l_r,hsv_h_r)
    
    #img_mask_r=cv2.erode(img_mask_r,np.ones((5,5),np.uint8),iterations = 1)
    
    img_mask_r=cv2.dilate(img_mask_r,np.ones((5,5),np.uint8),iterations = 1)
    
    contours_r, _ = cv2.findContours(img_mask_r, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    
    box_col=[]

    for cnt in contours_r:
        area=cv2.contourArea(cnt)

        if area>3000:
            #print(cnt)
            x,y,w,h = cv2.boundingRect(cnt)
            box_col.append(cv2.boundingRect(cnt))
                #cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
    if len(box_col)>0:
        cv2.putText(img,"Red lichee found",(10,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)    
    #nms_b=np.array(box_col)
    #for box in nms_b:
    #    x,y,w,h=box
    #    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
    cv2.imshow(win,img)
    #cv2.waitKey(3000)
    #cv2.destroyAllWindows()


#contains_red_lichee("imgs/350c52acce67bc6a9fc3dcbe2472bcf3.jpg")