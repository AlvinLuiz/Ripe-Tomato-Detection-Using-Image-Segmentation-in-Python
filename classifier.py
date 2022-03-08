#reads data from and write data to a variety of file formats
import scipy.io as sio

#Standard import for image processing and analysis
import numpy as np
import cv2
import imutils

#Path of the image assigns to test_image variable
test_image = "tomato.jpg"

#Reads the image
tomato = cv2.imread(test_image)
#Resize to 300x300 px
tomato = cv2.resize(tomato, (300, 300))
#Converts BGR to HSV
tomatohsv = cv2.cvtColor(tomato, cv2.COLOR_BGR2HSV)

#Boundaries of color in HSV
#Color threshold for ripe tomato
lower_red = np.array([0,20,100])
upper_red = np.array([10,255,255])

#Mask created from the HSV image and boundaries 
mask_red = cv2.inRange(tomatohsv, lower_red, upper_red,)

#Finding for the Contours in the image (mask used, Contour retrieval mode, contour approximation method)
cont_red = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cont_red = imutils.grab_contours(cont_red)

for c in cont_red:
  #The area of the contoured shape
  area1 = cv2.contourArea(c)
  #Only accepts object with area greater than 4000px to prevent detecting noises
  if area1 > 4000:
    #draw rectangle to enclose identified object
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02*peri, True)
    x, y, w, h,  = cv2.boundingRect(approx)
    cv2.rectangle(tomato, (x, y), (x+w, y+h), (0,255,0), 5)

    #Acquiring center of the contoured area
    M = cv2.moments(c)
    cx = int(M["m10"]/M["m00"])
    cy = int(M["m01"]/M["m00"])

    #Drawing a circle and labeling the area
    cv2.circle(tomato,(cx,cy),7,(255,255,255),-1)
    cv2.putText(tomato, "ripe", (cx-50, cy-10), cv2.FONT_HERSHEY_SIMPLEX,1.5, (255,255,0), 2)

#Show output image
cv2.imshow("Result", tomato)
