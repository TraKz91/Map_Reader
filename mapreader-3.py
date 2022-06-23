#Registration Number: 1905670


import sys, cv2, math
import numpy as np

# -------------------------------------------------------------------------------
# Main program.
# -------------------------------------------------------------------------------

# Ensure we were invoked with a single argument.

if len(sys.argv) != 2:
    print("Usage: %s <image-file>" % sys.argv[0], file=sys.stderr)
    exit(1)

font = cv2.FONT_HERSHEY_COMPLEX

# -------------------------------------------------------------------------------
# This part is to segment the image from the blue background
# -------------------------------------------------------------------------------
img = cv2.imread(sys.argv[1])  # read the file which is an image
blur = cv2.blur(img, (5, 5))  # blur the image
hsv_img = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)  # convert the image to hsv
white_1 = np.array([0, 0, 80], dtype=np.uint8)  # minimum value of white
white_2 = np.array([180, 160, 255], dtype=np.uint8)  # maximum value of white
mask = cv2.inRange(hsv_img, white_1, white_2)  # mask for white color
result = cv2.bitwise_and(img, img, mask=mask)   # Apply the mask on the image
kernel = np.ones((5, 5), np.uint8)
open = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)  # eliminate the noises outside contours
close = cv2.morphologyEx(open, cv2.MORPH_CLOSE, kernel)  # eliminate the noises inside contours
rgb = cv2.cvtColor(close, cv2.COLOR_HSV2RGB)  # convert to the image to rgb
gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)  # convert to the image to gray
contours, bim = cv2.findContours(gray, cv2.RETR_EXTERNAL,  # find the contours in the image
                                 cv2.CHAIN_APPROX_SIMPLE)
# -------------------------------------------------------------------------------


# -------------------------------------------------------------------------------
# This part is to rotate the map which was segmented from the blue backgroung
# -------------------------------------------------------------------------------
# Find the position of the corner of the rectangle
max_dist_cont = 0.1 * cv2.arcLength(contours[0], True)
approximative = cv2.approxPolyDP(contours[0], max_dist_cont, True)

# Assign corners to variables
c0 = approximative[1][0]
c1 = approximative[2][0]
c2 = approximative[3][0]
c3 = approximative[0][0]

# calculate the length of each height and width
width1 = math.sqrt((c0[0] - c3[0]) ** 2 + (c0[1] - c3[1]) ** 2)
width2 = math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)
height1 = math.sqrt((c0[0] - c1[0]) ** 2 + (c0[1] - c1[1]) ** 2)
height2 = math.sqrt((c2[0] - c3[0]) ** 2 + (c2[1] - c3[1]) ** 2)

# ---------------------------------------------------------------------------------------------------------
# The six lines of code under are adapted from : https://theailearner.com/tag/cv2-getperspectivetransform/
# ---------------------------------------------------------------------------------------------------------

# Find the maximum of the width and the height of the image
maximum_X = max(int(width1), int(width2))
maximum_Y = max(int(height1), int(height2))

# Assign original and new coordinates of the corners to variables
original_corners = np.float32([c0, c1, c2, c3])
new_corners = np.float32([[0, 0], [0, maximum_Y - 1], [maximum_X - 1, maximum_Y - 1], [maximum_X - 1, 0]])

# Apply the transformation and get the final form of the image
persp = cv2.getPerspectiveTransform(original_corners, new_corners)
output = cv2.warpPerspective(img, persp, (maximum_X, maximum_Y), flags=cv2.INTER_LINEAR)
# -------------------------------------------------------------------------------
cv2.imwrite("test.jpg", output) #save the new image
# -------------------------------------------------------------------------------


# -------------------------------------------------------------------------------
# This part is to segement the triangle from the map
# -------------------------------------------------------------------------------
img1 = output  # get the original image transformed
blur = cv2.blur(img1, (7, 7))  # blur the image
hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)  # convert to hsv
red_1 = np.array([0, 40, 170], dtype=np.uint8)  # minimum value of red
red_2 = np.array([180, 255, 255], dtype=np.uint8)  # minimum value of red
mask1 = cv2.inRange(hsv, red_1, red_2)  # mask for red color
res1 = cv2.bitwise_and(img1, img1, mask=mask1)  # Apply the mask on the image
rgb1 = cv2.cvtColor(res1, cv2.COLOR_HSV2RGB)  # convert to rgb
gray1 = cv2.cvtColor(rgb1, cv2.COLOR_RGB2GRAY)  # convert to gray
contoursbis, bimbis = cv2.findContours(gray1, cv2.RETR_EXTERNAL,  # find the contours
                                       cv2.CHAIN_APPROX_SIMPLE)
# -------------------------------------------------------------------------------


# -------------------------------------------------------------------------------
# Ths part is to find and claculate the bearing
# -------------------------------------------------------------------------------
#Use to find the corners of the triangle
epsi = 0.1 * cv2.arcLength(contoursbis[0], True)
approx1 = cv2.approxPolyDP(contoursbis[0], epsi, True)

# assign position of corners of the triangle to variable
x2 = approx1[0][0][0]
y2 = approx1[0][0][1]
x1 = approx1[1][0][0]
y1 = approx1[1][0][1]
x3 = approx1[2][0][0]
y3 = approx1[2][0][1]

# calculate the 3 sides of the triangle
dist1 = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
dist2 = math.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2)
dist3 = math.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)

# determine the pointer by finding the 2 longest side of the triangle
if dist1 > dist3 and dist2 > dist3:
    result = approx1[1][0]
    a1 = approx1[0][0][0]
    b1 = approx1[0][0][1]
    a2 = approx1[2][0][0]
    b2 = approx1[2][0][1]
elif dist1 > dist2 and dist3 > dist2:
    result = approx1[0][0]
    a1 = approx1[1][0][0]
    b1 = approx1[1][0][1]
    a2 = approx1[2][0][0]
    b2 = approx1[1][0][1]
else:
    result = approx1[2][0]
    a1 = approx1[0][0][0]
    b1 = approx1[0][0][1]
    a2 = approx1[1][0][0]
    b2 = approx1[1][0][1]

shp = np.shape(img1)  # the size of the image

# find the position of the middle of the side that is opposite of the pointer
#and it to the array "middle"
new_positionx = (a1 + a2) / 2
new_positiony = (b1 + b2) / 2
middle = []
middle.append(new_positionx)
middle.append(new_positiony)

# find the coordinate of the middle of the triangle to calculate the bearing
center_x = result[0] - new_positionx
center_y = result[1] - new_positiony

#-------------------------------------------------------------------------------
# math.atan2 were adapted fromhttps://www.geeksforgeeks.org/atan2-function-python/
#-------------------------------------------------------------------------------

teta = (math.atan2(center_y, center_x)) + (math.pi / 2) #Bearing in radian
teta_in_degree = (teta * 180) / math.pi #Bearing converted to degree

# -------------------------------------------------------------------------------


print("The filename to work on is %s." % sys.argv[1])
#we divide the result by the size of the image, depending if the the width or height, to have the x position
xpos = result[0] / shp[0] #postion of x
ypos = result[1] / shp[1] #position of y
hdg = teta_in_degree #Bearing

# Output the position and bearing for the user and in the right format for harness.py.
print("POSITION %.3f %.3f" % (xpos, ypos))
print("BEARING %.1f" % hdg)
# -------------------------------------------------------------------------------
