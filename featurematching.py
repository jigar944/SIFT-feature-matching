import cv2
import numpy as np
# read the images

import test
def HarrisCornerDetection(image1): 
     
    gaussian_blur_image = cv2.GaussianBlur(image1,(3,3),2)

    cv2.namedWindow("Blur", cv2.WINDOW_NORMAL)
    cv2.imshow("Blur",gaussian_blur_image)
    
    #sobel kernal x
    I_x = cv2.Sobel(gaussian_blur_image,cv2.CV_64F,1,0,ksize=3)
    cv2.namedWindow("Sobel_x", cv2.WINDOW_NORMAL)
    cv2.imshow("Sobel_x",I_x/255)
    
    #sobel kernal y
    I_y = cv2.Sobel(gaussian_blur_image,cv2.CV_64F,0,1,ksize=3)
    cv2.namedWindow("Sobel_y", cv2.WINDOW_NORMAL)
    cv2.imshow("Sobel_y",I_y/255)
    
    Ixx = np.square(I_x)
    cv2.namedWindow("Sobel_xx", cv2.WINDOW_NORMAL)
    cv2.imshow("Sobel_xx",Ixx/255)
    Iyy =np.square(I_y)
    cv2.namedWindow("Sobel_yy", cv2.WINDOW_NORMAL)
    cv2.imshow("Sobel_yy",Iyy/255)
    
    IxIy = np.multiply(I_x,I_y)
    cv2.namedWindow("IxIy", cv2.WINDOW_NORMAL)
    cv2.imshow("IxIy",IxIy/255)
    
    
    window_size=3
    height,width = image1.shape
    final_image = np.zeros((height,width))
    offset = int( window_size / 2 )
    for y in range(offset, height-offset):
        for x in range(offset, width-offset):
            
            Sxx = np.sum(Ixx[y-offset:y+1+offset, x-offset:x+1+offset])
            Syy = np.sum(Iyy[y-offset:y+1+offset, x-offset:x+1+offset])
            Sxy = np.sum(IxIy[y-offset:y+1+offset, x-offset:x+1+offset])
    
            H = np.array([[Sxx,Sxy],[Sxy,Syy]])
    
            det=np.linalg.det(H)
            tr=np.matrix.trace(H)
            R=det-0.04*(tr**2)
            final_image[y-offset, x-offset]=R
    
    #   Step 6 - Apply a threshold
    cv2.normalize(final_image, final_image, 0, 1, cv2.NORM_MINMAX)
    corners = []
    for y in range(10,height-9):
        for x in range(10,width-9):
                maxx = final_image[y,x]               
                neighbouhood = final_image[y-9:y+8,x-9:x+8]
                if np.amax(neighbouhood)<=maxx:
                    corners.append((x,y))

    return final_image,corners

def findkeyPoints(img_1,img_2,img_3,corner,final_image):
    image_h = img_2.shape[0]
    image_w = img_2.shape[1]

    kernal_size=3
    height=width=kernal_size//2
    keypoint=[]
    
    #gaussian_kernal = create_gaussianfilter(16,1.5)
    #y = np.sum(np.array(gaussian_kernal))
    
    for i in range(8,image_h-8):
        for j in range(8,image_w-8):
            main_point = img_2[i][j]     
            kernalOf1 = img_1[i-height:i+height+1,j-width:j+width+1]
            kernalOf2 = img_2[i-height:i+height+1,j-width:j+width+1]
            kernalOf3 = img_3[i-height:i+height+1,j-width:j+width+1]
            max_value = max([np.max(kernalOf1,initial=0),np.max(kernalOf2,initial=0),np.max(kernalOf3,initial=0)])
            min_value = min([np.min(kernalOf1,initial=0),np.min(kernalOf2,initial=0),np.min(kernalOf3,initial=0)])
            if(main_point>=max_value or main_point<=min_value) and final_image[i,j]>0.14:
                if(j,i) in corner:
                    keypoint.append((j,i))
   
    return keypoint

def orientation_asssignement(keys,img1,factor):
    
    points = []
    magnitude = []
    orientation = []
    for y,x in keys:
        kernalOf1 = img1[x-8:x+8,y-8:y+8]
        sobel_x = cv2.Sobel(kernalOf1,cv2.CV_64F,1,0,ksize=3)
        sobel_y = cv2.Sobel(kernalOf1,cv2.CV_64F,0,1,ksize=3)
        mag = np.array(np.sqrt((sobel_x**2)+(sobel_y**2)))
        ori = np.array(np.arctan2(sobel_y, sobel_x) * (180 / np.pi))
        bins = []
        for z in range(0,360,10):
            row,cols = np.where((ori>=z) & (ori<(z+10)))
            bins.append(np.sum(mag[row,cols]))
        bins = np.argsort(bins)
        point = cv2.KeyPoint(factor*y,factor*x,1,float(bins[9]*10))
        points.append(point)
        magnitude.append(mag)
        orientation.append(ori)
            
    return points,magnitude,orientation


def calculate_descriptor(mags,orie):
    
    des = []
    for y,x in zip(mags,orie):
        bins = []
        for i in range(0,16,4):
            for j in range(0,16,4):
                small_block_mag = y[i:i+4,j:j+4]
                small_block_ori = x[i:i+4,j:j+4]
                for z in range(0,360,45):
                    row = np.count_nonzero((small_block_ori>=z) & (small_block_ori<(z+45)))
                    row = row/16
                    bins.append(row)
        des.append(bins)
    return des
    
    
    

img1 = cv2.imread('Yosemite2.jpg') 
img1_1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
Harris1,corner = HarrisCornerDetection(img1_1)

img1 = cv2.imread('Yosemite2.jpg',cv2.IMREAD_GRAYSCALE) 


image_1_1 = cv2.GaussianBlur(img1,(3,3),1.6)
image_1_2 = cv2.GaussianBlur(img1,(3,3),2)
image_1_3 = cv2.GaussianBlur(img1,(3,3),2.3)
image_1_4 = cv2.GaussianBlur(img1,(3,3),1.5)
DOG_1_1 = image_1_2-image_1_1
DOG_1_2 = image_1_3 - image_1_2
DOG_1_3 = image_1_3 - image_1_4

key1 = findkeyPoints(DOG_1_1, DOG_1_2, DOG_1_3, corner,Harris1)
key1,magnitude,orientation = orientation_asssignement(key1,img1,1)


img1_half = cv2.resize(img1,None,fx=1/2,fy=1/2)
image_1_1 = cv2.GaussianBlur(img1_half,(3,3),1.6)
image_1_2 = cv2.GaussianBlur(img1_half,(3,3),2)
image_1_3 = cv2.GaussianBlur(img1_half,(3,3),2.3)
image_1_4 = cv2.GaussianBlur(img1_half,(3,3),1.5)
DOG_1_1 = image_1_2-image_1_1
DOG_1_2 = image_1_3 - image_1_2
DOG_1_3 = image_1_3 - image_1_4

key1_half = findkeyPoints(DOG_1_1, DOG_1_2, DOG_1_3, corner,Harris1)
key1_half,magnitude_half,orientation_half = orientation_asssignement(key1_half,img1_half,2)

key1 = key1 + key1_half
magnitude = magnitude + magnitude_half
orientation = orientation + orientation_half

kp_img = cv2.drawKeypoints(img1, key1, None, color=(0, 255, 0),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

des1 = calculate_descriptor(magnitude,orientation)

cv2.namedWindow("SIFT_1", cv2.WINDOW_NORMAL)
cv2.imshow("SIFT_1", kp_img)

print("done image 1")
print(len(key1),len(des1))

img2 = cv2.imread('Yosemite1.jpg') 
img1_1 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
Harris1,corner = HarrisCornerDetection(img1_1)

img2 = cv2.imread('Yosemite1.jpg',cv2.IMREAD_GRAYSCALE) 


image_1_1 = cv2.GaussianBlur(img2,(3,3),1.6)
image_1_2 = cv2.GaussianBlur(img2,(3,3),2)
image_1_3 = cv2.GaussianBlur(img2,(3,3),2.3)
image_1_4 = cv2.GaussianBlur(img2,(3,3),1.5)
DOG_1_1 = image_1_2 - image_1_1
DOG_1_2 = image_1_3 - image_1_2
DOG_1_3 = image_1_3 - image_1_4


key2 = findkeyPoints(DOG_1_1, DOG_1_2, DOG_1_3, corner,Harris1)
key2,magnitude,orientation = orientation_asssignement(key2,img2,1)

img2_half = cv2.resize(img2, None,fx=1/2,fy=1/2)
image_1_1 = cv2.GaussianBlur(img2_half,(3,3),1.6)
image_1_2 = cv2.GaussianBlur(img2_half,(3,3),2)
image_1_3 = cv2.GaussianBlur(img2_half,(3,3),2.3)
image_1_4 = cv2.GaussianBlur(img2_half,(3,3),1.5)
DOG_1_1 = image_1_2 - image_1_1
DOG_1_2 = image_1_3 - image_1_2
DOG_1_3 = image_1_3 - image_1_4

key2_half = findkeyPoints(DOG_1_1, DOG_1_2, DOG_1_3, corner,Harris1)
key2_half,magnitude_half,orientation_half = orientation_asssignement(key2_half,img2_half,2)

key2 = key2 + key2_half
magnitude = magnitude + magnitude_half
orientation = orientation + orientation_half

kp_img1 = cv2.drawKeypoints(img2, key2, None, color=(0, 255, 0),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
des2 = calculate_descriptor(magnitude,orientation)
cv2.namedWindow("SIFT_2", cv2.WINDOW_NORMAL)
cv2.imshow("SIFT_2", kp_img1)

print("done image 2")
print(len(key2),len(des2))


matches =[]
for z,y in enumerate(des1):
    matchPairs = []
    for x in des2:
        a = np.subtract(x,y)
        a = np.sum(np.square(a))
        matchPairs.append(a)
    minn = np.argsort(matchPairs) 
    print("image 1:",key1[z].pt)
    print("image 2:",key2[minn[0]].pt)
    print(matchPairs[minn[0]])
    if (matchPairs[minn[0]]/matchPairs[minn[1]])*1.2<0.8:
        print(matchPairs[minn[0]]/matchPairs[minn[1]])
        dmatch = cv2.DMatch(z,minn[0],matchPairs[minn[0]])
        matches.append(dmatch)
    if len(matches)==30:
        break
print(len(des1),len(des2))
print(len(matches))
matched_img = cv2.drawMatches(img1, key1, img2, key2, matches, img1,flags=2)
cv2.namedWindow("final_image", cv2.WINDOW_NORMAL)
cv2.imshow("final_image", matched_img)
cv2.imwrite("result.jpg", matched_img)



cv2.waitKey(0)