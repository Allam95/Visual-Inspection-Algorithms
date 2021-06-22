import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import *
from itertools import groupby
from operator import itemgetter

true_positives = []
false_positives = []
for image_number in range(0, 22):

        
        

        # 22 teeth    
        #reading images after output from processdefectssaving.py
        #filefiletosave_top_canny variabke is the filename reading step here ,not for saving.

        #filetosave_top_canny='D:/Uni/Vehcom/All Data/30images22/defecttopalignment/47/47_153007_1_' + str(image_number) + '.png'
        
        filename='D:/Uni/Vehcom/All Data/Rule-based data for thesis/Canny vertical/ROI_40/82/82_100719_1_' + str(image_number) + '.png'
       
        ground_truth_array = [0,16,17,18,19,20,21]
        print(filename)
        image=cv2.imread(filename)
        result = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        #plt.imshow(image)
        #plt.show()
        lower = np.array([120,255,255])
        upper = np.array([120,255,255])
        mask = cv2.inRange(image, lower, upper)
        result = cv2.bitwise_and(result, result, mask=mask)

        cv2.imshow('mask', mask)
        #cv2.imshow('result', result)
        cv2.waitKey(0)

        contours, hierarchy = cv2.findContours(image=mask , mode = cv2.RETR_TREE,method = cv2.CHAIN_APPROX_NONE)[-2:]

        # create an empty mask

        mask = np.zeros(mask.shape[:2],dtype=np.uint8)
        #print(contours)
        # loop through the contours

        for i,cnt in enumerate(contours):
            print("area of contours is",len(cnt))    
            # if the contour has no other contours inside of it
            if hierarchy[0][i][2] == -1 :
                    # if the size of the contour is greater than a threshold
                    if  cv2.contourArea(cnt) > 30:
                            cv2.drawContours(mask,[cnt], 0, (255), -1)
                            print('greater')    
                            #plt.imshow(mask,cmap='binary')
                            #plt.show()
        # display result
        #cv2.imshow("Mask", mask)
        print(mask)
        if [255] in mask :
            if image_number in ground_truth_array:
                #true_positives = true_positives + 1
                true_positives.append(image_number)
                print('true_positives loop',true_positives)
            else:
                #false_positives = false_positives +1     
                false_positives.append(image_number)
            
        else:
            print('non-defective')    
        plt.imshow(mask,cmap='gray')
        plt.title('image no.'+ str(image_number))
        #plt.show()

        plt.imshow(mask,cmap='binary')
        #plt.show()
        #, mode, method, mode, method, mode, methodcv2.imshow("Img", mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


        
print('true_positives',len(true_positives))
print('false_positives',len(false_positives))
false_negatives = len(ground_truth_array) - len(true_positives)
print('false_negatives',false_negatives)
#consecutive_images = 
def find_consecutive(array, num_total_images, min_num_consecutive = 2):
        
        num_consecutive = 0

        if num_total_images == 22:
            if (0 in array) & (21 in array) :
                array.append(-1)
            
        elif num_total_images == 26:
            if (0 in array) & (25 in array) :
                array.append(-1)
            
        
        
        
        
        array = sorted(array)
        

        for k, g in groupby(enumerate(array), lambda ix : ix[0] - ix[1]):
            #print('len list',len(list(map(itemgetter(1), g))))
            if (len(list(map(itemgetter(1), g))) >= min_num_consecutive):
                num_consecutive = num_consecutive + 1


        return num_consecutive,array
find_consecutive = find_consecutive(true_positives,22,2)        
print(find_consecutive)
#true_negatives = 22- len(ground_truth_array) 
        #img= img[240:308,1250:1725] #260qq
        #img= img[0:70,0:470]

        #plt.imshow(img,cmap='gray')
        
        #plt.show()
        #index=image_number # just to change the variable name because it was used before as image_number in processdefectsaving.py for the output

        #filename='D:/Uni/Vehcom/All Data/30images22/default/47/47_153007_1_' + str(index) + '.png'
        #filename='D:/Uni/Vehcom/All Data/Rule-based data for thesis/Canny vertical/ROI_50/82/82_100719_1_' + str(index) + '.png'
        
        #print('file to save',filename)
        #cv2.imwrite(filename,img)
        #print('file saved')


