import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import *
from itertools import groupby
from operator import itemgetter
from glob import glob
import os

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

all_true_positives = 0
all_false_positives = 0
all_true_negatives = 0 
all_false_negatives = 0
all_consecutive = 0
defective_gears = 0
ground_truth_array = []
gear_40_labels=[14,15,16,17,18,19,20]
ground_truth_array.append(gear_40_labels)
gear_44_labels=[15,16,17,18]
ground_truth_array.append(gear_44_labels)
gear_46_labels=[1,2,3,21]
ground_truth_array.append(gear_46_labels)
gear_47_labels=[17,18,19]
ground_truth_array.append(gear_47_labels)
gear_49_labels=[13,14,15]
ground_truth_array.append(gear_49_labels)
gear_50_labels=[16,17,18,19,20,21]
ground_truth_array.append(gear_50_labels)
gear_51_labels=[1,2,3,4]
ground_truth_array.append(gear_51_labels)
gear_52_labels=[3,4,5,6,7,8,9]
ground_truth_array.append(gear_52_labels)
gear_53_labels=[17,18,19,20]
ground_truth_array.append(gear_53_labels)
gear_55_labels=[1,2,3,18,19,20]
ground_truth_array.append(gear_55_labels)
gear_56_labels=[15,16,17,18]
ground_truth_array.append(gear_56_labels)
gear_57_labels=[16,17,18,19]
ground_truth_array.append(gear_57_labels)
gear_60_labels=[0,1,2,3,4]
ground_truth_array.append(gear_60_labels)
gear_62_labels=[16,17,18,19]
ground_truth_array.append(gear_62_labels)
gear_68_labels=[19,20,21]
ground_truth_array.append(gear_68_labels)
gear_73_labels=[18,19,20]
ground_truth_array.append(gear_73_labels)
gear_75_labels=[19,20,21]
ground_truth_array.append(gear_75_labels)
gear_77_labels=[13,14,15,16]
ground_truth_array.append(gear_77_labels)
gear_78_labels=[17,18,19]
ground_truth_array.append(gear_78_labels)
gear_79_labels=[0,1,21]
ground_truth_array.append(gear_79_labels)
gear_80_labels=[18,19,20]
ground_truth_array.append(gear_80_labels)
gear_81_labels=[1,2,3]
ground_truth_array.append(gear_81_labels)
gear_82_labels=[0,16,17,18,19,20,21]
ground_truth_array.append(gear_82_labels)
gear_85_labels=[0,1,2,3,4,5,24,25]
ground_truth_array.append(gear_85_labels)
gear_88_labels=[9,10,11,12,13,14,15]
ground_truth_array.append(gear_88_labels)
gear_91_labels=[0,1,2]
ground_truth_array.append(gear_91_labels)
gear_92_labels=[3,4,5,6,7,8,9,10,11,12,13,14]
ground_truth_array.append(gear_92_labels)
gear_95_labels=[7,8,9,10,11,12,13,14]
ground_truth_array.append(gear_95_labels)
gear_97_labels=[21,22,23,24,25]
ground_truth_array.append(gear_97_labels)
gear_99_labels=[0,1,2,3,4,5,23,24,25]
ground_truth_array.append(gear_99_labels)
#print('ground_truth_array',ground_truth_array)
#print('first item of gt array',ground_truth_array[0])
#print('last item of gt array',ground_truth_array[29])

#folders = glob ('D:/Uni/Vehcom/All Data/Rule-based data for thesis/Canny vertical/ROI_30/*')
folders = glob ('D:/Uni/Vehcom/All Data/Rule-based data for thesis/without_alignment/ROI_30/*')
for fold_number, fold in enumerate(folders):
    #print('fold number',fold_number)
    print('fold',fold)
    directory = (fold+"/*")    
    #print('directory ',directory)
    images_directory = glob (directory)
    #print('images_directory',images_directory)
    true_positives = []
    false_positives = []
    ground_truth_array_gear = ground_truth_array[fold_number]
    #print('ground_truth_array_gear',ground_truth_array_gear)
    for index in range(len(images_directory)):
            #print('index',index)
            image_name = os.path.basename(images_directory[index])       
            split_image_name = image_name.split('_')
            image_number =split_image_name[3].split('.')[0]

            #images_names_list.append(split_image_name[0])
            #print('image name',image_name)
            #print('image number',int(image_number))

   
            filename = images_directory[index]
            #print('filename',filename)



            
            #filename='D:/Uni/Vehcom/All Data/Rule-based data for thesis/Canny vertical/ROI_60/82/82_100719_1_' + str(image_number) + '.png'
           
            
            #print(filename)
            image=cv2.imread(filename)
            result = image.copy()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            #plt.imshow(image)
            #plt.show()
            lower = np.array([120,255,255])
            upper = np.array([120,255,255])
            mask = cv2.inRange(image, lower, upper)
            result = cv2.bitwise_and(result, result, mask=mask)

            #cv2.imshow('mask', mask)
            #cv2.imshow('result', result)
            #cv2.waitKey(0)

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
                        if  cv2.contourArea(cnt) > 50:
                                cv2.drawContours(mask,[cnt], 0, (255), -1)
                                print('greater')    
                                #plt.imshow(mask,cmap='binary')
                                #plt.show()
            # display result
            #cv2.imshow("Mask", mask)
            #print(mask)
            #plt.imshow(img)
            #plt.show()
            #print('image shape',len(img))

            #print('third channel',img[)
            
            #print('black ='+black + 'red='+red) 
            #print('count is', count)
            if [255] in mask :
                #print('hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh')
                #print('imaaaaaaaaaaaaaaage number',int(image_number))
                #print('grrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrounndd',ground_truth_array_gear)
                if int(image_number) in ground_truth_array_gear:
                    #print('yes')
                    #true_positives = true_positives + 1
                    true_positives.append(int(image_number))
                    
                else:
                    #false_positives = false_positives +1     
                    false_positives.append(int(image_number))
                    #print('no')
                
                #print('count',img.count([0,0,255]))

            # else:
            #   print('no') 


            #plt.imshow(img,cmap='gray')
            #plt.title('image no.'+ str(image_number))
            #plt.show()
                
    print('true_positives',len(true_positives))
    if len(true_positives) >=1:
        defective_gears = defective_gears + 1
    print('false_positives',len(false_positives))
    false_negatives = len(ground_truth_array_gear) - len(true_positives)
    print('false_negatives',false_negatives)
    #print(true_positives)
    true_negatives = len(images_directory) - (len(true_positives) + len(false_positives) +false_negatives )
    print('true_negatives',true_negatives)
    all_true_positives = all_true_positives + (len(true_positives))
    all_false_positives = all_false_positives + (len(false_positives))
    all_true_negatives = all_true_negatives +   true_negatives
    all_false_negatives = all_false_negatives + false_negatives

    find_consecutive_inspection = find_consecutive(true_positives,len(images_directory),3)        
    print('consecutive',find_consecutive_inspection)
    if find_consecutive_inspection[0] >= 1:
        all_consecutive = all_consecutive + 1
all_ground_truth_defects = 0
for gt in ground_truth_array:
    all_ground_truth_defects = all_ground_truth_defects + len(gt)

print('ground_truth_labels',all_ground_truth_defects)
print('all_true_positives',all_true_positives)
print('all_false_positives',all_false_positives)
print('all_false_negatives',all_false_negatives)
print('all_true_negatives',all_true_negatives)
print('number of defective gears',defective_gears)
precision = (all_true_positives)/(all_true_positives+all_false_positives)*100
recall = (all_true_positives)/(all_true_positives+all_false_negatives)*100
print('Precision',precision)
print('Recall',recall)
print('all_consecutive',all_consecutive)

        #consecutive_images = 
# def find_consecutive(array, num_total_images, min_num_consecutive = 2):
        
#         num_consecutive = 0

#         if num_total_images == 22:
#             if (0 in array) & (21 in array) :
#                 array.append(-1)
            
#         elif num_total_images == 26:
#             if (0 in array) & (25 in array) :
#                 array.append(-1)
            
        
        
        
        
#         array = sorted(array)
        

#         for k, g in groupby(enumerate(array), lambda ix : ix[0] - ix[1]):
#             #print('len list',len(list(map(itemgetter(1), g))))
#             if (len(list(map(itemgetter(1), g))) >= min_num_consecutive):
#                 num_consecutive = num_consecutive + 1


#         return num_consecutive,array


# find_consecutive = find_consecutive(true_positives,22,2)        
# print('consecutive',find_consecutive)
#true_negatives = 22- len(ground_truth_array_gear) 
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


