from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import random
import os
import cv2
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import  RandomForestClassifier
from sklearn import metrics
from skimage import feature
from skimage import exposure
from numba import guvectorize
from nms import non_max_suppression_slow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib; matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import mahotas
import mahotas.features

images_names_list =[]
detections_list=[]
ground_truth_list= []
precision_list=[]
recall_list=[]
TP_list = [] 
TN_list = []
FP_list = [] 
FN_list = []


dataDir='D:/Uni/Vehcom/All Data/Machine Learning/Labeling/classifiers evaluation/' # path to coco file and images folder (NickFailure-76)
dataType='lines-defects'
annFile='{}vehcon76-{}.json'.format(dataDir,dataType)

coco=COCO(annFile)

# Load the categories in a variable
catIDs = coco.getCatIds()
cats = coco.loadCats(catIDs)

print(cats)
print(catIDs)

def getClassName(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id']==classID:
            return cats[i]['name']
    return "None"

print('The class name is', getClassName(1, cats)) # just to print name of categories

# Define the classes defect and black lines
filterClasses = ['defect', 'Black Lines']

# Fetch class IDs only corresponding to the filterClasses
catIds = coco.getCatIds(catNms=filterClasses) 
# Get all images containing the above Category IDs
imgIds = coco.getImgIds()#catIds=catIds # will edit it later to add all imagesa and true negatives. remove catIds=catIds
print("Number of images containing all the  classes:", len(imgIds))

# load and display a random image
print('number of image ids is', len(imgIds))

#  make a for loop her to run through all images, also add the true negatives images to the json file 
#print(imgIds[np.random.randint(0,len(imgIds))])
#img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0] # Next step is to load images in sequence # put for loop instead of random............. to go through all the images

#trial_images = [2110,2111,2112,2113,2114]
#test_images = list(range(1959,1970))
#trial_images = list(range(1959,1970)) + list(range(1996,2004))
#trial_images = list(range(1996,2004))
#trial_images = [2000]

test_images = list(range(1959,1970)) + list(range(1996,2004)) +list(range(2013,2022)) +list(range(2022,2030))+list(range(2030,2038)) +list(range(2038,2050))+ list(range(2050,2058))\
        +list(range(2058,2073))+ list(range(2073,2082))+list(range(2082,2096))+list(range(2096,2106)) + list(range(2106,2115)) +list(range(2127,2137))+list(range(2145,2154))\
                +list(range(2190,2198)) + list(range(2198,2209))+list(range(2209,2216))+list(range(2216,2225))+list(range(2225,2231))+list(range(2231,2240))\
                +list(range(2240,2248)) +list(range(2248,2266)) +list(range(2289,2303))+list(range(2331,2346))+list(range(2367,2376))+ list(range(2376,2402))+list(range(2434,2449))\
                +list(range(2460,2475))+list(range(2493,2508))

'''

test_images=  list(range(2190,2198)) + list(range(2198,2209))+list(range(2209,2216))+list(range(2216,2225))+list(range(2225,2231))+list(range(2231,2240))\
                +list(range(2240,2248)) +list(range(2248,2266)) +list(range(2289,2303))+list(range(2331,2346))+list(range(2367,2376))+ list(range(2376,2402))+list(range(2434,2449))\
                +list(range(2460,2475))+list(range(2493,2508)) 
'''
print('list of images is:', test_images)
print('Number of images in list:',len(test_images))

for test in test_images:
    print(test)
    img = coco.loadImgs(imgIds[test])[0]
    print('img is ', img)
    I=cv2.imread('{}/NickFailure-76/{}'.format(dataDir,img['file_name']))
    print('shape of Image I is ',I.shape)
    print('file name is ',img['file_name'])
    images_names_list.append(img['file_name'])
    print (images_names_list)
    print(len(images_names_list))
    
    #comment plots for now
    #plt.axis('off')
    #plt.title('first')
    #plt.imshow(I,cmap='gray')
    #plt.show()
    
    gear_full= I

    try:


        edges= cv2.Canny(gear_full,10,200,apertureSize = 3) # 30,200 is good

        #plt.imshow(edges,cmap='gray') # important
        #plt.show()
        edges_color = cv2.cvtColor(edges,cv2.COLOR_GRAY2BGR)
        circles = cv2.HoughCircles(edges,cv2.HOUGH_GRADIENT,1,30,
                                    param1=9,param2=9,minRadius=580,maxRadius=590) #1,30,9,9,579,590(last thing used) ,  #1,10,40,30,550-575 what i am using# 520-555 is good #tried 545=580 #550-570

        circles = np.uint16(np.around(circles))

        for i in circles[0,:]:
            # draw the outer circle
            cv2.circle(edges_color,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(edges_color,(i[0],i[1]),2,(0,0,255),3)
        #comment plots for now    
        #plt.imshow(edges_color,cmap='gray') # important
        #plt.show()

        all_circles = circles[0,:]
        print('number of circles',len(all_circles))
        print('all_circles',all_circles)

        for value in range(len(all_circles)) :
         #print (all_circles[y_value][1])
         if (50 <= all_circles[value][1] <= 78) & (1450<=all_circles[value][0] <= 1520):
            print(all_circles[value][1])
            center_y =all_circles[value][1]
            print(all_circles[value][0])
            center_x =all_circles[value][0]

        print('center is ',center_x,center_y)

    except (TypeError,NameError) :

        print('Type error here')
        center_x,center_y = 1460 , 65
        #exit()
        #pass

    #except NameError: 
    #    print('Name error here')
    #    center_x,center_y = 1472  , 78
        #exit()
    #    pass

    print('center is ',center_x,center_y)


    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    print('Ids are',annIds)
    anns = coco.loadAnns(annIds)
    print ('details area',anns[0]['segmentation'])
    coco.showAnns(anns)


    filterClasses = ['defect', 'Black Lines']
    mask = np.zeros((img['height'],img['width']))
    print('yuppp')
    #comment plots for now
    #plt.title('second')
    #plt.imshow(mask)
    #plt.show()
    
    print(img['height'])
    print(len(anns))

    ground_truth= 0 # assuming that ground truth is 0 (true negative) at the beginning
    boxes_defects_list = np.empty((0,4), int)
    boxes_blacklines_list=  np.empty((0,4), int)
    for i in range(len(anns)):
        className = getClassName(anns[i]['category_id'], cats)
        print('class name is ',className)
        if className == 'defect': # check whether there is a defect in this image or not 
            print('defeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeect')

            segmentation = anns[i]['segmentation'][0] #use this also with black lines while applying the sliding window
            print ('segmentation is ',segmentation) # segmentation of defects has 8 elements in the list.

            #boxes_defects = anns[i]['bbox'] #use this also with black lines while applying the sliding window
            #boxes_defects_list= np.vstack((boxes_defects_list,boxes_defects))
            #print ('bbox is ',boxes_defects_list) # segmentation of defects has 8 elements in the list.

            print('len segmentation',len(segmentation))
            indexes = [0,1,2,3,4,5,6,7]
            odd= indexes[1::2]  #to get y-axis values of the segmentation
            print('odd are',odd)
            even=indexes[0::2] # to get x-axis values of the segmentation
            print('even are',even)
            #gear = gear[160+center_y:150+center_y+115,center_x-240:center_x+300] # just as indecation to the coordinates.
            for x_value in (even):
                # x axis was 1200-1800, y axis was 230-330
                if (center_x-240<= segmentation[x_value] <= center_x+300)  : #maybe change that to get the center of the defect to the x and y axis.#check whether x-axis values area within the area of interest or not
                # change these numbers with coordinates from hough circle automatic detection of area of interest
                    print('x_value is inside')
                    x_inside =True
                else:
                    x_inside= False    
            for y_value in (odd):        
                if (160+center_y<= segmentation[y_value] <= 150+center_y+115) :  #check whether y-axis values area within the area of interest or not
                # change these numbers with coordinates from hough circle automatic detection of area of interest
                    print('y_value is inside')
                    y_inside =True
                else:
                    y_inside=False    

            if (x_inside==True) & (y_inside == True) :
                print('insideeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee')  
                ground_truth= ground_truth+1 # Count the number of defects in the area of interest
                boxes_defects = anns[i]['bbox']  # also tried it with segmentation #use this also with black lines while applying the sliding window
                print('boxxxxxxxx is ',boxes_defects)
                print ('bbox is ',boxes_defects_list) # segmentation of defects has 8 elements in the list.
                boxes_defects = [anns[i]['bbox'][0],anns[i]['bbox'][1],anns[i]['bbox'][0]+anns[i]['bbox'][2],anns[i]['bbox'][1]+anns[i]['bbox'][3]]
                print ('bbox is ',boxes_defects) # segmentation of defects has 8 elements in the list.
                boxes_defects_list= np.vstack((boxes_defects_list,boxes_defects))
                print ('defects boxes are',boxes_defects_list) # segmentation of defects has 8 elements in the list.
                print('ground_truth is', ground_truth)
            else:
                print('outsideeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee')      
                #print('not in ROI')    
        elif className == 'Black Lines':
            ground_truth= ground_truth # if none defects 


            segmentation_black = anns[i]['segmentation'][0] #use this also with black lines while applying the sliding window
            print ('segmentation of black lines is  ',segmentation_black) # segmentation of defects has 8 elements in the list.

            #boxes_blacklines = anns[i]['bbox'] #use this also with black lines while applying the sliding window 
            boxes_blacklines = [anns[i]['bbox'][0],anns[i]['bbox'][1],anns[i]['bbox'][0]+anns[i]['bbox'][2],anns[i]['bbox'][1]+anns[i]['bbox'][3]]
            boxes_blacklines_list= np.vstack((boxes_blacklines_list,boxes_blacklines))
            print ('black_lines boxes are',boxes_blacklines_list) 
            print('Number of black boxer is',len(boxes_blacklines_list))

            print('ground truth is ',ground_truth)


        pixel_value = filterClasses.index(className)+1
        print('pixel_value is',pixel_value)
        mask = np.maximum(coco.annToMask(anns[i])*pixel_value, mask)


    print('shape of mask is',mask.shape)
    
    #comment plots for now
    #plt.title('thats it')
    #plt.imshow(mask,cmap='gray')
    #plt.show()

    #print('shape iversed',gear_full.shape.T)


    #gear_full= cv2.cvtColor(gear_full,cv2.COLOR_RGB2GRAY)
    print('gear full shape',gear_full.shape)
    #gear_thresh = cv2.threshold(gear_full,40,255, cv2.THRESH_BINARY)
    #plt.title('thresholding')
    #plt.imshow(gear_thresh)
    #plt.show()

    reto,mask_thresh = cv2.threshold(mask,1,2,cv2.THRESH_BINARY)
    #mask_thresh = np.minimum(coco.annToMask(anns[i])*pixel_value, mask)
    #mask_thresh = cv2.bitwise_and(gear_full,mask=mask)
    #mask_thresh= cv2.subtract(gear_full,mask)
    #print('mask thresh shape',mask_thresh.shape)

    #comment plots for now
    #plt.title('adding')
    #plt.imshow(mask_thresh,cmap='gray')
    #plt.show()

    mask_ROI = mask[160+center_y:150+center_y+115,center_x-240:center_x+300] # can edit this number later
    
    #comment plots for now
    #plt.title('Ground truth is: {}'.format(ground_truth))
    #plt.imshow(mask_ROI)
    #plt.show()
    print(ground_truth)
    gear= gear_full [160+center_y:150+center_y+115,center_x-240:center_x+300]
    #gear[segmentation_black]=[255,0,0]
    gear_mask_thresh = mask_thresh [160+center_y:150+center_y+115,center_x-240:center_x+300]
    
    #comment plots for now
    #plt.imshow(gear_mask_thresh,cmap='gray')
    #plt.show()

    print(gear_mask_thresh)
    try: 
        segmentation_black = [ int(x) for x in segmentation_black ]
        print('segmentation integers',segmentation_black)
    except NameError :
        pass
    #odd= indexes[1::2]  #to get y-axis values of the segmentation
    #print('odd are',odd)
    #even=indexes[0::2] # to get x-axis values of the segmentation
    #print('shapeeeeeeeeeeeeeeeeeeee',gear_full.shape)
    #gear_full=cv2.fillPoly(gear_full, pts =segmentation_black, color=(255,255,255))
    #gear_full[segmentation_black] =[255,0,0]
    gear_copy= gear.copy()

    #comment plots for now
    #plt.imshow(gear,cmap='gray')
    #plt.show()
    
    #plt.imshow(gear_full,cmap='gray')
    #plt.show()

    height= gear.shape[0]
    width = gear.shape[1]
    #print(height)
    #print(width)
    width_patches= np.arange(0,width,50) # 50 
    height_patches= np.arange(0,height,50) # 50


    # can edit patches later to get the last patch image as width or hight - 1500 or 2000 for example
    # so it will be 36 and 48 
     
    #filename = 'D:/Uni/Vehcom/All Data/Machine Learning/Labeling/hog_1_4x4.sav'
    #filename = 'D:/Uni/Vehcom/All Data/Machine Learning/Labeling/hog_middle_8x8,3x3.sav' # hog_middle_2x2
    #filename = 'D:/Uni/Vehcom/All Data/Machine Learning/Labeling/hog_middle_4x4,2x2.sav' # hog_middle_2x2
    #filename = 'D:/Uni/Vehcom/All Data/Machine Learning/Labeling/hog_middle_8x8,3x3_1.sav' # hog_middle_2x2

    #filename = 'D:/Uni/Vehcom/All Data/Machine Learning/Labeling/hog_middle_8x8,3x3_2_svm.sav' # impotant

    #filename = 'D:/Uni/Vehcom/All Data/Machine Learning/Labeling/hog_middle_8x8,3x3_2_RF.sav' # impotant
    #filename = 'D:/Uni/Vehcom/All Data/Machine Learning/Labeling/all_dataset_histogram_21_2_RF.sav'
    #filename = 'D:/Uni/Vehcom/All Data/Machine Learning/Labeling/all_dataset_histogram_21.sav'
    filename = 'D:/Uni/Vehcom/All Data/Machine Learning/Labeling/all_dataset_combined_14.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    #result = loaded_model.score(x_test, y_test)
    #print(result)

    false_positives_list =[]

    #@jit(objectmode=True) 
    #@jit(target ="gpu")
    #@cuda.jit
    #@jit
    #@cuda.jit(device=True)
    #@guvectorize


    def combinedclassification():
        image_coordinates = np.empty((0,4), int) # 2x2
        image_positives = np.empty((0,4), int)
        boundingBoxes= np.empty((0,4), int)
        boundingboxes_full_image = np.empty((0,4), int)
        for j in height_patches:
            #print('j',j)

            for i in width_patches:
                print('i',i)
                unlabeled = gear [ j : j +50  , i : i+50 ] # 30x30 window
                features_unlabeled_haralick= mahotas.features.haralick(unlabeled).mean(0)
                features_unlabeled_histogram = cv2.calcHist([unlabeled],[0],None,[256],[0,256])
                unlabeled = cv2.cvtColor(unlabeled,cv2.COLOR_BGR2GRAY)
                #print(list(features_unlabeled))
                #print('haralick is',list(features_unlabeled_haralick))
                y = [int(yes) for yes in features_unlabeled_histogram]
                #print('histogram i s',[y])
                all_features = np.append(list(features_unlabeled_haralick),[y])
                #print(len(all_features))
                #print(all_features)

                prediction = loaded_model.predict([all_features])
                print(prediction)
                false_positives_list.append(int(prediction))
                if prediction == 1 :

                    #plt.imshow(unlabeled,cmap='gray')
                    #plt.show()
                    inspect =cv2.rectangle(gear_copy, (i,j),(i+50,j+50), (0,255,0), 2) # 30x30 window
                    #plt.imshow(inspect)
                    #plt.show()
                    image_coordinates = [i , j , i+50,j +50] # start x ,start y , end x, end y
                    #image_coordinates_full_image = 160+center_y:150+center_y+115,center_x-240:center_x+300
                    #image_coordinates_full_image = [i+ center_x-240,j+160+center_y,i+50+center_x+300,j+50+center_y+115]
                    image_coordinates_full_image = [ i+ center_x-240,j+160+center_y,i+50+center_x-240,j+50+160+center_y] # commented this for now but important
                    #image_coordinates_full_image = [i+50+center_x-240, j+160+center_y,i+50+center_x-240,j+50+160+center_y, i+ center_x-240,j+50+160+center_y, i+ center_x-240,j+160+center_y]
                    boundingBoxes= np.vstack((boundingBoxes,image_coordinates))
                    boundingboxes_full_image = np.vstack((boundingboxes_full_image,image_coordinates_full_image))
                    print('image coordinates are ',image_coordinates)

        print('bounding boxes are ',boundingBoxes)
        print('number of boxes before non maximum',len(boundingBoxes))
        for (startX, startY, endX, endY) in boundingBoxes:
            cv2.rectangle(gear_copy, (startX, startY), (endX, endY), (0, 0, 255), 2)
        # perform non-maximum suppression on the bounding boxes
        pick = non_max_suppression_slow(boundingBoxes, 0.2) # tune this number
        pick_full_image = non_max_suppression_slow(boundingboxes_full_image, 0.2) # tune this number
         
        print('bounding boxes are',pick)
        print ("[x] after applying non-maximum, %d bounding boxes"% (len(pick)))
        # loop over the picked bounding boxes and draw them
        for (startX, startY, endX, endY) in pick:
            cv2.rectangle(gear, (startX, startY), (endX, endY), (0, 255, 0), 2) # put it gear_copy if I need the new bounding bozes on the same image. Put it gear, if I want the boxes on new image.

        # display the images
        
        #comment plots for now
        #plt.imshow(gear,cmap='gray')
        #plt.show()




        #return false_positives_list
        print(false_positives_list)
        print('number of elements',len(false_positives_list))
        print('number of ones',false_positives_list.count(1))
        print('number of zeros',false_positives_list.count(0))
        #plt.imshow(inspect)
        #plt.show()      
        #np.asarray(false_positives_list)
        #print(np.count_nonzero(false_positives_list==1))   
        return (pick_full_image,len(pick_full_image))


    unlabeled_combined = combinedclassification()


    def bb_intersection_over_union(boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
        
        # return the intersection over union value
        return iou

    #len_pick = unlabeledhog()[0]
    #print(len(pick))
    print('defect boxes',boxes_defects_list)
    print('black lines boxes',boxes_blacklines_list)
    boxes_defects_classifier_list = unlabeled_combined[0] # or unlabeled_histogram in case of histogram only
    print('classifier defects boxes',boxes_defects_classifier_list)
    #boxes_labels = 
    print(int(unlabeled_combined[1])) #unlabeled_histogram in case of histogram only
    detections = int(unlabeled_combined[1]) #unlabeled_histogram in case of histogram only


    # but braek after these conditions
    if ground_truth==0:
        if detections==0:
            print('True Negative')
        elif detections >=1 :
            false_positives = detections
            print('Number of False Positives is: {}'.format(detections))

    if (ground_truth >=1) & detections==0:
        print('Number of False Negatives is: {}'.format(ground_truth)) 


    image = cv2.imread('{}/NickFailure-76/{}'.format(dataDir,img['file_name']))
    image_copy = image.copy()

    # compute the intersection over union and display it
    print('len boxes_defects_list',len(boxes_defects_list))
    print ('black_lines boxes are',len(boxes_blacklines_list)) 
    all_detections = len(boxes_defects_classifier_list)
    print('len boxes_defects_classifier_list',len(boxes_defects_classifier_list))
    detections_list.append(len(boxes_defects_classifier_list))
    True_defects= 0
    True_black = 0
    false_positives=0
    True_defects_intersected = 0
    #for pr_defect in range(len(boxes_defects_classifier_list)): # can swith it with gt_defect
    for gt_defect in range(len(boxes_defects_list)): #can swith it with pr_defect
        #for gt_defect in range(len(boxes_defects_list)): #can swith it with pr_defect
        intersected_defects = []
        for pr_defect in range(len(boxes_defects_classifier_list)): # can swith it with gt_defect
            # Ground truth rectangle (startX, startY), (endX, endY)
            ground_truth_center = [int(boxes_defects_list[gt_defect][1]+(boxes_defects_list[gt_defect][3] - boxes_defects_list[gt_defect][1]) / 2 ),int(boxes_defects_list[gt_defect][0]+(boxes_defects_list[gt_defect][2]- boxes_defects_list[gt_defect][0]) / 2)]
       
            iou = bb_intersection_over_union(boxes_defects_classifier_list[pr_defect],boxes_defects_list[gt_defect])
            if iou >= .1 : # tune this number
                #True_defects =True_defects+1 #comented for now

                cv2.rectangle(image_copy, (int(ground_truth_center[1]-25),int(ground_truth_center[0]-25)),(int(ground_truth_center[1]+25),int(ground_truth_center[0]+25)), (0, 255, 0), 2) # good
                cv2.rectangle(image_copy, (boxes_defects_classifier_list[pr_defect][0],boxes_defects_classifier_list[pr_defect][1]),(boxes_defects_classifier_list[pr_defect][2],boxes_defects_classifier_list[pr_defect][3]), (0, 0, 255), 2) #good
            # show the output images
                ROI = image_copy[160+center_y:150+center_y+115,center_x-240:center_x+300]
                cv2.putText(ROI, "IoU: {:.4f}".format(iou), (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                print(": {:.4f}".format(iou))
                
                #comment plots for now
                #plt.imshow(ROI,cmap='gray')
                #plt.show()

                #if True_defects >1 :
                #    True_defects =1 
                intersected_defects.append(boxes_defects_classifier_list[pr_defect])
                print('intersected_defects is ',intersected_defects)
                print('lenghthhhhhhhhhhhh',len(intersected_defects))
            #else:
             #   false_positives = false_positives+1    
        if len(intersected_defects) >1 :
            True_defects_intersected = True_defects_intersected +1  
            print('intersecteeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeed',True_defects_intersected)
            all_detections = len(boxes_defects_classifier_list) -1 
        elif len(intersected_defects) == 1:
            True_defects = True_defects+1
            print('not intersecteeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeed',True_defects)    

    print('really true defects are',True_defects_intersected+True_defects)    
        #for idx,intersect in enumerate(intersected_defects) :
        #    print('index is',idx)
        #    print('intersect is ',intersect)
        #    print('intersect of index is ',intersect[idx])
        #    print('index zero is', intersected_defects[idx])
        #    print('index one is', intersected_defects[idx+1])
        #    if bb_intersection_over_union(intersected_defects[idx],intersected_defects[idx+1]) >=0.05:
        #        True_defects = True_defects -1 
        #        print('true defeeeeeeeeeeeeeeeeeects intersection',True_defects)

    for pr_defect in range(len(boxes_defects_classifier_list)): # can swith it with gt_defect
        
        for gt_black in range(len(boxes_blacklines_list)): #can swith it with pr_defect
            # Ground truth rectangle (startX, startY), (endX, endY)
            ground_truth_center = [int(boxes_blacklines_list[gt_black][1]+(boxes_blacklines_list[gt_black][3] - boxes_blacklines_list[gt_black][1]) / 2 ),int(boxes_blacklines_list[gt_black][0]+(boxes_blacklines_list[gt_black][2]- boxes_blacklines_list[gt_black][0]) / 2)]
            #print('ground truth center is ',ground_truth_center)
            
            iou_black= bb_intersection_over_union(boxes_defects_classifier_list[pr_defect],boxes_blacklines_list[gt_black])
            print('calculated iou_balck')
            #[ int(x) for x in segmentation_black ]
            #int(z) for z in boxes_blacklines_list[gt_black]
            print('coordinateeeeeeeeeeeeeeeeeeeeeees',boxes_defects_classifier_list[pr_defect])
            #black_thresh = mask_thresh[[int(z) for z in boxes_defects_classifier_list[pr_defect]]]
            black_thresh = mask_thresh[boxes_defects_classifier_list[pr_defect][1]:boxes_defects_classifier_list[pr_defect][3],boxes_defects_classifier_list[pr_defect][0]:boxes_defects_classifier_list[pr_defect][2]]
            print('yoooooooooooooooooooooooooooooooooooooooooooooooooooo',black_thresh)

            #comment plots for now
            #plt.imshow(black_thresh,cmap='gray')
            #plt.show()
            
            white_count = np.count_nonzero(black_thresh)
            print('white count pixels number is ',white_count)
            all_pixels_count= black_thresh.size 
            print('image size is and count',all_pixels_count)
            black_lines_percentage = white_count / all_pixels_count 
            print('percentage of white pixels',black_lines_percentage)
            if black_lines_percentage >0.05:
                True_black = True_black +1 

            
            #if iou_black>= .03 : # tune this number
             #   True_black =True_black+1 

                #cv2.rectangle(image_copy, (int(ground_truth_center[1]-25),int(ground_truth_center[0]-25)),(int(ground_truth_center[1]+25),int(ground_truth_center[0]+25)), (0, 255, 0), 2) # good
              #  cv2.rectangle(image_copy, (int(boxes_blacklines_list[gt_black][0]),int(boxes_blacklines_list[gt_black][1])),(int(boxes_blacklines_list[gt_black][2]),int(boxes_blacklines_list[gt_black][3])), (0, 255,0 ), 2) #good    
               # cv2.rectangle(image_copy, (boxes_defects_classifier_list[pr_defect][0],boxes_defects_classifier_list[pr_defect][1]),(boxes_defects_classifier_list[pr_defect][2],boxes_defects_classifier_list[pr_defect][3]), (0, 0, 255), 2) #good
                # show the output images
               # ROI = image_copy[160+center_y:150+center_y+115,center_x-240:center_x+300]
               # cv2.putText(ROI, "IoU: {:.4f}".format(iou_black), (10,30),
               # cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
               # print(": {:.4f}".format(iou_black))
               # plt.imshow(ROI,cmap='gray')
               # plt.show()
            

    print('True_black',True_black)
    if (True_defects <= len(boxes_defects_list)) or (True_defects >= len(boxes_defects_list)):
        print('enterd last if')
        #True_positives =True_defects
        True_positives = True_defects_intersected+True_defects
        #false_negatives = len(boxes_defects_list) - True_defects
        false_negatives = len(boxes_defects_list) - True_positives
        false_positives_black = True_black
        #false_positives= len(boxes_defects_classifier_list) - True_defects - True_black 
        #false_positives= all_detections - True_black - True_defects #len(boxes_defects_classifier_list) instead of all_detections#len(intersected_defects)
        false_positives= all_detections - True_black - True_positives
        if (false_positives==0)& (True_positives==0) & (false_negatives==0):
            True_negatives= 1
        else :  
            True_negatives= 0    

    try:  
        precision = True_positives/ (True_positives+false_positives)
    except ZeroDivisionError :
        precision = 'Div/0'

    try:
        recall = True_positives / (True_positives+false_negatives)
    except ZeroDivisionError:

        recall = 'Div/0'
    print('Ground truth is:{}'.format(ground_truth))
    ground_truth_list.append(ground_truth)
    print('Number of True positives is: {}'.format(True_positives))
    print('Number of False positives before black is: {}'.format(false_positives_black))
    print('Number of False positives is: {}'.format(false_positives))
    print('Number of False negatives is: {}'.format(false_negatives))
    print('Number of True negatives is: {}'.format(True_negatives))        # it was 0 instead if True_negatives before.
    print('Precision is: {}'.format(precision))
    print('Recall is: {}'.format(recall))
       


    TP_list.append(True_positives)
    print('TP_list is ',TP_list)
    TN_list.append(True_negatives)
    FP_list.append(false_positives)
    FN_list.append(false_negatives)
    precision_list.append(precision)
    recall_list.append(recall)

    metrics_dictionary = {'Img no.':images_names_list,'Detections':detections_list,'Ground_truth':ground_truth_list,
                            'True positives':TP_list,'True negatives':TN_list,'False positives':FP_list,'False negatives':FN_list,'precision':precision_list,'recall':recall_list}


df= pd.DataFrame(metrics_dictionary) # i think will change index [0] with index [id or index of the images in the foor loop]
print(df)
df.to_csv('D:/Uni/Vehcom/All Data/Machine Learning/Labeling/test_random_forest/models_analysis_automated.csv') 









































