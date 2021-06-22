import cv2
import numpy as np
from matplotlib import pyplot as plt
from glob import glob 
import os

# Reading default gear images with no changes. 
# Read first image of the gear to find the point of shift.
# Doing the same process for 30 gears, one after the other.

folders = glob ('D:/Uni/Vehcom/All Data/Rule-based data for thesis/Test set gears/*')


for fold in folders:
    print('fold',fold)
    directory = (fold+"/*")    
    print('directory ',directory)
    images_directory = glob (directory)
    print('images_directory',images_directory)
    
    img=cv2.imread(images_directory[0],0)

    edges= cv2.Canny(img,30,200,apertureSize = 3)

    #searching within this area
    edges= edges[630:670,1420:1550] #645:670
    for row_top_canny in range(0,40):
        if edges[row_top_canny,50]==255: # 50 is a constant x-axis value, algorithm searches for the the y-value of the white pixel.
            print('perfect top row for canny is',row_top_canny)
            center_row_top_canny = row_top_canny # the y-axis value is calculated now.


    #plt.imshow(edges,cmap='gray')
    #mng = plt.get_current_fig_manager()
    #mng.window.state('zoomed')
    #plt.show()


    all_rows_top_canny=[] # array with all y-axis values for 22 images of the gear for the white pixel.

    for index in range(len(images_directory)):
            print('index',index)
            image_name = os.path.basename(images_directory[index])       
            split_image_name = image_name.split('_')
            image_number =split_image_name[3]

            #images_names_list.append(split_image_name[0])
            print('image name',image_name)
            print('image number',image_number)

    #read all gear umages and apply the process one by one

            #filename = 'D:/Uni/Vehcom/All Data/Rule-based data for thesis/Test set gears/82/82_100719_1_' + str(index) + '.png'
            filename = images_directory[index]
           




            print('filename',filename)
            split = filename.split('/')[5].split('\\')[1] + '/' + filename.split('/')[5].split('\\')[2]
            print('split',split)
            img=cv2.imread(filename, 0)
            img_color= cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            edges= cv2.Canny(img,45,200,apertureSize = 3) #was 45,200,3
            edges= edges[630:670,1420:1550] #630:670 good #645:670 
            

            
            #plt.imshow(edges,cmap='gray')
            #plt.title('canny image no.'+ str(index))
            #plt.show()
            
           
            for row_top_canny_img in range(0,40):

              
                if  edges[row_top_canny_img,50]==255:
                    
                    print('this y-axis value for canny is',row_top_canny_img)
                    all_rows_top_canny.append(row_top_canny_img)
                    print('output of the array',all_rows_top_canny)

                    #previous_value = row_top_canny_img
                    #print('previous value ', previous_value)    
                    center_row_top_canny_img = row_top_canny_img  

                        
                
                    num_rows_top, num_cols_top = edges.shape[:2] 
                    print('edges top rows',num_rows_top)
                    print('edges top columns',num_cols_top)

                    #area_color_top= cv2.cvtColor(area_top, cv2.COLOR_GRAY2BGR)

                    #row_difference_top = row_top - all_rows_top[0]
                    row_difference_top_canny= center_row_top_canny_img - center_row_top_canny # get the diff. between the y-axis value of first image and the y value of next images one by one. 
                    print('row top',center_row_top_canny_img)
                    #print('all_rows_top[0]',all_rows_top[0])
                    print('important row_difference_top',row_difference_top_canny)

                    # Creating a translation matrix
                    translation_matrix_top_canny= np.float32([ [1,0,0], [0,1,-row_difference_top_canny] ])

                    img_translation_top_canny= cv2.warpAffine(edges, translation_matrix_top_canny, (num_cols_top,num_rows_top))
                                          


                    num_rows_img_top_canny, num_cols_img_top_canny= img_color.shape[:2]


                    print('all image canny_top rows',num_rows_img_top_canny)
                    print('all image canny columns',num_cols_img_top_canny)

                   
                    translation_matrix_img_top_canny= np.float32([ [1,0,0], [0,1,-row_difference_top_canny ]])

                    
                    img_translation_img_top_canny= cv2.warpAffine(img_color[0:805,0:2048], translation_matrix_img_top_canny, (num_cols_img_top_canny,num_rows_img_top_canny))
                   

                      




                    

                    #filetosave_top_canny='D:/Uni/Vehcom/All Data/Rule-based data for thesis/Canny vertical/Alignment/82/82_100719_1_' + str(index) + '.png'
                    filetosave_top_canny='D:/Uni/Vehcom/All Data/Rule-based data for thesis/Canny vertical/Alignment/' +str(split)

                    






                    print('file to save',filetosave_top_canny)
                    cv2.imwrite(filetosave_top_canny,img_translation_img_top_canny[0:805,0:2048])
                    print('file saved')
                    break
                    


    print('final',all_rows_top_canny)





