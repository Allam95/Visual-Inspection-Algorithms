import cv2
import numpy as np
from matplotlib import pyplot as plt
from glob import glob 
import os 
#applying this code after the alignment step with code alignmentprocess.py

class ProcessDefects():

    def __init__(self):

        self.scale = 0.25
        self.threshold = 30
        #self.size = 2048, 2448
        self.size = 1536,2048 # the one I am using # for processing images without alignment 
        #self.size= 805,2048 # using this one for alignment

        #self.size=1540,2052

    def chip_detection(self, input_image):
        input_image= cv2.imread('D:/Uni/Vehcom/All Data/Data/onsite_data_collection/VEHCOM/images/2019-12-05/9_134109_1_0.png')
        test_image = cv2.Canny(input_image, 100, 200)
        # (retval, test_image) = cv2.threshold(input_image, 150, 255,cv2.THRESH_BINARY)
        (retval, test_image) = cv2.threshold(input_image, 150, 255,cv2.THRESH_BINARY)
        test_image2 = cv2.adaptiveThreshold(input_image,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,31,-25)
        test_image3 = cv2.adaptiveThreshold(input_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,31,-25)
        # test_image = cv2.cvtColor(test_image, cv2.COLOR_GRAY2RGB)
        # cv2.imshow('edges',edge_image)

        cv2.imshow('thresh',cv2.resize(test_image,None, fx=self.scale, fy=self.scale))
        cv2.imshow('adapt_thresh_mean',cv2.resize(test_image2,None, fx=self.scale, fy=self.scale))
        cv2.imshow('adapt_thresh_gaus',cv2.resize(test_image3,None, fx=self.scale, fy=self.scale))
        cv2.waitKey(0)
        return 0


    def process_images(self, images_directory_2):
        image_array=[]
        # self.size = image_array[0].shape
        print(self.size)
        avg_image = np.zeros(self.size, dtype=np.uint32)
        image_diff = []
        image_defect = []
        image_thresh = []
        if len(image_array) < 1:

            #for index in range(0, 22):

            for index in range(len(images_directory_2)):
                print('index',index)
                image_name = os.path.basename(images_directory_2[index])       
                split_image_name = image_name.split('_')
                image_number =split_image_name[3]

                #images_names_list.append(split_image_name[0])
                print('image name',image_name)
                print('image number',image_number)

       
                filename = images_directory_2[index]
                print('filename',filename)
                #filename = filename [ ]
                #plt.imshow(cv2.imread(filename),cmap='gray')
                #plt.show()
                #split = filename.split('/')
                
                #split = filename.split('/')[6].split('\\')[1] + '/' + filename.split('/')[6].split('\\')[2] # for alignment 
                
                split = filename.split('/')[5].split('\\')[1] + '/' + filename.split('/')[5].split('\\')[2] # to process images without alignment
                print('split',split)
               
                #loading the output images after alignment with code alignmentprocess.py

               
                #filename='D:/Uni/Vehcom/All Data/30images22/topalignment/47/47_153007_1_' + str(index) + '.png'               
                #filename='D:/Uni/Vehcom/All Data/Rule-based data for thesis/Canny vertical/Alignment/82/82_100719_1_' + str(index) + '.png'               
                #print(filename)
                image_array.append(cv2.imread(filename, 0)) # add the cropping boundries when only applying  images without alignment 
                #cv2.imshow('image ' + str(index+1),cv2.resize(image_array[index],None, fx=self.scale, fy=self.scale))
                #print('entered for')
                avg_image = avg_image + image_array[index]

        else:
            for index in range(0, len(image_array)):
                cv2.imshow('image ' + str(index+1),cv2.resize(image_array[index],None, fx=self.scale, fy=self.scale))
                #print('entered else')
                avg_image = avg_image + image_array[index]

        avg_image = avg_image / len(image_array)
        avg_image = np.uint8(avg_image)

        #cv2.imshow("avg", cv2.resize(
         #   avg_image, None, fx=self.scale, fy=self.scale))
        

        #cv2.imwrite('C:/Users/abdo_/Desktop/average/198shiftedcenter.png',avg_image)

        
        #plt.imshow(avg_image,cmap='gray')
        #plt.show()

        for index in range(0, len(image_array)):
            image_diff.append(
                abs(np.int32(avg_image) - np.int32(image_array[index])))
            image_diff[index] = np.uint8(image_diff[index])
            # print("index ", index)
            # cv2.imshow('diff', cv2.resize(image_diff[index], None, fx=self.scale, fy=self.scale))
            (retval, img_thresh) = cv2.threshold(
                image_diff[index], self.threshold, 255, cv2.THRESH_BINARY)
            image_thresh.append(img_thresh)

            orig_image = cv2.cvtColor(
                image_array[index], cv2.COLOR_GRAY2RGB)
            image_overlay = cv2.cvtColor(
                image_thresh[index], cv2.COLOR_GRAY2RGB)
            orig_image = cv2.subtract(orig_image, image_overlay)
            # cv2.imshow('step 1', cv2.resize(orig_image, None, fx=self.scale, fy=self.scale))
            image_overlay[:, :, 0] = np.zeros(
                [image_overlay.shape[0], image_overlay.shape[1]])
            image_overlay[:, :, 1] = np.zeros(
                [image_overlay.shape[0], image_overlay.shape[1]])
            # cv2.imshow('step 2', cv2.resize(image_overlay, None, fx=self.scale, fy=self.scale))
            orig_image = cv2.add(orig_image, image_overlay)
            # cv2.imshow('step 3', cv2.resize(orig_image, None, fx=self.scale, fy=self.scale))
            image_defect.append(orig_image)
            # cv2.imshow('defect ' + str(index+1), cv2.resize(image_defect[index], None, fx=self.scale, fy=self.scale))

            # print(image_overlay)
        #img_after_shift=cv2.imread('C:/Users/abdo_/Desktop/presentation/average.png')  

        #cv2.imshow('image', cv2.resize(
         #   image_array[4], None, fx=self.scale, fy=self.scale))
        #cv2.imshow('image',img_after_shift)#, cv2.resize(
            #img_after_shift, None, fx=self.scale, fy=self.scale))
            

        #cv2.imshow('defect ' + str(6 + 1),
         #         cv2.resize(image_defect[4], None, fx=self.scale, fy=self.scale))

        #cv2.imshow('defect ' + str(6 + 1),qqqqqqqq
         #         cv2.resize(img_after_shift, None, fx=self.scale, fy=self.scale))
        #cv2.imshow('defect',img_after_shift)
        #for image_number in range (0,22):
        for image_number_number in range(len(images_directory_2)):
            print('image_number_number',image_number_number)
            image_name = os.path.basename(images_directory_2[image_number_number])       
            split_image_name = image_name.split('_')
            image_number =split_image_name[3]

            #images_names_list.append(split_image_name[0])
            print('image name',image_name)
            print('image number',image_number)

   
            filename = images_directory_2[image_number_number]
            print('filename',filename)
            #split = filename.split('/')
            #split = filename.split('/')[6].split('\\')[1] + '/' + filename.split('/')[6].split('\\')[2] #for alignment
            split = filename.split('/')[5].split('\\')[1] + '/' + filename.split('/')[5].split('\\')[2] # to process without alignment
            print('split',split)
            required= image_defect[image_number_number]
            #plt.imshow(required,cmap='gray')
            #plt.show()

            
            #filetosave_top_canny='D:/Uni/Vehcom/All Data/Rule-based data for thesis/Canny vertical/Thresholding_50/82/82_100719_1_' + str(image_number) + '.png'\\
            #filetosave_top_canny='D:/Uni/Vehcom/All Data/Rule-based data for thesis/without_alignment/Thresholding_100/' + str(split)
            #filetosave_top_canny='D:/Uni/Vehcom/All Data/Rule-based data for thesis/Canny vertical/Thresholding_100/' + str(split)
            #filetosave_top_canny='D:/Uni/Vehcom/All Data/Rule-based data for thesis/Canny vertical/Thresholding_30/' + str(split)
            filetosave_top_canny='D:/Uni/Vehcom/All Data/Rule-based data for thesis/without_alignment/Thresholding_30/' + str(split)

            print('file to save',filetosave_top_canny)
            cv2.imwrite(filetosave_top_canny,required[0:805,0:2048])
            print('first saved')


            #plt.imshow(img_after_shift,cmap='gray')
            #plt.imshow(required,cmap='gray')
            #plt.imshow(required)
            #plt.show()

        #plt.imshow(orig_image[4],cmap='gray')
        #plt.show()
        
       

        # cv2.imshow('thresh', cv2.resize(img_thresh, None, fx=scale, fy=scale))
        # cv2.imshow('defect', cv2.resize(image_defect[21], None, fx=scale, fy=scale))

        # image_overlay = cv2.colorChange()
        # print(image_array_converted[21].dtype, img_thresh.dtype)
        # cv2.imshow('overlay', cv2.resize(image_overlay, None, fx=scale, fy=scale))
        # print(image_array_converted[0].shape)
        # print(image_array_converted[0].dtype)
        #cv2.waitKey(0)

        return 0


if __name__ == '__main__':
    
    folders = glob ('D:/Uni/Vehcom/All Data/Rule-based data for thesis/Test set gears/*')
    #folders = glob ('D:/Uni/Vehcom/All Data/Rule-based data for thesis/Canny vertical/Alignment/*')

    for fold in folders:
        print('fold',fold)
        directory = (fold+"/*")    
        print('directory ',directory)
        images_directory = glob (directory)
        print('images_directory',images_directory)
        process = ProcessDefects()
        result = process.process_images(images_directory)
        
        
        # for index in range(len(images_directory)):
        #         print('index',index)
        #         image_name = os.path.basename(images_directory[index])       
        #         split_image_name = image_name.split('_')
        #         image_number =split_image_name[3]

        #         #images_names_list.append(split_image_name[0])
        #         print('image name',image_name)
        #         print('image number',image_number)

       
        #         filename = images_directory[index]
        #         print('filename',filename)
        #         #split = filename.split('/')
        #         split = filename.split('/')[6].split('\\')[1] + '/' + filename.split('/')[6].split('\\')[2]
        #         print('split',split)


                
        # #chip = process.chip_detection()
        # # print(result)