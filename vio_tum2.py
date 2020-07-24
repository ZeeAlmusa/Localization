#!/usr/bin/env python
# coding: utf-8

__author__ = "Masahiko Toyoshi"
__copyright__ = "Copyright 2007, Masahiko Toyoshi."
__license__ = "GPL"
__version__ = "1.0.0"


import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import serial



def calc_euclid_dist(p1, p2):
    a = math.pow((p1[0] - p2[0]), 2.0) + math.pow((p1[1] - p2[1]), 2.0)
    return math.sqrt(a)


def main():

    camera_matrix = np.load("camera_matrix.npy")
    cap = cv2.VideoCapture(0)

    feature_detector = cv2.FastFeatureDetector_create(threshold=25,
                                                      nonmaxSuppression=True)
    orb = cv2.ORB_create()
    
    lk_params = dict(winSize=(11, 11),
                     criteria=(cv2.TERM_CRITERIA_EPS |
                               cv2.TERM_CRITERIA_COUNT, 30, 0.03))

    current_pos = np.zeros((3, 1))
    current_rot = np.eye(3)
    
    old_odometry = np.zeros((4,1))
    new_odometry = np.zeros((4,1))
    
    # create graph.
    # position_figure = plt.figure()
    # position_axes = position_figure.add_subplot(1, 1, 1)
    # error_figure = plt.figure()
    # rotation_error_axes = error_figure.add_subplot(1, 1, 1)
    # rotation_error_list = []
    # frame_index_list = []

    # position_axes.set_aspect('equal', adjustable='box')
   


    prev_image = None
    scale = 0
    counter = 0
 
    
    try:
   
        while True:
            
            serial_input = arduino.readline()
            
            #print(serial_input)
            if serial_input.isascii() and counter >= 100:
                
                sensors = np.array(serial_input.decode('ascii').rstrip('\r\n').split('/'), dtype=np.float) #[x,y,yaw_odometry, yaw_IMU]      
                new_odometry = sensors
                print(sensors)
                scale = np.sqrt((new_odometry[0] - old_odometry[0])**2 +(new_odometry[1] - old_odometry[1])**2 )
            
            counter = counter + 1
            # load image
            ret, image = cap.read()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            image = image * 15
            
            # main process
            keypoint = feature_detector.detect(image, None)
            
            #keypoint = orb.detect(image, None)
            if prev_image is None:
                
                prev_image = image
                prev_keypoint = keypoint
                continue
            
            
            points = np.array([np.float32(x.pt) for x in prev_keypoint])
                          
            if points.size > 0:
                p1, st, err = cv2.calcOpticalFlowPyrLK(prev_image,
                                                       image, points,
                                                       None, **lk_params)
                st = st.flatten()
                
                p1 = p1[st==1,:]
                points = points[st==1,:]
                
                E, mask = cv2.findEssentialMat(p1, points, camera_matrix,
                                               cv2.RANSAC, 0.999, 1.0, None)
        
                retval, R, t, mask = cv2.recoverPose(E, p1, points, camera_matrix)
        

                # calc scale from ground truth if exists.
                if((abs(t[2]) > abs(t[0])) and (abs(t[2])>abs(t[1]))):
                    current_pos += current_rot.dot(t) * scale
                    current_rot = R.dot(current_rot)
                 
                    plt.scatter(current_pos[2], current_pos[0])
                    plt.axis('equal')
                 
              
                    
                
                print(points.size)
                print(current_pos)
                print("=======================")
                #img = cv2.drawKeypoints(image, keypoint, None)
        
                # cv2.imshow('image', image)
                # cv2.imshow('feature', img)
                # cv2.waitKey(1)
    
            prev_image = image
            prev_keypoint = keypoint
            old_odometry = new_odometry
            
            
        
            
        # position_figure.savefig("position_plot.png")
        # rotation_error_axes.bar(frame_index_list, rotation_error_list)
        # error_figure.savefig("error.png")
    
        
    except:
        
        print("ending program.")
        arduino.close()
     
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
   
    arduino = serial.Serial('com5', 19200)
 
    main()
  
 
     
      
        