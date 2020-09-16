import cv2 
import numpy as np
import pafy # pip3 install youtube-dl
import imutils # pip install imutils
from collections import Counter, defaultdict

x = input("Enter 1 for youtube, 2 for video,3 for webcam : ")
if x == '1':
    text = input("Enter your link: ")
    url = text
    video = pafy.new(url)
    best = video.getbest(preftype="mp4")

    cap = cv2.VideoCapture() #Youtube
    cap.open(best.url)

elif x == '2':
    x = input("Enter your video: Type Format ..avi or .mp4 : ")
    cap = cv2.VideoCapture(x) # Import Video File

elif x == '3':
    cap = cv2.VideoCapture(0) #webcam 
    cap.set(3, 1000)
    cap.set(4, 1000)
    print(cap.get(3))
    print(cap.get(4))

else:
    print("Your input is wrong")


# firstframe_path = input("Please Type your frame Image & Format Type : ")
# cap = cv2.imread(x) # Import Video File

#first frame extraction 
def FrameCapture(path): 
      
    # Path to video file 
    vidObj = cv2.VideoCapture(path) 
  
    # Used as counter variable 
    count = 0
  
    # checks whether frames were extracted 
    success = 1
  
    while success: 
  
        # vidObj object calls read 
        # function extract frames 
        success, image = vidObj.read() 
  
        # Saves the frames with frame-count 
        cv2.imwrite("frame%d.jpg" % count, image) 
  
        count = 0 

        break

  
# Driver Code 
if __name__ == '__main__': 
  
    # Calling the function 
    FrameCapture(x) 


# # location of first frame
# firstframe_path =r'FrameNo88_Train.png'
firstframe_path = r'frame0.jpg'
firstframe = cv2.imread(firstframe_path)
firstframe_gray = cv2.cvtColor(firstframe, cv2.COLOR_BGR2GRAY)
firstframe_blur = cv2.GaussianBlur(firstframe_gray,(21,21),0)

#---------------------------------
#size the window first
#---------------------------------
cv2.namedWindow('CannyEdgeDet',cv2.WINDOW_NORMAL)
cv2.namedWindow('Abandoned Object Detection',cv2.WINDOW_NORMAL)
cv2.namedWindow('Morph_CLOSE',cv2.WINDOW_NORMAL)

# location of video
file_path = x

cap = cv2.VideoCapture(file_path)

consecutiveframe=20

track_temp=[]
track_master=[]
track_temp2=[]

top_contour_dict = defaultdict(int)
obj_detected_dict = defaultdict(int)

frameno = 0
while (cap.isOpened()):
    ret, frame = cap.read()
    cv2.imshow('main',frame)
    
    if ret==0:
        break
    
    frameno = frameno + 1
    #cv2.putText(frame,'%s%.f'%('Frameno:',frameno), (400,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
    
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_blur = cv2.GaussianBlur(frame_gray,(21,21),0)
     
    frame_diff = cv2.absdiff(firstframe, frame)
  
    #Canny Edge Detection
    edged = cv2.Canny(frame_diff,10,200) #any gradient between 30 and 150 are considered edges
    cv2.imshow('CannyEdgeDet',edged)
    kernel2 = np.ones((5,5),np.uint8) #higher the kernel, eg (10,10), more will be eroded or dilated
    thresh2 = cv2.morphologyEx(edged,cv2.MORPH_CLOSE, kernel2,iterations=2)
    cv2.imshow('Morph_Close', thresh2)
    
    #Create a copy of the thresh to find contours    
    #(_,cnts, _) = cv2.findContours(thresh2.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    (cnts, _) = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    mycnts =[] # every new frame, set to empty list. 
    # loop over the contours
    for c in cnts:


        # Calculate Centroid using cv2.moments
        M = cv2.moments(c)
        if M['m00'] == 0: 
            pass
        else:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])


            #----------------------------------------------------------------
            # Set contour criteria
            #----------------------------------------------------------------
            
            if cv2.contourArea(c) < 200 or cv2.contourArea(c)> 400:
                pass
            else:
                mycnts.append(c)
                  
                # compute the bounding box for the contour, draw it on the frame,
                # and update the text
                (x, y, w, h) = cv2.boundingRect(c)
                # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                #cv2.putText(frame,'C %s,%s,%.0f'%(cx,cy,cx+cy), (cx,cy),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0),2) 
                
                
                #Store the cx+cy, a single value into a list ; max length of 10000
                #Once hit 10000, tranfer top 20 points to dictionary ; empty list
                sumcxcy=cx+cy
                
                
                
                #track_list.append(cx+cy)
                track_temp.append([cx+cy,frameno])
                
                
                track_master.append([cx+cy,frameno])
                countuniqueframe = set(j for i, j in track_master) # get a set of unique frameno. then len(countuniqueframe)
                
                #----------------------------------------------------------------
                # Store history of frames ; no. of frames stored set by 'consecutiveframe' ;
                # if no. of no. of unique frames > consecutiveframes, then 'pop or remove' the earliest frame ; defined by
                # minframeno. Objective is to count the same values occurs in all the frames under this list. if yes, 
                # it is likely that it is a stationary object and not a passing object (walking) 
                # And the value is stored separately in top_contour_dict , and counted each time. This dict is the master
                # dict to store the list of suspecious object. Ideally, it should be a short list. if there is a long list
                # there will be many false detection. To keep the list short, increase the 'consecutiveframe'.
                # Keep the number of frames to , remove the minframeno.; but hard to remove, rather form a new list without
                #the minframeno.
                #----------------------------------------------------------------
                if len(countuniqueframe)>consecutiveframe or False: 
                    minframeno=min(j for i, j in track_master)
                    for i, j in track_master:
                        if j != minframeno: # get a new list. omit the those with the minframeno
                            track_temp2.append([i,j])
                
                    track_master=list(track_temp2) # transfer to the master list
                    track_temp2=[]
                    
                
                #print 'After',track_master
                
                #count each of the sumcxcy
                #if the same sumcxcy occurs in all the frames, store in master contour dictionary, add 1
                
                countcxcy = Counter(i for i, j in track_master)
                #print countcxcy
                #example countcxcy : Counter({544: 1, 537: 1, 530: 1, 523: 1, 516: 1})
                #if j which is the count occurs in all the frame, store the sumcxcy in dictionary, add 1
                for i,j in countcxcy.items(): 
                    if j>=consecutiveframe:
                        top_contour_dict[i] += 1
  
                
                if sumcxcy in top_contour_dict:
                    if top_contour_dict[sumcxcy]>200:
                        cv2.rectangle(frame, (x, y), (x + w * 3, y + h * 3), (255, 0, 0), 2)
                        cv2.putText(frame,'%s'%('Abandoned object detected'), (cx,cy),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255),2)
                        print ('Detected : ', sumcxcy,frameno, obj_detected_dict)
                        
                        # Store those objects that are detected, and store the last frame that it happened.
                        # Need to find a way to clean the top_contour_dict, else contour will be detected after the 
                        # object is removed because the value is still in the dict.
                        # Method is to record the last frame that the object is detected with the Current Frame (frameno)
                        # if Current Frame - Last Frame detected > some big number say 100 x 3, then it means that 
                        # object may have been removed because it has not been detected for 100x3 frames.
                        
                        obj_detected_dict[sumcxcy]=frameno

    for i, j in obj_detected_dict.items():
        if frameno - obj_detected_dict[i]>200:
            print ('PopBefore',i, obj_detected_dict[i],frameno,obj_detected_dict)
            print ('PopBefore : top_contour :',top_contour_dict)
            obj_detected_dict.pop(i)
            
            # Set the count for eg 448 to zero. because it has not be 'activated' for 200 frames. Likely, to have been removed.
            top_contour_dict[i]=0
            print ('PopAfter',i, obj_detected_dict[i],frameno,obj_detected_dict)
            print ('PopAfter : top_contour :',top_contour_dict)

                        
    
    
    cv2.imshow('Abandoned Object Detection',frame)
         
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    
cap.release()
cv2.destroyAllWindows()


# In[ ]:




