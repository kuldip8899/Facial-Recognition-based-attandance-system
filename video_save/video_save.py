#library
import numpy as np
import cv2

#for video capture
cap = cv2.VideoCapture(0)

# Get the resolutions
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the filename.
out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

#return frame.
while(cap.isOpened()):
    ret, frame = cap.read()
	#condition
    if ret==True:

        # write the  frame
        out.write(frame)
		#display an image in a window
        cv2.imshow('frame',frame)
		
		#terminate condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()