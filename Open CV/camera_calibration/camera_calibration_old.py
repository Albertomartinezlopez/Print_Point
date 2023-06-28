import numpy as np
import cv2 as cv
import glob
import pickle

chessboardSize = (8,5)
frameSize = (821,616)


# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

size_of_chessboard_squares_mm = 30
objp = objp * size_of_chessboard_squares_mm


# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.


images = glob.glob('cam_cali/*.jpg')

for image in images:

    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
    
    lwr = np.array([0, 0, 0])
    upr = np.array([0, 0, 128])
    hsv = cv.cvtColor(gray, cv.COLOR_BGR2HSV)
    msk = cv.inRange(hsv, lwr, upr)

    # Extract chess-board
    krn = cv.getStructuringElement(cv.MORPH_RECT, (50, 30))
    dlt = cv.dilate(msk, krn, iterations=5)
    res = 255 - cv.bitwise_and(dlt, msk)
    #cv.imshow('gray', res)
    #cv.waitKey(1000)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(res, chessboardSize, None)

    print(ret)
    print(corners)
    # If found, add object points, image points (after refining them)
    if ret == True:

        objpoints.append(objp)
        corners2 = cv.cornerSubPix(res, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(1000)


#cv.destroyAllWindows()

############## CALIBRATION #######################################################
print(len(objpoints))
print(len(imgpoints))
ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)

# Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
pickle.dump((cameraMatrix, dist), open( "calibration.pkl", "wb" ))
# pickle.dump(cameraMatrix, open( "cameraMatrix.pkl", "wb" ))
# pickle.dump(dist, open( "dist.pkl", "wb" ))


############## UNDISTORTION #####################################################

img = cv.imread('img_cv/aruco2.png')#cam_cali/cali22.jpg')
h,  w = img.shape[:2]
newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))


# Undistort
dst = cv.undistort(img, cameraMatrix, dist, None, newCameraMatrix)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('caliResult1.png', dst)


# Undistort with Remapping
mapx, mapy = cv.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (w,h), 5)
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('caliResult2.png', dst)


# Reprojection Error
mean_error = 0

for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error

print( "total error: {}".format(mean_error/len(objpoints)) )
