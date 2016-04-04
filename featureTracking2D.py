import numpy as np
import cv2
from PIL import Image
from draw_matches import drawMatches
from matplotlib import pyplot as plt
import sys
import operator

# Set default window location for image display
windowX,windowY = 0,0

# Start video capturing
cap = cv2.VideoCapture(0)

# Read first frame as reference
ret, img1 = cap.read()

# Function to show the final images
def showMe(name,image):
    global windowX,windowY
    cv2.namedWindow(name,cv2.WINDOW_AUTOSIZE)
    cv2.imshow(name,image)
    #Get image shape and update window location so windows don't come up on top of eachother
    tup = image.shape
    windowX= windowX+tup[1] +4
    if windowX > 1366:
        windowX=0
        windowY= windowY+tup[0]
        if windowY > 784:
            windowY=0

landmarks_kp=[]
landmarks_des=[]
landmarks=[]
# Start the loop for main function
while(cv2.waitKey(1) & (0xFF != ord('q'))):

    # Capture frame-by-frame
    ret, img2 = cap.read()

    # Create ORB detector with 1000 keypoints with a scaling pyramid factor
    # of 1.2

    # ORB is basically a fusion of FAST keypoint detector and BRIEF descriptor with many modifications to enhance the
    # performance. First it use FAST to find keypoints, then apply Harris corner measure to find top N points among
    # them. It also use pyramid to produce multiscale-features. But one roblem is that, FAST doesn't compute the
    # orientation. So what about rotation invariance? Authors came up with following modification.
    # It computes the intensity weighted centroid of the patch with located corner at center. The direction of the
    # vector from this corner point to centroid gives the orientation. To improve the rotation invariance, moments are
    # computed with x and y which should be in a circular region of radius r, where r is the size of the patch.
    # Now for descriptors, ORB use BRIEF descriptors. But we have already seen that BRIEF performs poorly with rotation.
    # So what ORB does is to "Steer" BRIEF according to the orientation of keypoints. For any feature set of n binary
    # tests at location (x_i, y_i), define a 2*n matrix, S which contains the coordinates of these pixels.
    # Then using the orientation of patch, theta, its rotation matrix is found and rotates the S to get
    # steered(rotated) version Stheta.
    # ORB discretize the angle to increments of 2*pi/30 (12 degrees), and construct a lookup table of precomputed
    # BRIEF patterns. As long as the keypoint orientation theta is consistent across views, the correct set of points
    # Stheta will be used to compute its descriptor.
    # BRIEF has an important property that each bit feature has a large variance and a mean near 0.5. But once it is
    # oriented along keypoint direction, it loses this property and become more distributed. High variance makes a
    # feature more discriminative, since it responds differentially to inputs. Another desirable property is to have
    # the tests uncorrelated, since then each test will contribute to the result. To resolve all these, ORB runs a
    # greedy search among all possible binary tests to find the ones that have both high variance and means close to
    # 0.5, as well as being uncorrelated. The result is called rBRIEF.
    # For descriptor matching, multi-probe LSH which improves on the traditional LSH, is used. The paper says ORB
    # is much faster than SURF and SIFT and ORB descriptor works better than SURF. ORB is a good choice in low-power
    # devices for panorama stitching etc.
    # Source: openCV Documentation

    orb = cv2.ORB(100, 1.2)

    # Detect keypoints of first image
    (kp1,des1) = orb.detectAndCompute(img1, None)

    # Detect keypoints of second image
    (kp2,des2) = orb.detectAndCompute(img2, None)

    # Create matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Do matching
    matches = bf.match(des1,des2)

    # Sort the matches based on distance.  Least distance is better
    matches = sorted(matches, key=lambda val: val.distance)

    # Filter the matches with minimum distance based on user selected threshold
    good = []
    for m in matches:
        if m.distance < 15:
            good.append(m)
    landmarks.append(len(good[:10]))
    landmarks_kp.append(kp2)
    landmarks_des.append(des2)

    '''
    # Display the coordinates of selected matches in both images
    for m in good:
        print("(x,y) in image 1")
        print(kp2[m.trainIdx].pt)
        print("(x,y) in image 2")
        print(kp1[m.queryIdx].pt)
    '''
    # Show only the top 10 matches - also save a copy for use later
    out = drawMatches(img1, kp1, img2, kp2, good[:10])

    # Display the matched image
    showMe('test', out)
    plt.show()

    # Find Homography using RANSAC Algorithm
    '''
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    matchesMask = mask.ravel().tolist()
    h,w,l = img1.shape
    img22 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    flattened  = [val for sublist in dst for val in sublist]
    #dst =  reduce(operator.add, dst[i])
    print("flattened")
    print(flattened)
    img22 = cv2.polylines(img22,flattened,True,255,3, cv2.CV_AA)
    print(img22.shape)
    '''
    # Replace image 2 by image 1
    img1=img2
    print(landmarks)
    for i in range(len(landmarks_des)):
        # Do matching
        matches = bf.match(des2,landmarks_des[i])

        # Sort the matches based on distance.  Least distance is better
        matches = sorted(matches, key=lambda val: val.distance)
        good = []
        for m in matches:
            if m.distance < 15:
                good.append(m)
        if(len(good[:10])>=5):
            print("Current Frame %d has %d matches in Frame %d" % (len(landmarks_des),len(good[:10]),i))


cap.release()
cv2.destroyAllWindows()
sys.exit()
