import cv2 as cv
import numpy as np
from glob import glob
import os
from PIL import Image
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--imgs_dir")
parser.add_argument("--save_neg", action="store_true", default=False,
                        help="Save Negative Image?")
args = parser.parse_args()
data_folder = args.imgs_dir
save_neg = args.save_neg
mouseX=0 ; mouseY=0
def draw_circle(event,x,y,flags,param):
    global mouseX,mouseY, pts
    if event == cv.EVENT_LBUTTONDOWN:
        cv.circle(img,(x,y),2,(0,255,0),-1)
        mouseX,mouseY = x,y
        if len(pts)>0:
            cv.line(img, (pts[-1][0], pts[-1][1]), (x,y), (0,255,0))
        pts.append([mouseX,mouseY])

for image_file in glob(os.path.join(data_folder, 'DJI*.JPG')):
    print "Working on Image: {}".format(os.path.basename(image_file))
    if os.path.join(os.path.dirname(image_file), "masked_" + os.path.basename(image_file)[:-3]+ "jpg") in glob(os.path.join(data_folder, '*.jpg')):
    loop = True
    while(loop):
        pts = []
        img = cv.imread(image_file)
        ORG_img = np.copy(img)
        img = cv.resize(img, (0,0), fx=.25, fy=.25)
        org_img = np.copy(img)

        img = cv.cvtColor(img, cv.COLOR_RGB2RGBA)
        img = np.multiply(img, np.array([[1,1,1,0]])).astype(np.uint8)
        cv.namedWindow('image')
        cv.setMouseCallback('image',draw_circle)

        while(loop):
            cv.imshow('image', img)
            k = cv.waitKey(20) & 0xFF
            if k == 27:
                break
            # elif k == ord('a'):
            #     print mouseX,mouseY
            #     pts = pts[:-1]
            #     mouseX,mouseY = pts[-1]
            elif k == ord('x'):
                break
            elif k == ord('s'):
                loop = False
                print "\tSkipping Image: {}".format(os.path.basename(image_file))
            elif k ==ord('q'):
                print "User Quit!"
                exit()


        if loop:
            background_color=[255,0,0,66]
            masked_img = np.copy(org_img)
            cv.fillPoly(img, np.array([pts]), background_color)
            gt_bg = np.all(img == background_color, axis=2)
            img[gt_bg==False]=[0,0,0,0]
            # masked_img[gt_bg==False]=[0,0,0]
            masked_img[gt_bg==False]=[0,0,0]
            masked_img[gt_bg==True]=[1,1,1]
            cv.imshow("Filled Image", img)
            k=cv.waitKey()
            if k==ord('x'):
                 cv.destroyAllWindows()
                 # masked_img = Image.fromarray(masked_img[:,:,:3])
                 masked_img = cv.resize(masked_img, (0,0), fx=4, fy=4)
                 print "\tSaving masked_{}.jpg".format(os.path.basename(image_file).split('.')[0])
                 cv.imwrite("{}/masked_{}.jpg".format(os.path.dirname(image_file), os.path.basename(image_file).split('.')[0]), masked_img[:,:,:3]*ORG_img)
                 # masked_img.save("{}/masked_{}.jpg".format(os.path.dirname(image_file), os.path.basename(image_file).split('.')[0]))
                 if save_neg: np.save("{}/{}".format(os.path.dirname(image_file), os.path.basename(image_file).split('.')[0]), np.argwhere(img[:,:,-1]!=0))
                 break
            elif k==ord('r'):
                print "Repeating on Image: {}".format(os.path.basename(image_file))
                cv.destroyAllWindows()

            elif k ==ord('q'):
                print "User Quit!"
                exit()
