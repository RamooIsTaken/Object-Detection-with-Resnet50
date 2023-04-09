from tensorflow import keras
from keras.applications import ResNet50
from keras.applications.resnet import preprocess_input
from keras.utils import img_to_array
from keras.applications import imagenet_utils
import numpy as np
import cv2

from ımgPyramid import *

# ilklendirme parametreleri

WIDTH = 600
HEIGHT = 600
PYRSCALE = 1.5
WIN_STEP = 16
ROI_SIZE = (200,150)
INPUT_SIZE = (224,224)


model = ResNet50(weights="imagenet",include_top=True)

orj = cv2.imread("husky.jpg")
orj = cv2.resize(orj,(WIDTH,HEIGHT))

H,W = orj.shape[:2]

#Image Pyramid 
pyramid = ımagePyramid(orj,PYRSCALE,ROI_SIZE)

rois = []
locs = []

for image in pyramid :
    scale = W/float(image.shape[1])

    for (x,y,roiOrg) in slidingWindow(image,WIN_STEP,ROI_SIZE):

        x = int(x*scale)
        y = int(y*scale)
        w = int(ROI_SIZE[0]*scale)
        h = int(ROI_SIZE[1]*scale)

        roi = cv2.resize(roiOrg,INPUT_SIZE)
        roi = img_to_array(roi)
        roi = preprocess_input(roi)
        
        rois.append(roi)
        locs.append((x,y,x+w,y+h))


rois = np.array(rois,dtype="float32")

#Classification


preds = model.predict(rois)

preds = imagenet_utils.decode_predictions(preds,1)

labels = {}
minConf = 0.9

for (i,p) in enumerate(preds) :

    (_,label,prob) = p[0]

    if prob >= minConf :

        box = locs[i]
        L = labels.get(label,[])
        L.append((box,prob))
        labels[label] = L


for label in labels.keys() :
    
    clone = orj.copy()

    for (box,prob) in labels[label] :
        
        (startX,startY,endX,endY) = box
        cv2.rectangle(clone,(startX,startY),(endX,endY),(0,255,0),2)

    plt.figure()
    plt.imshow(clone)
    plt.show()

    clone = orj.copy()

    boxes = np.array([p[0] for p in labels[label]])
    proba = np.array([p[1] for p in labels[label]])

    boxes = nonMaxiSuppression(boxes,proba)

    for (startX,startY,endX,endY) in boxes :
        cv2.rectangle(clone,(startX,startY),(endX,endY),(0,255,0),2)
        y = startY-10 if startY-10 > 10 else startY + 10
        cv2.putText(clone,label,(startX,y),cv2.FONT_HERSHEY_DUPLEX,0.45,(0,255,0),2)

    cv2.imshow("Maxima",clone)
    if cv2.waitKey(100) & 0xFF == ord("q") : break

