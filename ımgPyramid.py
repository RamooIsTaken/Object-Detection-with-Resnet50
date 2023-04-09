import cv2
import matplotlib.pyplot as plt
import numpy as np



def ımagePyramid(image,scale=1.5,minSize=(224,224)):
    """Piramit yöntemi"""
    yield image

    while True :
        w = int(image.shape[1]/scale)
        image = cv2.resize(image,dsize=(w,w))

        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break


        yield image

def slidingWindow(image,step,ws):   
    """Kayan pencere yöntemi ,ws pencerenin boyutu(w,h) olarak"""
    for y in range(0,image.shape[0]-ws[1],step):
        for x in range(0,image.shape[1]-ws[0],step):
            yield(x,y,image[y:y+ws[1],x:x+ws[0]])

def nonMaxiSuppression(boxes,probs = None,overlapThresh=0.3):

    if len(boxes) == 0 :
        return []
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    #alanı bul
    area = (x2-x1+1) * (y2-y1+1)
    idxs = y2

    #probability
    if probs is not None :
        idxs = probs
    #indeks sırala
    idxs = np.argsort(idxs)


    pick = [] #seçilen kutular

    while len(idxs) > 0 :
        last = len(idxs) - 1 
        i = idxs[last]
        pick.append(i)

        #en büyük ve en küçük x,y bul
        xx1 = np.maximum(x1[i],x1[idxs[:last]])
        yy1 = np.maximum(y1[i],y1[idxs[:last]])
        xx2 = np.minimum(x2[i],x2[idxs[:last]])
        yy2 = np.minimum(y2[i],y2[idxs[:last]])

        # w,h bul
        w = np.maximum(0,xx2-xx1+1)
        h = np.maximum(0,yy2-yy1+1)

        #overlap
        overlap = (w*h)/area[idxs[:last]]
        
        idxs = np.delete(idxs,np.concatenate(([last],np.where(overlap > overlapThresh)[0])))
    
    return boxes[pick].astype("int")


"""img = cv2.imread("husky.jpg")
im = ımagePyramid(img,scale=1.5,minSize=(10,10))

for i,image in enumerate(im):
    print(i)
    if i == 8:
        plt.figure()
        plt.imshow(image)
        plt.show() """

"""img = cv2.imread("husky.jpg")
im = slidingWindow(img,5,(200,150))
for i, image in enumerate(im):
    print(i)
    if i == 1598 :
        plt.figure()
        plt.imshow(image[2])
        plt.show()
        print(image[0],image[1])"""





 