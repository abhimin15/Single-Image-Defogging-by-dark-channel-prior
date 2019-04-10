import math
import numpy as np
import cv2

def AirlightEstimation(image,dep):
    [h,w] = image.shape[:2]
    imsz = h*w
    numpx = int(max(math.floor(imsz/1000),1))
    darkvec = dep.reshape(imsz,1);
    imvec = image.reshape(imsz,3)
 
    indices = darkvec.argsort();
    indices = indices[imsz-numpx::]
 
    atmlit = np.zeros([1,3])
    for ind in range(1,numpx):
        atmlit = atmlit + imvec[indices[ind]]
 
    air = atmlit / numpx
    return air
	
def DepthFusion(image,sz):
    b,g,r = cv2.split(image)
    dist = cv2.min(cv2.min(r,g),b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(sz,sz))
    dep = cv2.erode(dist,kernel)
    return dep
	
def DepthEstimate(image,air,sz):
    alpha = 0.95
    im3 = np.empty(image.shape,image.dtype)
 
    for ind in range(0,3):
        im3[:,:,ind] = image[:,:,ind]/air[0,ind]
 
    depth = 1 - alpha*DepthFusion(im3,sz)
    return depth
	
def Depthfilter(image,p,r,eps):
    mean1 = cv2.boxFilter(image,cv2.CV_64F,(r,r))
    mean2 = cv2.boxFilter(p, cv2.CV_64F,(r,r))
    mean3 = cv2.boxFilter(image*p,cv2.CV_64F,(r,r))
    cov = mean3 - mean1*mean2
 
    meanI = cv2.boxFilter(image*image,cv2.CV_64F,(r,r))
    var_I   = meanI - mean1*mean2
 
    a = cov/(var_I + eps)
    b = mean2 - a*mean1
 
    a_mean = cv2.boxFilter(a,cv2.CV_64F,(r,r))
    b_mean = cv2.boxFilter(b,cv2.CV_64F,(r,r))
 
    q = a_mean*image + b_mean
    return q
	
def DepthMap(image,et):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY);
    gray = np.float64(gray)/255;
    r = 60
    eps = 0.0001
    t = Depthfilter(gray,et,r,eps)
 
    return t
	
def SceneReflect(image,t,air,tx = 0.1):
    res = np.empty(image.shape,image.dtype)
    t = cv2.max(t,tx)
 
    for ind in range(0,3):
        res[:,:,ind] = (image[:,:,ind]-air[0,ind])/t + air[0,ind]
 
    return res

if __name__ == '__main__':
    fn = 'media/hazed.jpg'
    src = cv2.imread(fn);
    I = src.astype('float64')/255
    dep = DepthFusion(I,15)
    air = AirlightEstimation(I,dep);
    te = DepthEstimate(I,air,15)
    t = DepthMap(src,te)
    J = SceneReflect(I,t,air,0.1)
    arr = np.hstack((I, J))
    #cv2.imshow("contrast", arr)
    cv2.imwrite("defog.png", J*255 )
    cv2.imwrite("contrast.png", arr*255)
    cv2.waitKey(0)