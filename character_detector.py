import cv2
import numpy as np

lowThreshold = 135
highThreshold = 180
centerThreshold = 125

# Select threshold
def isHighThreshold(gray_image, upbound, downbound):
    count = 0
    for j in range(upbound,downbound):
        for i in range(0,580):
            if (gray_image[j][i] > 160) & (gray_image[j][i] < 175):
                count = count + 1
        if count > 5000:
            return False
    return True

# Gray image to binary image 
def binaryCvt(gray_image):
    binary_image = gray_image.copy()
    x, y = np.shape(gray_image)
    
    # Devide image and select threshold for each  
    for k in range(0, 2):
        upbound = (int) (k*x/2)
        downbound = (int) ((k+1)*x/2)
        if isHighThreshold(gray_image, upbound, downbound):
            for i in range(upbound, downbound):
                for j in range(0, y):
                    binary_image[i][j] = 0 if (gray_image[i][j] > highThreshold) else 255
                for j in range((int)(y/2-15),(int)(y/2+15)):
                    binary_image[i][j] = 0 if (gray_image[i][j] > centerThreshold) else 255
        else:
            for i in range(upbound, downbound):
                for j in range(0, y):
                    binary_image[i][j] = 0 if (gray_image[i][j] > lowThreshold) else 255
    return binary_image

# Horizontal bold text
def boldText(binary_image):
    boldImage = binary_image.copy()
    x, y = np.shape(boldImage)
    for i in range(1, x-1):
        for j in range(1, y-1):
            if (binary_image[i][j] == 255):
                up = (binary_image[i-1][j] == 255)
                down = (binary_image[i+1][j] == 255)
                if (up & down):
                    boldImage[i][j+1] = 255
                    boldImage[i][j-1] = 255
    return boldImage

def seperateText(binary_image, contours):
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 80:
            continue
        if w < 10 or h < 10:
            continue
        ratio = h / w
        if ((ratio > 1.5) & (ratio < 2.3)):
            for i in range(x, x + w):
                binary_image[y + (int)(h/2)][i] = 0
                binary_image[y + (int)(h/2)+1][i] = 0
                binary_image[y + (int)(h/2)-1][i] = 0
                binary_image[y + (int)(h/2)-2][i] = 0
        if ((ratio > 2.4) & (ratio < 3.6)):
            for i in range(x, x + w):
                binary_image[y + (int)(h/3)][i] = 0
                binary_image[y + (int)(h/3)+1][i] = 0
                binary_image[y + (int)(h/3)-1][i] = 0
                binary_image[y + (int)(h/3)-2][i] = 0 

                binary_image[y + (int)(2*h/3)][i] = 0
                binary_image[y + (int)(2*h/3)+1][i] = 0
                binary_image[y + (int)(2*h/3)-1][i] = 0
                binary_image[y + (int)(2*h/3)-2][i] = 0 
    return binary_image

def printDetectImage(image):
    arrOutput = []
    txtOutput = ""
    ih = np.shape(image)[0]
    iw = np.shape(image)[1]
    det_image = image.copy()
    gray_image = cv2.cvtColor(det_image, cv2.COLOR_BGR2GRAY)
    binary_image = binaryCvt(gray_image)
    boldImage = boldText(binary_image)
    # Find contours and separate text
    contours, _ = cv2.findContours(boldImage, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    boldImage = seperateText(boldImage, contours)
    # Find contours after separate
    contours, _ = cv2.findContours(boldImage, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    for contour in contours:    
        x, y, w, h = cv2.boundingRect(contour)
        if w > 50:
            continue
        if h > 40:
            continue
        if w < 13 or h < 13:
            continue
        if w / h > 2.7 or h / w > 4:
            continue
        if (x > np.shape(boldImage)[1] - 57) | (x < 50):
            continue
        if (y > np.shape(boldImage)[0] - 20) | (y < 20):
            continue
        # Draw bounding rectangle 
        cv2.rectangle(det_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    cv2.imshow("output", det_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detectImage(image):
    arrOutput = []
    txtOutput = ""
    ih = np.shape(image)[0]
    iw = np.shape(image)[1]
    det_image = image.copy()
    gray_image = cv2.cvtColor(det_image, cv2.COLOR_BGR2GRAY)
    binary_image = binaryCvt(gray_image)
    boldImage = boldText(binary_image)
    # Find contours and separate text
    contours, _ = cv2.findContours(boldImage, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    boldImage = seperateText(boldImage, contours)
    # Find contours after separate
    contours, _ = cv2.findContours(boldImage, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    for contour in contours:    
        x, y, w, h = cv2.boundingRect(contour)
        if w > 50:
            continue
        if h > 40:
            continue
        if w < 13 or h < 13:
            continue
        if w / h > 2.7 or h / w > 4:
            continue
        if (x > np.shape(boldImage)[1] - 57) | (x < 50):
            continue
        if (y > np.shape(boldImage)[0] - 20) | (y < 20):
            continue
        # Draw bounding rectangle 
        # cv2.rectangle(det_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        x_center = (x + w/2) / iw
        y_center = (y + h/2) / ih
        w = w / iw
        h = h / ih 
        arrOutput.append([1,x_center,y_center,w,h])

    txtOutput += format(arrOutput[0][0], '.0f') + " " + format(arrOutput[0][1], '.6f') + " " + format(arrOutput[0][2], '.6f') + " " + format(arrOutput[0][3], '.6f') + " " + format(arrOutput[0][4], '.6f')
    for label in arrOutput:
        txtOutput += "\n"
        txtOutput += format(label[0], '.0f')
        for i in range(1,5):
            txtOutput += " " + format(label[i], '.6f')
    # printImage(det_image)
    return txtOutput

def read_label(img, str_output):
    gt = []
    for line in str_output.strip().split("\n"):
        tmp = line.strip().split(' ')

        w, h = img.shape[1], img.shape[0]
        x = [(float)(w.strip()) for w in tmp]

        x1 = int(x[1] * w)
        width = int(x[3] * w)

        y1 = int(x[2] * h)
        height = int(x[4] * h)

        gt += [(x1, y1, width, height, 0, 0, 0)]

    return gt

class HanNomOCR:

    def __init__(self, noise=50):
        """
        You should hard fix all the requirement parameters
        """
        self.name = 'HanNomOCR'
        self.noise = noise

        np.random.seed(1)

    def detect(self, img):
        label_test = detectImage(img)
        base_outputs = read_label(img, label_test)
        noise = np.random.randint(0, self.noise, size=(len(base_outputs), 4)) - (self.noise // 2)
        preds = []

        for i in range(len(base_outputs)):
            confidence = np.sum(np.abs(noise[i, :]))
            confidence = 1 - 1.0*confidence/200
            preds += [(confidence, base_outputs[i][0] + noise[i][0],
                              base_outputs[i][1] + noise[i][1],
                              base_outputs[i][2] + noise[i][2],
                              base_outputs[i][3] + noise[i][3])]
        # List of confidence, xcenter, ycenter, width, height
        return np.array(preds)
