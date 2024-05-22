import cv2
import numpy as np
import os
from os import listdir
from os.path import isfile, join
path = 'C:\\Users\\kyria\\Desktop\\Image-Segmentation\\Segmented-images'
binpath = 'C:\\Users\\kyria\\Desktop\\Image-Segmentation\\JustBinarisedImages'

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(grayscale_image, (5, 5), 0)
    ret, binary_image = cv2.threshold(blurred_image, 150, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (6,6))
    mask = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6,6))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.resize(mask, (1440, 720))
    cv2.imshow('Localized Image with New Bounding Boxes', mask) #uncomment to show the image
    cv2.waitKey(0) 
    cv2.destroyAllWindows()
    return binary_image , mask

def masked_iamge(image_path):
    img = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh2 = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (150,2))
    mask = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    bboxes = []
    bboxes_img = img.copy()
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    for cntr in contours:
        x,y,w,h = cv2.boundingRect(cntr)
        cv2.rectangle(bboxes_img, (x, y), (x+w, y+h), (0,0,255), 1)
        bboxes.append((x,y,w,h))
    #resized_image = cv2.resize(mask, (1440, 720))
    #cv2.imshow('Localized Image with New Bounding Boxes', bboxes_img) #uncomment to show the image
    #cv2.waitKey(0) 
    #cv2.destroyAllWindows()
    return bboxes_img

def localize_text(binary_image,mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    localized_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
    bounding_boxes = []
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h > 100:
            cv2.rectangle(localized_image, (x, y-5), (x+w, y+h+5), (0, 255, 0), 2)
            bounding_boxes.append((x, y, w, h))
    #localized_image = cv2.resize(localized_image, (1440, 720))
    #cv2.imshow('Localized Image with New Bounding Boxes', localized_image) #uncomment to show the image
    #cv2.waitKey(0) 
    #cv2.destroyAllWindows()
    return localized_image, bounding_boxes

def segment_text(image, bounding_boxes):
    segmented_text = []
    new_bounding_boxes = []
    
    bounding_boxes = sorted(bounding_boxes, key=lambda x: (x[1], x[0]))
    
    for (x, y, w, h) in bounding_boxes:
        segment = image[y:y+h, x:x+w]
        
        # Additional morphological operations to enhance segmentation
        kernel = np.ones((3, 3), np.uint8)
        segment = cv2.morphologyEx(segment, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(segment, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x_c, y_c, w_c, h_c = cv2.boundingRect(contour)
            if w_c * h_c > 10:
                x_new, y_new = x + x_c, y + y_c
                new_bounding_boxes.append((x_new, y_new, w_c, h_c))
                
                cv2.rectangle(image, (x_new, y_new), (x_new+w_c, y_new+h_c), (0, 255, 0), 2)
                segmented_text.append(segment[y_c:y_c+h_c, x_c:x_c+w_c])
    
    return segmented_text, new_bounding_boxes
onlyfiles = [f for f in listdir(binpath) if isfile(join(binpath, f))]
for file in onlyfiles:
    binary_image , mask = preprocess_image(os.path.join("image-data" , file))
    localized_image, bounding_boxes = localize_text(binary_image,mask)
    segmented_text, new_bounding_boxes = segment_text(binary_image, bounding_boxes)

    #cv2.imshow('Localized Image with New Bounding Boxes', localized_image) #uncomment to show the image
    #cv2.waitKey(0) 
    #cv2.destroyAllWindows()
    cv2.imwrite(os.path.join(path , f'image_{file.replace(".jpg", "")}_with_bounding_boxes.png'),localized_image)
    cv2.imwrite(os.path.join(path , f'image_{file.replace(".jpg", "")}_with_bounding_lines.png'),masked_iamge(os.path.join("image-data" , file)))
    
    for i, (x, y, w, h) in enumerate(new_bounding_boxes):
        segment = segmented_text[i]
        cv2.imwrite(os.path.join(path , f'image_{file.replace(".jpg", "")}_segment_{i}_with_bounding_box.png'),segment)
        #clean up might be needed after changing paths and amount of segments (less segments means old ones stay same with images)
