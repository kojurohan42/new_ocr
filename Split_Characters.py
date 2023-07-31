import cv2
import copy
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

lowerModel =load_model('Model/devanagari_lowerMod_model.h5')
coreModel = load_model('Model/best_val_loss.hdf5')
upperModel = load_model('Model/devanagari_upperMod_model.h5')


def Split(Words):
    Characters = []

    for Word in Words:
        gray = cv2.cvtColor(Word, cv2.COLOR_BGR2GRAY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

        for i in range(morph.shape[0]):
            for j in range(morph.shape[1]):
                if not morph[i][j]:
                    morph[i][j] = 1

        div = gray / morph
        gray = np.array(cv2.normalize(
            div, div, 0, 255, cv2.NORM_MINMAX), np.uint8)

        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)


        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        thresh = cv2.morphologyEx(
            thresh, cv2.MORPH_DILATE, kernel, iterations=1)
        plt.imshow(thresh,cmap='gray')
        plt.show()
        
        original_thresh = thresh.copy()

        h_proj = np.sum(thresh, axis=1)
        print(len(h_proj))
        Max = np.max(h_proj)/1.1
        print(Max)

        upper = None
        lower = None
        for i in range(h_proj.shape[0]):
            proj = h_proj[i]
            

            if proj > Max and upper == None:
                upper = i
            elif proj < Max and upper != None and lower == None:
                lower = i
                break
        print(upper, lower)

        if thresh.shape[1] > 100:
            for row in range(upper-7 if upper>7 else upper,lower+7):
                thresh[row] = 0
        plt.imshow(thresh,cmap='gray')
        plt.show()
        base = identify_lower_baseline(thresh)
        print("lower",base)
        
        segments=character_segmentation(thresh)
        for simg in segments[0]:
            plt.imshow(simg)
            plt.show()
            seg = modifier_segmentation(simg,base)
            

    return Characters

def character_segmentation(bordered, thresh=255, min_seg=5, scheck=0.15):
    try:
        shape = bordered.shape
        check = int(scheck * shape[0])
        image = bordered[:]
        image = image[check:].T
        shape = image.shape
        bg = np.repeat(255, shape[1])
        bg_keys = []
        for row in range(1, shape[0]):
            if  (np.equal(bg, image[row]).any()):
                bg_keys.append(row) 
        print(bg_keys)

        lenkeys = len(bg_keys)-1
        new_keys = [bg_keys[1], bg_keys[-1]]
        print(new_keys)
        #print(lenkeys)
        for i in range(1, lenkeys):
            if (bg_keys[i+1] - bg_keys[i]) > check:
                new_keys.append(bg_keys[i])
                new_keys.append(bg_keys[i+1])
                #print(i)

        new_keys = sorted(new_keys)
        print(new_keys)
        segmented_templates = []
        first = new_keys[0]
        bounding_boxes = []
        for i in range(1,len(new_keys)-1,2):
            segment = bordered.T[first:new_keys[i]]
            if segment.shape[0]>=min_seg and segment.shape[1]>=min_seg:
                print('here')
                segmented_templates.append(segment.T)
                bounding_boxes.append((first, new_keys[i]))
            first = new_keys[i+1]       
        last_segment = bordered.T[new_keys[-2]:]

        
        if last_segment.shape[0]>=min_seg and last_segment.shape[1]>=min_seg:
            segmented_templates.append(last_segment.T)
            bounding_boxes.append((new_keys[-1], new_keys[-1]+last_segment.shape[0]))

        return(segmented_templates, bounding_boxes)
    except:
        return [bordered, (0, bordered.shape[1])]
    

def modifier_segmentation(bordered,base, thresh=255, min_seg=5, scheck=0.15):
    try:
        print("try")
        shape = bordered.shape
        print(shape)
        check = int(scheck * shape[1])
        checkhor = int(0.05 * shape[0])
        image = bordered[:]
        #find the background color for empty column
        bg = np.repeat(255, shape[1])
        bg_keys = []
        print(shape)
        print("ss")
        for row in range(1, shape[0]):
            if  (np.equal(bg, image[row]).any()):
                bg_keys.append(row) 
        print(bg_keys)
        

        lenkeys = len(bg_keys)-1
        new_keys = [bg_keys[0], bg_keys[-1]]
        print(new_keys)
        for i in range(1, lenkeys):
            if (bg_keys[i+1] - bg_keys[i]) > check or (bg_keys[i+1] - bg_keys[i]) > checkhor:
                new_keys.append(bg_keys[i])
                new_keys.append(bg_keys[i+1])
    
        
        print("vas",base)
        new_keys = sorted(new_keys)
        print("traile",new_keys)
        segmented_templates = []
        first = new_keys[0]
        last = new_keys[-1]
        bounding_boxes = []
        if len(new_keys) >2 :
            upper_modifier = bordered[first:new_keys[1]]
            predit_uppper(upper_modifier)
            first = new_keys[2]
        core_modifier = bordered[first:base]
        # plt.imshow(core_modifier)
        # plt.show()
        cores = aakar_seg(core_modifier)
        for core in cores:
            predit_core(core)
            
        lower_modifier = bordered[base:last]
        if lower_modifier.shape[0]>=min_seg and lower_modifier.shape[1]>=min_seg: 
            predit_lower(lower_modifier)


        return(segmented_templates, bounding_boxes)
    except:
        return [bordered, (0, bordered.shape[1])]
    

def identify_lower_baseline(image):
    height, width = image.shape
    print("height", height,width)
    transitions = []
    
    for row in range(height):
        sum = 0 
        for i in range(width-1):
            if image[row][i] != image[row][i+1]:
                sum +=1
        transitions.append(sum)
    mean = np.mean(transitions)
    
    print('transitions',transitions)
    print("mewn",mean)
    for row in range(height-1, height//2, -1):
        print('herr')
        print(transitions[row])
        if transitions[row] >= mean:
            base = row+10
            print('vase',base)
            return base

    return height
    


def aakar_seg(bordered,thresh = 255, scheck=0.15):
    try:
        shape = bordered.shape
        check = int(scheck * shape[1])
        image = bordered[:]
        image = image[check:].T
        shape = image.shape

        #find the background color for empty column
        bg = np.repeat(255 - thresh, shape[1])
        bg_keys = []
        for row in range(1, shape[0]):
            if  (np.equal(bg, image[row]).all()):
                bg_keys.append(row)            

        lenkeys = len(bg_keys)-1
        new_keys = [bg_keys[1], bg_keys[-1]]
        #print(lenkeys)
        for i in range(1, lenkeys):
            if (bg_keys[i+1] - bg_keys[i]) > check:
                new_keys.append(bg_keys[i])
                #print(i)

        new_keys = sorted(new_keys)
        #print(new_keys)
        segmented_templates = []
        first = 0
        for key in new_keys[1:]:
            segment = bordered.T[first:key]
            if segment.shape[0]>=check and segment.shape[1]>=check:
                segmented_templates.append(segment.T)

            first = key
        last_segment = bordered.T[new_keys[-1]:]
        if last_segment.shape[0]>=check and last_segment.shape[1]>=check:
            segmented_templates.append(last_segment.T)
        
        #check if each segment shape is enough to do recognition
        

        return(segmented_templates)
    except:
        return [bordered]
    

def predit_uppper(image):

    plt.imshow(image, cmap='gray')
    plt.show()
    inverted_image = cv2.bitwise_not(image)
    rgb_image = cv2.cvtColor(inverted_image, cv2.COLOR_GRAY2RGB)
    resized_image = cv2.resize(rgb_image, (32, 32))

    # Preprocess the image to match the input requirements of VGG16
    preprocessed_image = resized_image.astype('float32')
    preprocessed_image /= 255.0
    x = preprocessed_image.reshape(1, 32, 32, 3)
    y = np.argmax(upperModel.predict(x))
    print("Upper Modifiers",y)


def predit_core(image):
    plt.imshow(image, cmap='gray')
    plt.show()
    print("------------------------")
    thresh = cv2.resize(image, (32, 32), interpolation = cv2.INTER_AREA)
    plt.imshow(thresh)
    plt.show()
    x = np.array([thresh]).reshape(-1, 32, 32, 1) / 255.0
    y = np.argmax(coreModel.predict(x))
    print("core Modifiers",y)

def predit_lower(image):
    plt.imshow(image, cmap='gray')
    plt.show()
    inverted_image = cv2.bitwise_not(image)
    rgb_image = cv2.cvtColor(inverted_image, cv2.COLOR_GRAY2RGB)
    resized_image = cv2.resize(rgb_image, (32, 32))

    # Preprocess the image to match the input requirements of VGG16
    preprocessed_image = resized_image.astype('float32')
    preprocessed_image /= 255.0
    x = preprocessed_image.reshape(1, 32, 32, 3)
    y = np.argmax(lowerModel.predict(x))
    print("Upper Modifiers",y)


