import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

def Sorting_Key(rect):
    global Lines, Size

    x, y, w, h = rect

    cx = x + int(w / 2)
    cy = y + int(h / 2)

    for i, (upper, lower) in enumerate(Lines):
        if not any([all([upper > y + h, lower > y + h]), all([upper < y, lower < y])]):
            return cx + ((i + 1) * Size)


def Split(Image):
    global Lines, Size

    gray = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    for i in range(morph.shape[0]):
        for j in range(morph.shape[1]):
            if not morph[i][j]:
                morph[i][j] = 1

    div = gray / morph
    gray = np.array(cv2.normalize(div, div, 0, 255, cv2.NORM_MINMAX), np.uint8)

    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel, iterations=1)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    i = 0
    contours = list(contours)
    Length = len(contours)
    while i < Length:
        x, y, w, h = cv2.boundingRect(contours[i])

        if w * h <= 100:
            del contours[i]
            i -= 1
            Length -= 1
        i += 1

    h_proj = np.sum(thresh, axis=1)

    upper = None
    lower = None
    Lines = []
    for i in range(h_proj.shape[0]):
        proj = h_proj[i]

        if proj != 0 and upper == None:
            upper = i
        elif proj == 0 and upper != None and lower == None:
            lower = i
            if lower - upper >= 30:
                Lines.append([upper, lower])
            upper = None
            lower = None

    if upper:
        Lines.append([upper, h_proj.shape[0] - 1])

    Size = thresh.shape[1]

    bounding_rects = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x -= 20
        y -=20
        w+=20
        h+=20

        for upper, lower in Lines:
            if not any([all([upper > y + h, lower > y + h]), all([upper < y, lower < y])]):
                bounding_rects.append([x, y, w, h])

    i = 0
    Length = len(bounding_rects)
    while i < Length:
        x, y, w, h = bounding_rects[i]
        j = 0

        while j < Length:
            distancex = abs(
                bounding_rects[j][0] - (bounding_rects[i][0] + bounding_rects[i][2]))
            distancey = abs(
                bounding_rects[j][1] - (bounding_rects[i][1] + bounding_rects[i][3]))

            threshx = max(abs(bounding_rects[j][0] - (bounding_rects[i][0] + bounding_rects[i][2])),
                          abs(bounding_rects[j][0] - bounding_rects[i][0]),
                          abs((
                              bounding_rects[j][0] + bounding_rects[j][2]) - bounding_rects[i][0]),
                          abs((bounding_rects[j][0] + bounding_rects[j][2]) - (bounding_rects[i][0] + bounding_rects[i][2])))

            threshy = max(abs(bounding_rects[j][1] - (bounding_rects[i][1] + bounding_rects[i][3])),
                          abs(bounding_rects[j][1] - bounding_rects[i][1]),
                          abs((
                              bounding_rects[j][1] + bounding_rects[j][3]) - bounding_rects[i][1]),
                          abs((bounding_rects[j][1] + bounding_rects[j][3]) - (bounding_rects[i][1] + bounding_rects[i][3])))

            if i != j and any([all([not any([all([bounding_rects[j][1] > y + h, bounding_rects[j][1] + bounding_rects[j][3] > y + h]), all([bounding_rects[j][1] < y, bounding_rects[j][1] + bounding_rects[j][3] < y])]),
                                   not any([all([bounding_rects[j][0] > x + w, bounding_rects[j][0] + bounding_rects[j][2] > x + w]), all([bounding_rects[j][0] < x, bounding_rects[j][0] + bounding_rects[j][2] < x])])]),
                              all([distancex <= 10, bounding_rects[i][3] + bounding_rects[j][3] + 10 >= threshy]), all([bounding_rects[i][2] + bounding_rects[j][2] + 10 >= threshx, distancey <= 10])]):

                x = min(bounding_rects[i][0], bounding_rects[j][0])
                w = max(bounding_rects[i][0] + bounding_rects[i][2],
                        bounding_rects[j][0] + bounding_rects[j][2]) - x
                y = min(bounding_rects[i][1], bounding_rects[j][1])
                h = max(bounding_rects[i][1] + bounding_rects[i][3],
                        bounding_rects[j][1] + bounding_rects[j][3]) - y

                bounding_rects[i] = [x, y, w, h]
                del bounding_rects[j]
                i = -1
                Length -= 1
                break

            j += 1
        i += 1

    bounding_rects.sort(key=Sorting_Key)

    Words = []
    for x, y, w, h in bounding_rects:
        crop = Image[y:y + h, x:x + w]

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # plt.imshow(gray)
        # plt.show()
        # cv2.imshow('gerya',gray)

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
        if(crop.shape[1]>100):
            rotated = Split_Image_Words(thresh,crop)
        else:
            rotated = crop

        Words.append(rotated.copy())

    return Words




def Split_Image_Words(img,crop):
    P = []
    PX = set()
    PY = set()
    x = 100
    shape = img.shape
    print(shape)
    for x in range(0, shape[1],int(shape[1]/4)):
        for y in range(shape[0]):
            if img[y][x] == 255 and y<shape[0]/1.5:
                P.append((x,y))
                PX.add(x)
                PY.add(y)
                break

# Determine eligibility of each pixel in P

    print(P)
    T = set()


    for i in range(len(P)-2):
        p1 = P[i]
        p2 = P[i+1]
        p3 = P[i+2]



        angle = math.degrees(math.atan2(p3[1]-p2[1], p3[0]-p2[0]) - math.atan2(p1[1]-p2[1], p1[0]-p2[0]))

        if(angle<0):
            angle += 360

        if angle > 200 or angle < 165:
            if p1[1] - p2[1] > p2[1] - p3[1]:
                T.add(p2)
            elif p1[1] - p2[1] <= p2[1] -p3[1]:
                T.add(p1)


    P = set(P)
    A = P - T
    print(A)
    points = np.array(list(A))
    print(points)
    slope, intercept = np.polyfit(points[:,0], points[:,1], 1)

    # Compute the angle with the x-axis
    angle = np.arctan(slope) * 180 / np.pi
    print(shape[0])
    print("Angle with the x-axis:", angle, "degrees")
    if angle < -45:
        angle = -(90 + angle)  
    elif shape[0] < 70*3 and abs(angle) > 10:
        print('here')
        angle = 0 
 
    # otherwise, just take the inverse of the angle to make it positive
    elif abs(angle) > 15:
        angle = angle
    else:
        # angle = angle/1.5
        # angle = angle/1.15
        angle = angle

    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_img = cv2.warpAffine(crop, M, (w, h),
                                
    flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    plt.subplot(211)
    plt.imshow(crop)
    plt.subplot(212)
    plt.imshow(rotated_img)
    plt.show()
    return rotated_img

