import cv2
import numpy as np

cone_hsv_lb = np.array([127, 98, 131])
cone_hsv_ub = np.array([180, 255, 255])

def convex_hull_pointing_up(ch):

    points_above_center, points_below_center = [], []
    
    x, y, w, h = cv2.boundingRect(ch)
    aspect_ratio = w / h

    if aspect_ratio < 0.8:
        vertical_center = y + h / 2

        for point in ch:
            if point[0][1] < vertical_center:
                points_above_center.append(point)
            elif point[0][1] >= vertical_center:
                points_below_center.append(point)

        left_x = points_below_center[0][0][0]
        right_x = points_below_center[0][0][0]
        for point in points_below_center:
            if point[0][0] < left_x:
                left_x = point[0][0]
            if point[0][0] > right_x:
                right_x = point[0][0]

        for point in points_above_center:
            if (point[0][0] < left_x) or (point[0][0] > right_x):
                return False
    else:
        return False
        
    return True

def detect_cones(frame, hsv_lb, hsv_ub):
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    img_thresh = cv2.inRange(frame_hsv, hsv_lb, hsv_ub)
    # cv2.imshow('threshold', img_thresh)

    kernel = np.ones((5, 5))
    img_thresh_opened = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel)
    # cv2.imshow('thresh opened', img_thresh_opened)

    img_thresh_blurred = cv2.medianBlur(img_thresh_opened, 5)

    img_edges = cv2.Canny(img_thresh_blurred, 80, 160)

    contours, _ = cv2.findContours(np.array(img_edges), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # img_contours = np.zeros_like(img_edges)
    # cv2.drawContours(img_contours, contours, -1, (255,255,255), 2)
    # cv2.imshow('contours', img_contours)

    approx_contours = []
    for c in contours:
        approx = cv2.approxPolyDP(c, 10, closed = True)
        approx_contours.append(approx)
    # img_approx_contours = np.zeros_like(img_edges)
    # cv2.drawContours(img_approx_contours, approx_contours, -1, (255,255,255), 1)

    all_convex_hulls = []
    for ac in approx_contours:
        all_convex_hulls.append(cv2.convexHull(ac))
    # img_all_convex_hulls = np.zeros_like(img_edges)
    # cv2.drawContours(img_all_convex_hulls, all_convex_hulls, -1, (255,255,255), 2)

    convex_hulls_3to10 = []
    for ch in all_convex_hulls:
        if 3 <= len(ch) <= 10:
            convex_hulls_3to10.append(cv2.convexHull(ch))
    # img_convex_hulls_3to10 = np.zeros_like(img_edges)
    # cv2.drawContours(img_convex_hulls_3to10, convex_hulls_3to10, -1, (255,255,255), 2)

    cones = []
    bounding_rects = []
    for ch in convex_hulls_3to10:
        if convex_hull_pointing_up(ch):
            cones.append(ch)
            rect = cv2.boundingRect(ch)
            bounding_rects.append(rect)
    # img_cones = np.zeros_like(img_edges)
    # cv2.drawContours(img_cones, cones, -1, (255,255,255), 2)
    # cv2.drawContours(img_cones, bounding_rects, -1, (1,255,1), 2)
    # cv2.imshow('find cones', img_cones)

    # img_res = frame
    # cv2.drawContours(img_res, cones, -1, (255,255,255), 2)

    # for rect in bounding_rects:
    #     cv2.rectangle(img_res, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (1, 255, 1), 3)
    # cv2.imshow('cone detection result', img_res)

    return bounding_rects

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame. Exiting...")
        break
    # cv2.imshow('frame', frame)

    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    img_thresh = cv2.inRange(frame_hsv, cone_hsv_lb, cone_hsv_ub)
    # cv2.imshow('threshold', img_thresh)

    kernel = np.ones((5, 5))
    img_thresh_opened = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel)
    # cv2.imshow('thresh opened', img_thresh_opened)

    img_thresh_blurred = cv2.medianBlur(img_thresh_opened, 5)

    img_edges = cv2.Canny(img_thresh_blurred, 80, 160)

    contours, _ = cv2.findContours(np.array(img_edges), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_contours = np.zeros_like(img_edges)
    cv2.drawContours(img_contours, contours, -1, (255,255,255), 2)
    # cv2.imshow('contours', img_contours)

    approx_contours = []
    for c in contours:
        approx = cv2.approxPolyDP(c, 10, closed = True)
        approx_contours.append(approx)
    img_approx_contours = np.zeros_like(img_edges)
    cv2.drawContours(img_approx_contours, approx_contours, -1, (255,255,255), 1)

    all_convex_hulls = []
    for ac in approx_contours:
        all_convex_hulls.append(cv2.convexHull(ac))
    img_all_convex_hulls = np.zeros_like(img_edges)
    cv2.drawContours(img_all_convex_hulls, all_convex_hulls, -1, (255,255,255), 2)

    convex_hulls_3to10 = []
    for ch in all_convex_hulls:
        if 3 <= len(ch) <= 10:
            convex_hulls_3to10.append(cv2.convexHull(ch))
    img_convex_hulls_3to10 = np.zeros_like(img_edges)
    cv2.drawContours(img_convex_hulls_3to10, convex_hulls_3to10, -1, (255,255,255), 2)

    cones = []
    bounding_rects = []
    for ch in convex_hulls_3to10:
        if convex_hull_pointing_up(ch):
            cones.append(ch)
            rect = cv2.boundingRect(ch)
            bounding_rects.append(rect)
    img_cones = np.zeros_like(img_edges)
    cv2.drawContours(img_cones, cones, -1, (255,255,255), 2)
    # cv2.drawContours(img_cones, bounding_rects, -1, (1,255,1), 2)
    # cv2.imshow('find cones', img_cones)

    img_res = frame
    cv2.drawContours(img_res, cones, -1, (255,255,255), 2)

    for rect in bounding_rects:
        cv2.rectangle(img_res, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (1, 255, 1), 3)
    # cv2.imshow('cone detection result', img_res)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

