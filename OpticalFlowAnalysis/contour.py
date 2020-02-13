import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture('test3.mp4')

frame_width = int( cap.get(cv2.CAP_PROP_FRAME_WIDTH))

frame_height =int( cap.get( cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'MPEG')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
color = np.random.randint(0,255,(100,3))
ret, frame1 = cap.read()
ret, frame2 = cap.read()
print(frame1.shape)

# Create mask
mask = np.zeros_like(frame1)
# Sets image saturation to maximum
mask[..., 1] = 255


# def get_points_changed(arr1, arr2):
#     diff = arr1 - arr2
#     indices = []
#     for idx, each in enumerate(diff):
#         if each.any() > 0:
#             indices.append(idx)
#
#     return indices, diff

# array_of_difference = np.zeros((1040, 2048))
while cap.isOpened():
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)

        if cv2.contourArea(contour) < 900:
            continue
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
        #cv2.putText(frame1, "Status: {}".format('Movement'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
        #1, (0, 0, 255), 3)

    cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2)

    image = cv2.resize(frame1, (1280,720))
    out.write(image)
    #cv2.imshow("feed", frame1)
    frame1 = frame2
    ret, frame2 = cap.read()
    prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, pyr_scale=0.5, levels=5, winsize=11, iterations=5,
                                        poly_n=5, poly_sigma=1.1, flags=0)
    # Compute the magnitude and angle of the 2D vectors

    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # changed_points, array_of_difference = get_points_changed(magnitude, array_of_difference)
    print("magnitude in 2D array", magnitude)
    print("angle in 2D array", angle)
    # print("change points", changed_points)
    print(f"Len: {len(magnitude)}, type: {type(magnitude)}, shape: {magnitude.shape}");
    mask[..., 0] = angle * 180 / np.pi
    # Set image value according to the optical flow magnitude (normalized)
    mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    # Convert HSV to RGB (BGR) color representation
    rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)

    # converting 2D array into 1D array and plotting histogram
    mag1D = np.ravel(magnitude)
    ang1D=np.ravel(angle)
    # print("size of angle 1 d array-------",ang1D.size)
    print("magnitude in one D array------", mag1D)
    print("angles in one D array------", ang1D)
    print("max value in Angle Array",np.amax(ang1D))
    print("min value in Angle Array", np.amin(ang1D))
    print("max value in Magnitude Array", np.amax(mag1D))
    print("min value in Magnitude Array", np.amin(mag1D))
    # plt.hist(ang1D)
    # plt.show()
    # # plt.pause(1)
    # plt.close()
    #
    bins_number = 50  # the [0, 360) interval will be subdivided into this
    # number of equal bins
    bins = np.linspace(0.0, 2 * np.pi, bins_number + 1)
    print("bins created---------", bins)
    angles = 2 * np.pi * mag1D
    print("angles in degree---------", angles)
    n, _, _ = plt.hist(angles, bins)
    plt.clf()
    width = 2 * np.pi / bins_number
    ax = plt.subplot(1, 1, 1, projection='polar')
    bars = ax.bar(bins[:bins_number], n, width=width, bottom=0.0)
    for  bar in  bars:

        bar.set_alpha(0.5)
    plt.show()
    plt.pause(1)
    plt.close()


    # N = 20
    # bins_number = 8
    # bins = np.linspace(0.0, 2 * np.pi, bins_number, endpoint=False)
    # angles= 2 * np.pi *ang1D
    # width = 2 * np.pi / 8
    # # n, _, _ = plt.hist(angle, bins)
    # # plt.clf()
    # ax = plt.subplot(111, projection='polar')
    # bars = ax.bar(bins, angles, width=width, bottom=0.0)
    #
    # # Use custom colors and opacity
    # # for r, bar in zip(angles, bars):
    # #
    # #     bar.set_alpha(0.5)
    #
    # plt.show()
    # plt.pause(1)
    # plt.close()


    def draw_flow(img, flow, step=8):
        h, w = img.shape[:2]
        y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
        fx, fy = flow[y, x].T
        lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines + 0.5)
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.polylines(vis, lines, 0, (0, 255, 0))
        for (x1, y1), (x2, y2) in lines:
            cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
        return vis

    dense_flow = cv2.addWeighted(frame1, 1, rgb, 2, 0)
    cv2.imshow("Dense optical flow", draw_flow(next_gray, flow))
    # cv2.imshow("Dense optical flow", dense_flow)
    out.write(dense_flow)

    if cv2.waitKey(40) == 27:
        break

cv2.destroyAllWindows()
cap.release()
out.release()
