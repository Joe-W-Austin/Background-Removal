import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, misc
import os
from mrcnn import utils
from mrcnn.config import Config
from mrcnn import model as modellib
from skimage import feature
import imageio
from sklearn.cluster import k_means


def computeYCbCr_cv(bgr):
    m = np.array(
        ([0.5, -0.418688, -0.081312],
         [-0.168736, -0.331264, 0.5],
         [0.299, 0.587, 0.114]))

    v = np.dot(m, bgr)
    v[1] = v[1] + 128
    v[2] = v[2] + 128

    return v
    # m = np.array(
    #     ([1, 0.5, -0.418688, -0.081312],
    #      [1, -0.168736, -0.331264, 0.5],
    #      [0, 0.299, 0.587,	0.114]))
    #
    # bgr = np.insert(bgr, 0, 128)
    #
    # return np.dot(m, bgr)


def computeYCbCr(rgb):
    m = np.array(
        ([0, 0.299, 0.587, 0.114],
         [1, -0.168736, -0.331264, 0.5],
         [1, 0.5, -0.418688, -0.081312]))

    rgb = np.insert(rgb, 0, 128)

    print(m.shape)
    print(rgb.shape)
    print(rgb)

    v = np.dot(m, rgb)
    # v[1] = 128 - v[1]
    # v[2] = 128 + v[2]

    print(v.shape)
    print(v)


def play_y_cb_cr(file):
    cap = cv2.VideoCapture(file)

    while (1):
        ret, frame = cap.read()
        # Frame is in BGR format

        # rows = frame.shape[0]
        # cols = frame.shape[1]
        # for r in range(rows):
        #     for c in range(cols):
        #         frame[r,c] = computeYCbCr_cv(frame[r,c])

        cv2.imshow('YCbCr', frame)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('yCbCr.png', frame)

    cap.release()
    cv2.destroyAllWindows()


def apply_sobel(image):
    # img = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=5)
    # img = ndimage.sobel(image)
    image = image.astype('int32')
    dx = ndimage.sobel(image, 0)  # horizontal derivative
    dy = ndimage.sobel(image, 1)  # vertical derivative
    mag = np.hypot(dx, dy)  # magnitude
    mag *= 255.0 / np.max(mag)  # normalize (Q&D)
    return mag
    # return img


def apply_laplacian(image):
    # img = cv2.Laplacian(image, cv2.CV_32F)
    img = ndimage.laplace(image)
    print(img.min())
    print(img.max())
    return img


def test_opencv():
    img = cv2.imread('test2.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = cv2.GaussianBlur(img, (3, 3), 0)
    img = apply_sobel(img)
    # cv2.imshow('image', img)
    cv2.imshow('image', cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=5))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def apply_gaussian(img):
    return ndimage.gaussian_filter(img, 1)


def test_scipy(img):
    threshold = 120
    fig = plt.figure()
    plt.gray()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    result = ndimage.gaussian_filter(img, 1)
    result = ndimage.sobel(result, axis=-1)
    result[result >= threshold] = 255
    result[result < threshold] = 0
    ax1.imshow(img)
    ax2.imshow(result)
    plt.show()


def test_orb(img):
    # img = cv2.imread('simple.jpg', 0)
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except:
        print("Error converting")
        # img = cv2(cv2.COLOR)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # plt.imshow(img), plt.show()

    # Initiate STAR detector
    orb = cv2.ORB_create()

    # find the keypoints with ORB
    kp = orb.detect(img, None)

    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)

    count = 0

    # draw only keypoints location,not size and orientation
    img2 = img.copy()

    for marker in kp:
        count = count + 1
        img2 = cv2.drawMarker(img2, tuple(int(i) for i in marker.pt), color=(255, 0, 0))

    print("Found Marker Size: " + str(count))

    plt.imshow(img2), plt.show()


def test_sift(img):
    cv2.xfeatures2d.SIFT_create()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT()
    kp = sift.detect(gray, None)

    img = cv2.drawKeypoints(gray, kp)

    plt.imshow(img), plt.show()


def show_image(img):
    plt.gray()
    plt.imshow(img), plt.show()


def show_mask1(img1, img2):
    threshold = 120
    fig = plt.figure()
    plt.gray()
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    smooth1 = ndimage.gaussian_filter(img1, 1)
    smooth2 = ndimage.gaussian_filter(img2, 1)

    mask = smooth2 - smooth1

    ax1.imshow(img1)
    ax2.imshow(img2)
    ax3.imshow(mask)
    plt.show()


def show_mask2(img1, img2):
    threshold = 15
    fig = plt.figure()
    plt.gray()
    ax1 = fig.add_subplot(221)
    ax3 = fig.add_subplot(222)
    ax2 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    smooth1 = np.dot(img1[..., :3], [0.2989, 0.5870, 0.1140])
    smooth2 = np.dot(img2[..., :3], [0.2989, 0.5870, 0.1140])

    smooth1 = ndimage.gaussian_filter(smooth1, 1)
    smooth2 = ndimage.gaussian_filter(smooth2, 1)

    # mask = smooth2 - smooth1
    mask = np.absolute(np.array(smooth2) - np.array(smooth1))
    # mask[mask >= threshold] = 255
    # mask[mask < threshold] = 0

    # maskFrom = smooth1 - smooth2
    binaryMask = mask.copy()
    binaryMask[binaryMask < threshold] = 0
    binaryMask[binaryMask >= threshold] = 255
    # maskFrom[maskFrom >= threshold] = 255
    # maskFrom[maskFrom < threshold] = 0

    ax1.imshow(img1)
    ax2.imshow(img2)
    ax4.imshow(binaryMask)
    ax3.imshow(mask)
    plt.show()


def mask_video(path):
    cap = cv2.VideoCapture(path)
    blur_size = 1

    ret, frame1 = cap.read()
    # prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    prvs = np.dot(frame1[..., :3], [0.2989, 0.5870, 0.1140])
    prvs = ndimage.gaussian_filter(prvs, blur_size)

    while (1):
        ret, frame2 = cap.read()
        # next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        next = np.dot(frame2[..., :3], [0.2989, 0.5870, 0.1140])
        next = ndimage.gaussian_filter(next, blur_size)

        mask = np.absolute(np.array(next) - np.array(prvs))

        cv2.imshow('frame2', mask)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('opticalfb.png', frame2)
            cv2.imwrite('opticalhsv.png', mask)
        prvs = next

    cap.release()
    cv2.destroyAllWindows()


def optical_flow_dense_video(path):
    cap = cv2.VideoCapture(path)

    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255

    while (1):
        ret, frame2 = cap.read()
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        cv2.imshow('frame2', rgb)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('opticalfb.png', frame2)
            cv2.imwrite('opticalhsv.png', rgb)
        prvs = next

    cap.release()
    cv2.destroyAllWindows()


def optical_flow_dense_real_frame_video(path):
    cap = cv2.VideoCapture(path)

    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    while (1):
        ret, frame2 = cap.read()
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        print(mag.max())
        print(mag.min())
        mean = mag.mean()
        median = np.median(mag)

        avg = (mean + median) / 2.0

        # frame2[np.where(mag >= 0.4)] = np.asarray([0, 0, 0])
        frame2[np.where(mag <= avg)] = np.asarray([0, 0, 0])
        # print(rgb.shape)

        cv2.imshow('frame2', frame2)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('opticalfb.png', frame2)
            # cv2.imwrite('opticalhsv.png', rgb)
        prvs = next

    cap.release()
    cv2.destroyAllWindows()


def get_mask(img1, img2):
    return np.absolute(np.array(img2) - np.array(img1))


def canny_video_cv(path: str, perform_edge: bool = True, showUnderlyingVideo: bool = True):
    cap = cv2.VideoCapture(path)
    kernel_size = 5

    while (1):
        ret, frame = cap.read()
        # Apply a 3 x 3 Gaussian Blur
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)

        if perform_edge:
            mask = cv2.Canny(frame, 100, 200)
            frame[:, :, 0] = np.where(
                mask > 0,
                0,
                gray[:, :, 0]
            )
            frame[:, :, 1] = np.where(
                mask > 0,
                0,
                gray[:, :, 0]
            )
            frame[:, :, 2] = np.where(
                mask > 0,
                255,
                gray[:, :, 0]
            )

        if showUnderlyingVideo:
            cv2.imshow('frame2', frame)
        else:
            cv2.imshow('frame2', mask)

        k = cv2.waitKey(10) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('edgefb.png', frame)

    cap.release()
    cv2.destroyAllWindows()


def video_edge(path):
    cap = cv2.VideoCapture(path)

    while (1):
        ret, frame2 = cap.read()
        next = apply_gaussian(cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY))
        next = apply_laplacian(next)
        # next = apply_sobel(next)

        cv2.imshow('frame2', next)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('edgefb.png', frame2)
            cv2.imwrite('sobelhsv.png', next)

    cap.release()
    cv2.destroyAllWindows()


def optical_flow_dense_video_edge(path):
    cap = cv2.VideoCapture(path)

    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255

    while (1):
        ret, frame2 = cap.read()
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        # rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        rgb = apply_laplacian(mag)

        cv2.imshow('frame2', rgb)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('opticalfb.png', frame2)
            cv2.imwrite('opticalhsv.png', rgb)
        prvs = next

    cap.release()
    cv2.destroyAllWindows()


def optical_flow_dense_video_mask(path):
    cap = cv2.VideoCapture(path)

    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    print(frame1.shape)
    (x, y, z) = frame1.shape
    print(str(x))
    print(str(y))
    prev_mag = np.zeros((x, y))

    while (1):
        ret, frame2 = cap.read()
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # mag[mag>1] = 255

        f = get_mask(prev_mag, mag)
        prev_mag = mag

        # rgb = get_mask(prev_mag, mag)
        # f = get_mask(prev_mag, rgb)
        # prev_mag = rgb

        cv2.imshow('frame2', f)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('opticalfb.png', frame2)
            cv2.imwrite('opticalhsv.png', f)
        prvs = next

    cap.release()
    cv2.destroyAllWindows()


def optical_flow_dense_video_mask_edge(path):
    cap = cv2.VideoCapture(path)

    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    print(frame1.shape)
    (x, y, z) = frame1.shape
    print(str(x))
    print(str(y))
    prev_mag = np.zeros((x, y))

    while (1):
        ret, frame2 = cap.read()
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # mag[mag>1] = 255

        f = get_mask(prev_mag, mag)
        prev_mag = mag

        # rgb = get_mask(prev_mag, mag)
        # f = get_mask(prev_mag, rgb)
        # prev_mag = rgb

        cv2.imshow('frame2', apply_laplacian(f))
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('opticalfb.png', frame2)
            cv2.imwrite('opticalhsv.png', f)
        prvs = next

    cap.release()
    cv2.destroyAllWindows()


def optical_flow_dense_video_with_subtract(path):
    cap = cv2.VideoCapture(path)

    backSub = cv2.createBackgroundSubtractorMOG2()
    ret, frame1 = cap.read()
    # prvs = cv2.cvtColor(backSub.apply(frame1), cv2.COLOR_BGR2GRAY)
    prvs = backSub.apply(frame1)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255

    while (1):
        ret, frame2 = cap.read()
        # next = cv2.cvtColor(backSub.apply(frame2), cv2.COLOR_BGR2GRAY)
        next = backSub.apply(frame2)

        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        cv2.imshow('frame2', rgb)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('opticalfb.png', frame2)
            cv2.imwrite('opticalhsv.png', rgb)
        prvs = next

    cap.release()
    cv2.destroyAllWindows()


def simple_subtract(file):
    capture = cv2.VideoCapture(file)
    backSub = cv2.createBackgroundSubtractorMOG2()
    prevFrame = None

    while True:
        ret, frame = capture.read()
        if frame is None:
            break

        if prevFrame is None:
            prevFrame = frame
            continue

        diffFrame = frame - prevFrame
        prevFrame = frame

        cv2.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
        cv2.putText(frame, str(capture.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        cv2.imshow('FG Mask', diffFrame)

        keyboard = cv2.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break


def background_subtract(file):
    capture = cv2.VideoCapture(file)
    backSub = cv2.createBackgroundSubtractorMOG2()

    while True:
        ret, frame = capture.read()
        if frame is None:
            break

        fgMask = backSub.apply(frame)

        cv2.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
        cv2.putText(frame, str(capture.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        # cv2.imshow('Frame', frame)
        cv2.imshow('FG Mask', fgMask)

        keyboard = cv2.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break


def test_video_rcnn(file, show_edge=False, eliminate_background=True):
    capture = cv2.VideoCapture(file)
    ROOT_DIR = os.getcwd()
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
    print(COCO_MODEL_PATH)
    if not os.path.exists(COCO_MODEL_PATH):
        print("Uh oh")

    # Change the config infermation
    class CocoConfig(Config):
        """Configuration for training on MS COCO.
        Derives from the base Config class and overrides values specific
        to the COCO dataset.
        """
        # Give the configuration a recognizable name
        NAME = "coco"

        # We use a GPU with 12GB memory, which can fit two images.
        # Adjust down if you use a smaller GPU.
        IMAGES_PER_GPU = 1

        # Uncomment to train on 8 GPUs (default is 1)
        GPU_COUNT = 1

        # Number of classes (including background)
        NUM_CLASSES = 1 + 80

    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=CocoConfig())
    model.load_weights(COCO_MODEL_PATH, by_name=True)

    class_names = [
        'person'
    ]

    while True:
        ret, frame = capture.read()
        if frame is None:
            break
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        def apply_mask(image, mask):
            if eliminate_background:
                image[:, :, 0] = np.where(
                    mask == 0,
                    0,
                    image[:, :, 0]
                )
                image[:, :, 1] = np.where(
                    mask == 0,
                    0,
                    image[:, :, 1]
                )
                image[:, :, 2] = np.where(
                    mask == 0,
                    0,
                    image[:, :, 2])
            else:
                image[:, :, 0] = np.where(
                    mask == 0,
                    gray_image[:, :],
                    image[:, :, 0]
                )
                image[:, :, 1] = np.where(
                    mask == 0,
                    gray_image[:, :],
                    image[:, :, 1]
                )
                image[:, :, 2] = np.where(
                    mask == 0,
                    gray_image[:, :],
                    image[:, :, 2])

            return image

        def apply_masks(image, masks):
            print(masks.shape)
            x = image.shape[0]
            y = image.shape[1]
            n = masks.shape[2]

            print("Image Shape: " + str(image.shape))
            print("N: " + str(n))
            merged_mask = np.zeros((x, y), bool)

            for i in range(n):
                mask = masks[:, :, i]
                merged_mask[:, :] = np.where(
                    mask,
                    True,
                    merged_mask[:, :]
                )

            print(merged_mask.max())
            return apply_mask(image, merged_mask)

        def display_instances(image, boxes, masks, ids, names, scores):
            # max_area will save the largest object for all the detection results
            max_area = 0

            # n_instances saves the amount of all objects
            print("Before Masks Shape: " + str(masks.shape))
            n_instances = boxes.shape[0]
            person_masks = []

            if not n_instances:
                print('NO INSTANCES TO DISPLAY')
            else:
                assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]

            for i in range(n_instances):
                if not np.any(boxes[i]):
                    continue

                if ids[i] == 1:
                    mask = masks[:, :, i]
                    person_masks.append(mask)

                # compute the square of each object
                y1, x1, y2, x2 = boxes[i]
                square = (y2 - y1) * (x2 - x1)

                # use label to select person object from all the 80 classes in COCO dataset
                # label = names[ids[i]]
                print("Class: " + str(ids[i]))

                # if square > max_area:
                #   max_area = square
                #  mask = masks[:, :, i]
                #  print("Applying Mask " + str(i))
                #  apply_mask(image, mask, base_image)

                # if label == 'person':
                #   print("Found PERSON!!!!!!")
                # save the largest object in the image as main character
                # other people will be regarded as background
                #  if square > max_area:
                #     max_area = square
                #    mask = masks[:, :, i]
                # else:
                #   continue
                # else:
                #   continue

                # apply mask for the image
            # by mistake you put apply_mask inside for loop or you can write continue in if also
            # image = apply_mask(image, mask)

            take_indicies = np.argwhere(ids == 1).flatten()
            print("Take Indicies: " + str(take_indicies))
            print(take_indicies.shape)
            person_mask_array = masks[:, :, take_indicies]
            print("New Shape: " + str(person_mask_array.shape))
            image = apply_masks(image, person_mask_array)

            return image

        results = model.detect([frame], verbose=0)
        r = results[0]

        try:
            frame = display_instances(
                frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores']
            )
            if show_edge:
                frame = apply_gaussian(frame)
                frame = apply_laplacian(frame)

        except:
            print("No people?")

        cv2.imshow('save_image', frame)
        keyboard = cv2.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break


def test_mask(path):
    image = cv2.imread(path)
    cv2.imshow("original image", image)

    print("Read original image successfully")
    print(image.shape)
    print("Press ESC to exit or press s to save and exit")

    # Wait
    k = cv2.waitKey(0)
    if k == 27:
        cv2.destroyAllWindows()
    elif k == ord('d'):
        cv2.imwrite('original_image.jpg', image)
        cv2.destroyAllWindows()

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray_image', gray_image)

    print("Change gray image successfully! The gray image shape is:")
    print(gray_image.shape)
    print("Press ESC to exit or press s to save and exit.")

    # Wait for keys to exit or save
    k = cv2.waitKey(0)
    if k == 27:
        cv2.destroyAllWindows()
    elif k == ord('s'):
        cv2.imwrite('gray_image.jpg', image)
        cv2.destroyAllWindows()

    ROOT_DIR = os.getcwd()
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
    print(COCO_MODEL_PATH)
    if not os.path.exists(COCO_MODEL_PATH):
        print("Uh oh")

    # Change the config infermation
    class CocoConfig(Config):
        """Configuration for training on MS COCO.
        Derives from the base Config class and overrides values specific
        to the COCO dataset.
        """
        # Give the configuration a recognizable name
        NAME = "coco"

        # We use a GPU with 12GB memory, which can fit two images.
        # Adjust down if you use a smaller GPU.
        IMAGES_PER_GPU = 1

        # Uncomment to train on 8 GPUs (default is 1)
        GPU_COUNT = 1

        # Number of classes (including background)
        NUM_CLASSES = 1 + 80

    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=CocoConfig())
    model.load_weights(COCO_MODEL_PATH, by_name=True)

    class_names = [
        'person'
    ]

    def apply_mask(image, mask):
        image[:, :, 0] = np.where(
            mask == 0,
            gray_image[:, :],
            image[:, :, 0]
        )
        image[:, :, 1] = np.where(
            mask == 0,
            gray_image[:, :],
            image[:, :, 1]
        )
        image[:, :, 2] = np.where(
            mask == 0,
            gray_image[:, :],
            image[:, :, 2]
        )

        return image

    def apply_masks(image, masks):
        print(masks.shape)
        x = image.shape[0]
        y = image.shape[1]
        n = masks.shape[2]

        print("Image Shape: " + str(image.shape))
        print("N: " + str(n))
        merged_mask = np.zeros((x, y), bool)

        for i in range(n):
            mask = masks[:, :, i]
            merged_mask[:, :] = np.where(
                mask,
                True,
                merged_mask[:, :]
            )

        print(merged_mask.max())
        return apply_mask(image, merged_mask)

    def display_instances(image, boxes, masks, ids, names, scores):
        # max_area will save the largest object for all the detection results
        max_area = 0

        # n_instances saves the amount of all objects
        print("Before Masks Shape: " + str(masks.shape))
        n_instances = boxes.shape[0]
        person_masks = []

        if not n_instances:
            print('NO INSTANCES TO DISPLAY')
        else:
            assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]

        for i in range(n_instances):
            if not np.any(boxes[i]):
                continue

            if ids[i] == 1:
                mask = masks[:, :, i]
                person_masks.append(mask)

            # compute the square of each object
            y1, x1, y2, x2 = boxes[i]
            square = (y2 - y1) * (x2 - x1)

            # use label to select person object from all the 80 classes in COCO dataset
            # label = names[ids[i]]
            print("Class: " + str(ids[i]))

            # if square > max_area:
            #   max_area = square
            #  mask = masks[:, :, i]
            #  print("Applying Mask " + str(i))
            #  apply_mask(image, mask, base_image)

            # if label == 'person':
            #   print("Found PERSON!!!!!!")
            # save the largest object in the image as main character
            # other people will be regarded as background
            #  if square > max_area:
            #     max_area = square
            #    mask = masks[:, :, i]
            # else:
            #   continue
            # else:
            #   continue

            # apply mask for the image
        # by mistake you put apply_mask inside for loop or you can write continue in if also
        # image = apply_mask(image, mask)

        take_indicies = np.argwhere(ids == 1).flatten()
        print("Take Indicies: " + str(take_indicies))
        print(take_indicies.shape)
        person_mask_array = masks[:, :, take_indicies]
        print("New Shape: " + str(person_mask_array.shape))
        image = apply_masks(image, person_mask_array)

        return image

    results = model.detect([image], verbose=0)
    r = results[0]
    frame = display_instances(
        image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores']
    )
    cv2.imshow('save_image', frame)

    # Wait for keys to exit or save
    k = cv2.waitKey(0)
    if k == 27:
        cv2.destroyAllWindows()
    elif k == ord('s'):
        cv2.imwrite('save_image.jpg', image)
        cv2.destroyAllWindows()

    frame = apply_gaussian(frame)
    frame = apply_laplacian(frame)
    cv2.imshow('edge', frame)

    # Wait for keys to exit or save
    k = cv2.waitKey(0)
    if k == 27:
        cv2.destroyAllWindows()
    elif k == ord('s'):
        cv2.imwrite('save_image.jpg', image)
        cv2.destroyAllWindows()


def covariance_test():
    m = np.array(
        [[0, 1, 2, 3, 4],
         [1, 0, 2, 3, 4],
         [5, 6, 7, 8, 9],
         [10, 11, 12, 13, 14],
         [15, 16, 17, 18, 19]])

    m1 = m[1:3, 1:3]
    print(m1)
    cov = np.cov(m1)
    print(cov)

    m2 = np.array(
        [[0, 1, 2],
         [2, 1, 0]])

    print(m2)
    c = np.cov(m2)
    print(c)


if __name__ == '__main__':
    # img = apply_sobel(apply_gaussian(cv2.imread('o1.png', 0)))
    # test_orb(img)

    # test_orb(cv2.imread('m0.jpg', 0))
    # test_sift(cv2.imread('m0.jpg'))

    # img1 = imageio.imread('ss/ss0092.png')
    # img2 = imageio.imread('ss/ss0093.png')
    # img1 = apply_laplacian(img1)
    # img2 = apply_laplacian(img2)
    # show_mask2(img1, img2)

    # -------
    # img = imageio.imread("ng0001.png")
    # gray = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])

    # l = apply_laplacian(gray)
    # l[l <= 0] = 0
    # l[l > 1] = 255

    # print(l.min())
    # print(l.max())
    # show_image(l)
    # ----------

    # file = "ss.mp4"
    # file = "ot.mp4"
    file = "data/video/mb.mp4"

    # optical_flow_dense_real_frame_video(file)
    # background_subtract(file)
    simple_subtract(file)
    # optical_flow_dense_video_mask_edge(file)
    # optical_flow_dense_video_mask(file)
    # optical_flow_dense_video_edge(file)
    # optical_flow_dense_video(file)
    # mask_video(file)
    # optical_flow_dense_video_with_subtract(file)
    # video_edge(file)

    # file = 'bs.jpeg'
    # file = 'ng0010.png'
    # file = "ng0498.png"
    # test_mask(file)

    # file = "data/video/colbert.mp4"
    # test_video_rcnn(file, False, True)
    # computeYCbCr(np.array([255., 0., 255.]))
    # play_y_cb_cr(file)

    # file = "aoc.mp4"
    # canny_video_cv(file, True, False)

    # covariance_test()

# test_opencv()

# img = imageio.imread('test.jpg')
# gray = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
# test_scipy(gray)
