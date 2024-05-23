import cv2
import numpy as np

# ----------------------------------------------------------
#                   add gaussian noise
# ----------------------------------------------------------


def _add_gaussian_noise(image_in, noise_sigma):
    temp_image = np.float64(np.copy(image_in))
    h, w, _ = temp_image.shape
    # 标准正态分布*noise_sigma
    noise = np.random.randn(h, w) * noise_sigma

    noisy_image = np.zeros(temp_image.shape, np.float64)
    if len(temp_image.shape) == 2:
        noisy_image = temp_image + noise
    else:
        noisy_image[:, :, 0] = temp_image[:, :, 0] + noise
        noisy_image[:, :, 1] = temp_image[:, :, 1] + noise
        noisy_image[:, :, 2] = temp_image[:, :, 2] + noise

    return noisy_image.clip(0, 255)


def _noise(image, bboxes):
    noise_sigma = 8 if np.random.randint(0, 2) else 6
    image = _add_gaussian_noise(image, noise_sigma)
    return image, bboxes


def _test_noise():
    image = cv2.imread('/data_ssd2/hzh/paperworks/dataset/screenshots/00036.png')
    # image = torch.from_numpy(image.transpose(2, 0, 1)).float()
    boxes = np.array([[100, 100, 1820, 980]])
    image, boxes = _noise(image, boxes)
    image = image.astype(np.uint8)
    cv2.rectangle(image, (int(boxes[0][0]), int(boxes[0][1])), (int(boxes[0][2]), int(boxes[0][3])), (0, 0, 255), 2)
    cv2.imshow("", image)
    # cv2.imwrite('/data_ssd2/hzh/paperworks/dataset/screenshots/00036_test.png', image, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    cv2.waitKey(0)


# ----------------------------------------------------------
#                   random flip
# ----------------------------------------------------------

def _mirror(image, bboxes):
    img_center = np.array(image.shape[:2])[::-1] / 2
    img_center = np.hstack((img_center, img_center))

    image = image[:, ::-1, :]
    bboxes[:, [0, 2]] += 2 * (img_center[[0, 2]] - bboxes[:, [0, 2]])

    box_w = abs(bboxes[:, 0] - bboxes[:, 2])

    bboxes[:, 0] -= box_w
    bboxes[:, 2] += box_w

    return image, bboxes


def _test_mirror():
    image = cv2.imread('/data_ssd2/hzh/paperworks/dataset/screenshots/00036.png')
    boxes = np.array([[100, 100, 960, 540]]).astype(np.float)
    cv2.rectangle(image, (int(boxes[0][0]), int(boxes[0][1])), (int(boxes[0][2]), int(boxes[0][3])), (255, 0, 0), 2)
    cv2.imshow("", image)
    cv2.waitKey(0)
    image, boxes = _mirror(image, boxes)
    image = image.astype(np.uint8)
    cv2.rectangle(image, (int(boxes[0][0]), int(boxes[0][1])), (int(boxes[0][2]), int(boxes[0][3])), (0, 0, 255), 2)
    cv2.imshow("", image)
    cv2.waitKey(0)

# ----------------------------------------------------------
#                   rotate image and bboxes
# ----------------------------------------------------------


def get_corners(bboxes):

    width = (bboxes[:, 2] - bboxes[:, 0]).reshape(-1, 1)
    height = (bboxes[:, 3] - bboxes[:, 1]).reshape(-1, 1)

    x1 = bboxes[:, 0].reshape(-1, 1)
    y1 = bboxes[:, 1].reshape(-1, 1)

    x2 = x1 + width
    y2 = y1

    x3 = x1
    y3 = y1 + height

    x4 = bboxes[:, 2].reshape(-1, 1)
    y4 = bboxes[:, 3].reshape(-1, 1)

    corners = np.hstack((x1, y1, x2, y2, x3, y3, x4, y4))

    return corners


def rotate_im(image, angle):
    # grab the dimensions of the image and then determine the
    # centre
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    image = cv2.warpAffine(image, M, (nW, nH))

    #    image = cv2.resize(image, (w,h))
    return image


def rotate_box(corners, angle, cx, cy, h, w):

    corners = corners.reshape(-1, 2)
    corners = np.hstack((corners, np.ones((corners.shape[0], 1), dtype=type(corners[0][0]))))

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy
    # Prepare the vector to be transformed
    calculated = np.dot(M, corners.T).T

    calculated = calculated.reshape(-1, 8)

    return calculated


def get_enclosing_box(corners):

    x_ = corners[:, [0, 2, 4, 6]]
    y_ = corners[:, [1, 3, 5, 7]]

    xmin = np.min(x_, 1).reshape(-1, 1)
    ymin = np.min(y_, 1).reshape(-1, 1)
    xmax = np.max(x_, 1).reshape(-1, 1)
    ymax = np.max(y_, 1).reshape(-1, 1)

    final = np.hstack((xmin, ymin, xmax, ymax, corners[:, 8:]))

    return final


def _rotate(image, bboxes):
    """
    Args:
        img (PIL Image): Image to be flipped.
    Returns:
        PIL Image: Randomly flipped image.

    """

    # angle = 1 if np.random.randint(0, 2) else -1.2
    angle = 3
    # clockwise = 1 if np.random.randint(0, 2) else -1
    # angle = clockwise * np.random.rand() * 1.2

    w, h = image.shape[1], image.shape[0]
    cx, cy = w // 2, h // 2

    corners = get_corners(bboxes)

    corners = np.hstack((corners, bboxes[:, 4:]))

    image = rotate_im(image, angle)

    corners[:, :8] = rotate_box(corners[:, :8], angle, cx, cy, h, w)

    new_bbox = get_enclosing_box(corners)

    scale_factor_x = image.shape[1] / w

    scale_factor_y = image.shape[0] / h

    image = cv2.resize(image, (w, h))

    new_bbox[:, :4] /= [scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y]

    bboxes = new_bbox

    return image, bboxes


def _test_rotate():
    image = cv2.imread('/data_ssd2/hzh/paperworks/dataset/screenshots/00036.png')
    boxes = np.array([[100, 100, 1820, 980]]).astype(np.float)
    cv2.rectangle(image, (int(boxes[0][0]), int(boxes[0][1])), (int(boxes[0][2]), int(boxes[0][3])), (255, 0, 0), 2)
    image, boxes = _rotate(image, boxes)
    image = image.astype(np.uint8)
    cv2.rectangle(image, (int(boxes[0][0]), int(boxes[0][1])), (int(boxes[0][2]), int(boxes[0][3])), (0, 0, 255), 2)
    cv2.imshow("", image)
    cv2.waitKey(0)


# ----------------------------------------------------------
#       perspective transform of image and bboxes
# ----------------------------------------------------------


def _persp(image, bboxes):

    H, W = image.shape[:2]
    pts = np.float32([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]])
    # m = np.random.randint(10, 20)
    m = 50
    rand_num = np.random.randint(0, 4)
    if rand_num == 0:
        pts_ = np.float32([[m, 0], [W - 1 - m, 0], [W - 1, H - 1], [0, H - 1]])
    elif rand_num == 1:
        pts_ = np.float32([[0, 0], [W - 1, 0], [W - 1 - m, H - 1], [m, H - 1]])
    elif rand_num == 2:
        pts_ = np.float32([[0, m], [W - 1, 0], [W - 1, H - 1], [0, H - 1 - m]])
    else:
        pts_ = np.float32([[0, 0], [W - 1, m], [W - 1, H - 1 - m], [0, H - 1]])

    # 透视变换矩阵
    M = cv2.getPerspectiveTransform(pts, pts_)
    image = cv2.warpPerspective(image, M, (W, H))

    # bbox变换
    corners = get_corners(bboxes)
    corners = corners.reshape(-1, 2)
    corners = np.hstack((corners, np.ones((corners.shape[0], 1), dtype=type(corners[0][0]))))

    # Prepare the vector to be transformed
    calculated = np.matmul(M, corners.T).T
    calculated = calculated[:, :-1] / calculated[:, -1].reshape(-1, 1)
    calculated = calculated.reshape(-1, 8)
    bboxes = get_enclosing_box(calculated)
    # bboxes = bboxes.astype(np.int)

    return image, bboxes


def _test_persp():
    image = cv2.imread('/data_ssd2/hzh/paperworks/dataset/screenshots/00036.png')

    # 测试标记
    bboxes = np.array([[100, 100, 500, 500],
                       [200, 400, 600, 700]])
    image = cv2.rectangle(image, (bboxes[0][0], bboxes[0][1]), (bboxes[0][2], bboxes[0][3]), (0, 0, 255), 2)
    image = cv2.rectangle(image, (bboxes[1][0], bboxes[1][1]), (bboxes[1][2], bboxes[1][3]), (0, 0, 255), 2)

    image, bboxes = _persp(image, bboxes)
    image = cv2.rectangle(image, (bboxes[0][0], bboxes[0][1]), (bboxes[0][2], bboxes[0][3]), (255, 0, 0), 2)
    image = cv2.rectangle(image, (bboxes[1][0], bboxes[1][1]), (bboxes[1][2], bboxes[1][3]), (255, 0, 0), 2)

    cv2.imshow("", image)
    cv2.waitKey(0)


if __name__ == "__main__":
    image = cv2.imread('/data_ssd2/hzh/paperworks/dataset/original/screenshots/00036.png')
    image = cv2.imread('/home/hzh/桌面/bkg.jpg')
    bboxes = np.array([[100, 100, 500, 500],
                       [200, 400, 600, 700]])
    # image = cv2.rectangle(image, (bboxes[0][0], bboxes[0][1]), (bboxes[0][2], bboxes[0][3]), (0, 0, 255), 2)
    # image = cv2.rectangle(image, (bboxes[1][0], bboxes[1][1]), (bboxes[1][2], bboxes[1][3]), (0, 0, 255), 2)
    boxes = bboxes.astype(np.float)
    image, boxes = _noise(image, boxes)
    cv2.imwrite('/home/hzh/桌面/1noise.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    image, boxes = _mirror(image, boxes)
    cv2.imwrite('/home/hzh/桌面/2mirror.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    image, boxes = _rotate(image, boxes)
    cv2.imwrite('/home/hzh/桌面/3rotate.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    image, boxes = _persp(image, boxes)
    cv2.imwrite('/home/hzh/桌面/4persp.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    image = image.astype(np.uint8)

    # boxes = boxes.astype(np.int)
    # image = cv2.rectangle(image, (boxes[0][0], boxes[0][1]), (boxes[0][2], boxes[0][3]), (255, 0, 0), 2)
    # image = cv2.rectangle(image, (boxes[1][0], boxes[1][1]), (boxes[1][2], boxes[1][3]), (255, 0, 0), 2)

    cv2.imshow("", image)
    cv2.waitKey(0)

