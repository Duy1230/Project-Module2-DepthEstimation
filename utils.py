import numpy as np
import cv2


def compute_distance_l1(pt1, pt2):
    return np.abs(pt1-pt2)


def compute_distance_l2(pt1, pt2):
    return (pt1-pt2)**2


def cosine_similarity(x, y):
    numerator = np.dot(x, y)
    denominator = np.linalg.norm(x)*np.linalg.norm(y)
    return numerator/denominator


def compute_d_optimal_1(left, right, height, width, disparty_range, compute_distance=compute_distance_l1):
    max_cost = 255
    cost = list()
    for i in range(disparty_range+1):
        if width - i < 0:
            cost.append(max_cost)
        else:
            cost.append(compute_distance(
                left[height, width], right[height, width-i]))

    return np.argmin(cost)


def compute_d_optimal_2(left, right, height, width, kernel_half, disparty_range, compute_distance=compute_distance_l1):
    max_value = 255 * 9
    disparity = 0
    cost_min = 65534
    for j in range(disparty_range):
        total = 0
        value = 0
        for v in range(-kernel_half, kernel_half):
            for u in range(-kernel_half, kernel_half):
                value = max_value
                if ((width + u - j) >= 0):
                    value = compute_distance(
                        int(left[height + v, width + u]), int(right[height + v, width + u - j]))
                total += value
        if total < cost_min:
            cost_min = total
            disparity = j

    return disparity
