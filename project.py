import cv2
import numpy as np
from utils import compute_d_optimal_1, compute_d_optimal_2, compute_distance_l1, compute_distance_l2, cosine_similarity


def pixel_wise_matching(left_img, right_img, disparity_range=16, scale=16, save_resutlts=True, distance_methods=compute_distance_l1):
    left = cv2.imread(left_img, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    right = cv2.imread(right_img, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    height, width = left.shape[:2]

    # create the result matrix
    depth = np.zeros((height, width))
    for row in range(height):
        for col in range(width):
            depth[row, col] = compute_d_optimal_1(
                left, right, row, col, disparity_range, distance_methods) * scale
    depth = depth.astype(np.uint8)
    if save_resutlts:
        print("Saving results ...")
        if distance_methods == compute_distance_l1:
            cv2.imwrite("results\\depth_l1.png", depth)
            cv2.imwrite("results\\depth_l1_color.png", cv2.applyColorMap(
                depth, cv2.COLORMAP_JET))

        elif distance_methods == compute_distance_l2:
            cv2.imwrite("results\\depth_l2.png", depth)
            cv2.imwrite("results\\depth_l2_color.png", cv2.applyColorMap(
                depth, cv2.COLORMAP_JET))
        print("Done!")
    return depth


def window_based_matching(left_img, right_img, disparity_range=64, kernel_size=3, save_resutlts=True, distance_methods=compute_distance_l1):
    left = cv2.imread(left_img, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    right = cv2.imread(right_img, cv2.IMREAD_GRAYSCALE).astype(np.float32)

    height, width = left.shape[:2]

    # create the result matrix
    depth = np.zeros((height, width))
    kernel_half = int((kernel_size-1)/2)
    scale = 3

    for y in range(kernel_half, height - kernel_half + 1):
        for x in range(kernel_half, width - kernel_half + 1):
            depth[y, x] = compute_d_optimal_2(
                left, right, y, x, kernel_half, disparity_range, distance_methods) * scale

    depth = depth.astype(np.uint8)
    if save_resutlts:
        print("Saving results ...")
        if distance_methods == compute_distance_l1:
            cv2.imwrite("results\\window_based_l1.png", depth)
            cv2.imwrite("results\\window_based_l1_color.png", cv2.applyColorMap(
                depth, cv2.COLORMAP_JET))

        elif distance_methods == compute_distance_l2:
            cv2.imwrite("results\\window_based_l2.png", depth)
            cv2.imwrite("results\\window_based_l2_color.png", cv2.applyColorMap(
                depth, cv2.COLORMAP_JET))
        print("Done!")
    return depth


def window_based_matching_cosine(left_img, right_img, disparity_range=64, kernel_size=3, save_resutlts=True):
    left = cv2.imread(left_img, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    right = cv2.imread(right_img, cv2.IMREAD_GRAYSCALE).astype(np.float32)

    height, width = left.shape[:2]

    # create the result matrix
    kernel_half = int((kernel_size-1)/2)
    depth = np.zeros((height, width))
    scale = 3

    for y in range(kernel_half, height - kernel_half):
        for x in range(kernel_half, width - kernel_half):
            disparity = 0
            cost_optimal = -1

            for j in range(disparity_range):
                d = x - j
                cost = -1
                if (d - kernel_half) >= 0:
                    wp = left[(y - kernel_half): (y + kernel_half) + 1, (x -
                                                                         kernel_half): (x + kernel_half) + 1]
                    wqd = right[(y - kernel_half):(y + kernel_half) + 1, (d -
                                                                          kernel_half):(d + kernel_half) + 1]

                    # cosine_similarity(wp, wqd)
                    cost = cosine_similarity(wp.flatten(), wqd.flatten())
                if cost > cost_optimal:
                    cost_optimal = cost
                    disparity = j

                depth[y, x] = disparity * scale

    depth = depth.astype(np.uint8)
    if save_resutlts:
        print("Saving results ...")
        cv2.imwrite("results\\window_based_cosine.png", depth)
        cv2.imwrite("results\\window_based_cosine_color.png", cv2.applyColorMap(
            depth, cv2.COLORMAP_JET))
        print("Done!")
    return depth


if __name__ == "__main__":
    LEFT_PATH_PIXEL = "data\\tsukuba\\left.png"
    RIGHT_PATH_PIXEL = "data\\tsukuba\\right.png"

    LEFT_PATH_WINDOW = "data\\aloe\\Aloe_left_1.png"
    RIGHT_PATH_WINDOW = "data\\aloe\\Aloe_right_1.png"

    pixel_wise_matching(LEFT_PATH_PIXEL, RIGHT_PATH_PIXEL, save_resutlts=True,
                        distance_methods=compute_distance_l1)

    pixel_wise_matching(LEFT_PATH_PIXEL, RIGHT_PATH_PIXEL, save_resutlts=True,
                        distance_methods=compute_distance_l2)

    window_based_matching(LEFT_PATH_WINDOW, RIGHT_PATH_WINDOW, save_resutlts=True,
                          distance_methods=compute_distance_l1)

    window_based_matching(LEFT_PATH_WINDOW, RIGHT_PATH_WINDOW, save_resutlts=True,
                          distance_methods=compute_distance_l2)

    window_based_matching_cosine(
        LEFT_PATH_WINDOW, RIGHT_PATH_WINDOW, save_resutlts=True)
    window_based_matching_cosine(
        LEFT_PATH_WINDOW, RIGHT_PATH_WINDOW, save_resutlts=True)
