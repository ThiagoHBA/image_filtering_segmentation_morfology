from random import seed
import cv2
import math
import numpy as np

def main():
  
    # Q1: 
    # image_path = 'images/orion.png'
    # image = cv2.imread(image_path)
    # image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('Original',image_gray)
    # pixels_matrix = image_gray.tolist()
    # cv2.imshow('LIMIARIZATION', np.array(perform_limiarization_filter(pixels_matrix), dtype=np.uint8))

    # Q2:
    # image_path = 'images/orion.png'
    # image = cv2.imread(image_path)
    # image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('Original',image_gray)
    # cv2.imshow('Dilation', np.array(apply_dilation(image_gray), dtype=np.uint8))
    # cv2.imshow('Erosion', np.array(apply_erosion(image_gray), dtype=np.uint8))
    # cv2.imshow('Erosion + Dilation', np.array(apply_dilation(apply_erosion(image_gray), iterations=3), dtype=np.uint8))
    # cv2.imshow('Erosion + Limiarization', np.array(perform_limiarization_filter(apply_erosion(image_gray), threshold=100), dtype=np.uint8))
    # pixels_matrix = image_gray.tolist()
    # cv2.imshow('Limiarization + Dilation', np.array(apply_dilation(np.array(perform_limiarization_filter(pixels_matrix), dtype=np.uint8)), dtype=np.uint8))
   
    # Q3:
    # image_path = 'images/local_limiarization_example7.png'
    # image = cv2.imread(image_path)
    # image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('ORIGINAL',image_gray)
    # pixels_matrix = image_gray.tolist()
    # cv2.imshow('LOCAL LIMIARIZATION', np.array(perform_local_mean_limiarization_filter(pixels_matrix), dtype=np.uint8))
    # cv2.imshow('LOCAL LIMIARIZATION BASE ', np.array(cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)))
  
    # Q4: 
    image_path = 'images/region_growing_example.png'
    image = cv2.imread(image_path, 0)

    cv2.imshow('ORIGINAL',image)
    cv2.imshow('REGION GROWING GRAY', np.array(growingRegionGray(image, 264, 378, 90), dtype=np.uint8))

    #Q5:
    # image_path = 'images/hough_example2.jpg'
    # image = cv2.imread(image_path)
    # image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # cv2.imshow('ORIGINAL',image_gray)
    # cv2.imshow('HOUGH', detect_lines_with_hough(image_gray))
    

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def growingRegionGray(image, x, y, differential):
    result_image = image

    threshold = image[x][y]
    kernel = 3
    kernelSize = kernel*kernel
    kernelA = int(math.sqrt(kernelSize))
    kernelB = int(kernelA)
    kernelStep = -1*int(kernelA/2)
    
    pixelsGrowing = [[x, y]]

    for u in pixelsGrowing:
        line = u[0]
        col = u[1]
        for a in range(kernelStep, kernelA + kernelStep):
            for b in range(kernelStep, kernelB + kernelStep):
                try:
                    if (threshold - differential <= image[line + a][col + b] <= threshold + differential):
                        if [line + a, col + b] in pixelsGrowing:
                            continue
                        else:
                            result_image[line + a][col + b] = threshold
                            image[line + a][col + b] = 1
                            pixelsGrowing.append([line + a, col + b])
                except:
                    continue

    return result_image


def detect_lines_with_hough(image_gray, hough_threshold=200):
    sobel = np.array(perform_high_pass_filter_sobel(image_gray.tolist()), dtype=np.uint8)
    lines = cv2.HoughLines(sobel, 1, np.pi / 180, hough_threshold)
    
    if lines is not None:
        for rho, theta in lines[:, 0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(image_gray, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return image_gray

def perform_high_pass_filter_sobel(pixels_matrix):
    kernel_x = [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ]

    kernel_y = [
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ]

    rows, cols = len(pixels_matrix), len(pixels_matrix[0])
    
    result_matrix_x = [[0 for _ in range(cols)] for _ in range(rows)]
    result_matrix_y = [[0 for _ in range(cols)] for _ in range(rows)]
    result_matrix = [[0 for _ in range(cols)] for _ in range(rows)]

    kernel_center_row = len(kernel_x) // 2
    kernel_center_col = len(kernel_x[0]) // 2

    for i in range(rows):
        for j in range(cols):
            value_x = 0
            value_y = 0
            for m in range(len(kernel_x)):
                for n in range(len(kernel_x[0])):
                    row = i + (m - kernel_center_row)
                    col = j + (n - kernel_center_col)

                    if 0 <= row < rows and 0 <= col < cols:
                        value_x += pixels_matrix[row][col] * kernel_x[m][n]
                        value_y += pixels_matrix[row][col] * kernel_y[m][n]

            result_matrix_x[i][j] = value_x
            result_matrix_y[i][j] = value_y

    for i in range(rows):
        for j in range(cols):
            gradient_magnitude = np.hypot(result_matrix_x[i][j], result_matrix_y[i][j])
            result_matrix[i][j] = min(max(int(gradient_magnitude), 0), 255)

    return result_matrix


def perform_local_mean_limiarization_filter(pixels_matrix):
    #2x2
    # kernel = [
    #     [1, 1],
    #     [1, 1]
    # ]
    #3x3
    # kernel = [
    #     [1, 1, 1],
    #     [1, 1, 1],
    #     [1, 1, 1]
    # ]
    #5x5
    kernel = [
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1]
    ]
    # kernel = [
    #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    # ]
    # kernel = [
    #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    # ]

    result_matrix = []

    for _ in range(len(pixels_matrix)):
        result_matrix.append([])

    for i in range(len(pixels_matrix)):
        for j in range(len(pixels_matrix[0])):
            colision_matrix = calculate_colision_matrix(pixels_matrix, kernel, i, j)
            pixels_mean = math.floor(calculate_matrix_mean(colision_matrix))

            if pixels_matrix[i][j] >= pixels_mean:
                result_matrix[i].append(255)
                continue
            result_matrix[i].append(0)
    
    return result_matrix

def apply_erosion(image, kernel_size=(3,3), iterations=1):
    kernel = np.ones(kernel_size, np.uint8)
    eroded_img = cv2.erode(image, kernel, iterations=iterations)
    return eroded_img

def apply_dilation(image, kernel_size=(3, 3), iterations=1):
    kernel = np.ones(kernel_size, np.uint8)
    dilated_img = cv2.dilate(image, kernel, iterations=iterations)
    return dilated_img

def perform_limiarization_filter(pixels_matrix, threshold = 180):
    result_matrix = []

    for _ in range(len(pixels_matrix)):
        result_matrix.append([])

    for i in range(len(pixels_matrix)):
        for j in range(len(pixels_matrix[0])):
            pixel = pixels_matrix[i][j]
            if pixel > threshold:
                result_matrix[i].append(255)
            else:
                result_matrix[i].append(0)
    
    return result_matrix

def calculate_matrix_mean(matrix):
    sum = 0
    for i in matrix:
        sum += i
    
    return sum / len(matrix)

def calculate_colision_matrix(matrix, kernel, baseRow, baseCol):
    colision_matrix = []
    kernel_center_row = len(kernel) // 2
    kernel_center_col = len(kernel[0]) // 2

    for m in range(len(kernel)):
        for n in range(len(kernel[0])):
            row = baseRow + (m - kernel_center_row)
            col = baseCol + (n - kernel_center_col)

            if 0 <= row < len(matrix) and 0 <= col < len(matrix[0]):
                colision_matrix.append(matrix[row][col])

    return colision_matrix

if __name__ == '__main__':
    main()