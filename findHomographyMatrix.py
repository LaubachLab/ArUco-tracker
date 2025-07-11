import numpy as np
import cv2 #include <opencv2/opencv.hpp>
import csv
import os
import sys
from pathlib import Path

# define the homography matrix
H = np.full((3, 3), -1)

# ANSI terminal color codes
BLACK = '\033[30m'
RED = '\033[31m'
GREEN = '\033[32m'
YELLOW = '\033[33m'
BLUE = '\033[34m'
MAGENTA = '\033[35m'
CYAN = '\033[36m'
WHITE = '\033[37m'
RESET = '\033[0m'

# file paths
base_dir = Path(r"C:\Users\raysh\OneDrive\Documents\bonsai\REAL\arucotracking")
corners = base_dir / "chessboardCorners.csv"
output = base_dir / "homography.csv"
lock = base_dir / "lock.lock" 

# find homography matrix from chessboard corner coordinates (in pixels) (once at the very beginning)
def findHomographyMatrix():
    global H

    with open(corners, 'r', newline = '') as csvfile:
        # read the input row
        reader = csv.reader(csvfile)
        row = next(reader)

        # must have 7x7 = 49 pairs of "(x, y)" for a 7x7 chessboard
        if not len(row) == 49:
            sys.exit(f"{RED}Invalid chessboard input{RESET}")

        try:
            # remove weird bonsai formatting
            cornersUnpaired = []
            for pair in row:
                pair = pair.strip('"()')
                nums = [float(v.strip()) for v in pair.split(',')]
                cornersUnpaired.extend(nums)
            # group coordinate pairs
            coords = [[cornersUnpaired[i], cornersUnpaired[i + 1]] for i in range(0, len(cornersUnpaired), 2)]
            # (x, y) (in pixels) of 5 points on the chessboard
            x1, y1 = coords[0]
            x2, y2 = coords[6]
            x3, y3 = coords[24]
            x4, y4 = coords[42]
            x5, y5 = coords[48]
            print(f"{CYAN}Chessboard calibration points (pixels):{RESET}\n({x1:.3f}, {y1:.3f}), ({x2:.3f}, {y2:.3f}), ({x3:.3f}, {y3:.3f}), ({x4:.3f}, {y4:.3f}), ({x5:.3f}, {y5:.3f})")

            # source points of the 5 points (in pixels)
            pos_src = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4], [x5, y5]], dtype = np.float32)
        
            # destination points of the 5 points (in squares)
            pos_dst = np.array([[0, 0], [6, 0], [3, 3], [0, 6], [6, 6]], dtype = np.float32)

            # compute & ensure valid homography matrix
            H_check, status = cv2.findHomography(pos_src, pos_dst, cv2.RANSAC)
            if H_check is not None:
                H = H_check

                # write to homography matrix csv file
                with open(output, "w", newline = '') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(H)

                # remove the lock, allowing bonsai to use the now computed homography matrix csv
                os.remove(lock)
                print(f"\nHomography matrix:\n{np.round(H, 3)}")
            else:
                sys.exit(f"{RED}Invalid homography matrix computation{RESET}")
        except ValueError as e:
            sys.exit(f"{RED}Error converting to float: {e}{RESET}")
            
# function call
print(f"{MAGENTA}------------------------------------------------------------------------------{RESET}")
findHomographyMatrix()
