import numpy as np
import scipy.io as sio
import sys
import os
from PIL import Image
from scipy.spatial.distance import cdist
import cv2

### PART 2 ###
def main():
    if len(sys.argv) != 4:
        print("Usage: python main2.py path_to_refimgdir path_images_dir path_output_dir")
        sys.exit(1)

    path_to_refimg = sys.argv[1]
    path_images_dir = sys.argv[2]
    path_output_dir = sys.argv[3]

    os.makedirs(path_output_dir, exist_ok=True)

    ## 1. Get reference image, depth, and intrinsics from path_to_refimgdir

    refimg_path = next(
        f for f in os.listdir(path_to_refimg)
        if f.lower().endswith(".jpg")
    )
    
    image_path = os.path.join(path_to_refimg, refimg_path)
    ref_image = np.array(Image.open(image_path))

    refdepth_path = next(
        f for f in os.listdir(path_to_refimg)
        if f.lower().endswith(".mat")
    )
    
    ref_depth_path = os.path.join(path_to_refimg, refdepth_path)
    mat = sio.loadmat(ref_depth_path)
    ref_depth = mat['depth']
    ref_intrinsics = mat['K']

    sift = cv2.SIFT_create()    
    ref_gray = cv2.cvtColor(ref_image, cv2.COLOR_RGB2GRAY)
    kp_ref, des_ref = sift.detectAndCompute(ref_gray, None)

    ## 2. For each image and depth in path_images_dir:
        # Compute feature matches with SIFT
        # Compute point cloud for that image
        # Procrustes
        # RANSAC outlier removal, save final R and T to path_output_dir

    image_files = sorted(
        f for f in os.listdir(path_images_dir)
        if f.lower().endswith('.jpg')
    )
    print(image_files)
    for fname in image_files:
        print(fname)
        img_path = os.path.join(path_images_dir, fname)
        img = np.array(Image.open(img_path))

        name, _ = os.path.splitext(fname)
        depth_name = name.split('.')[-1]
        depth_file = os.path.join(path_images_dir, f"{depth_name}.mat")
        mat = sio.loadmat(depth_file)
        depth = mat['depth']
        K = mat['K']

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        kp, des = sift.detectAndCompute(gray, None)

        distances = cdist(des, des_ref, metric='euclidean')
        points_img1 = []
        points_img2 = []
        for i in range(distances.shape[0]):
            sorted_indices = np.argpartition(distances[i], 2)[:2]
            best_candidates = sorted_indices[np.argsort(distances[i, sorted_indices])]
            best_index = best_candidates[0]
            second_best_index = best_candidates[1]
            distance1 = distances[i, best_index]
            distance2 = distances[i, second_best_index]
    
            if distance1 < 0.75 * distance2: # Lowe's ratio test
                points_img1.append(kp[i].pt)
                points_img2.append(kp_ref[best_index].pt)

        matches = list(zip(points_img1, points_img2))
        print("Number of matches: ", len(matches))

        pt_cloud1, pt_cloud2 = getPointClouds(matches, depth, K, ref_depth, ref_intrinsics)

        R_ransac, t_ransac, inliers = procrustes_ransac(pt_cloud1, pt_cloud2)
        print("RANSAC Estimated rotation R:\n", R_ransac)
        print("RANSAC Estimated translation t:\n", t_ransac)
        print(f"Number of inliers: {len(inliers)} / {pt_cloud1.shape[0]}")

        number = depth_name.split('_')[-1]
        output_path = os.path.join(path_output_dir, f"transform_{number}.mat")
        sio.savemat(output_path, {'R': R_ransac, 'T': t_ransac})
    

    ## 3. Object removal (optional right now)
        # (optional for part 3) RANSAC plane fitting
        # (optional for part 3) Remove objects outside the document plane


### Write a report describing:
    # Algorithmic details (feature choice, RANSAC parameters, numerical methods)
    # Experimental setup and evaluation
    # Example results showing rectified views and depth-based segmentation


def getPointClouds(matches, depth1, K1, depth2, K2):
    points1 = []
    points2 = []

    for (pt1, pt2) in matches:
        u1, v1 = pt1
        u2, v2 = pt2
    
        z1 = depth1[int(v1), int(u1)]
        z2 = depth2[int(v2), int(u2)]
    
        X1 = (u1 - K1[0, 2]) * z1 / K1[0, 0]
        Y1 = (v1 - K1[1, 2]) * z1 / K1[1, 1]
        X2 = (u2 - K2[0, 2]) * z2 / K2[0, 0]
        Y2 = (v2 - K2[1, 2]) * z2 / K2[1, 1]
    
        points1.append([X1, Y1, z1])
        points2.append([X2, Y2, z2])
    
    P1 = np.array(points1)
    P2 = np.array(points2)
    
    return P1, P2
    
    
def procrustes_rigid(A, B):
    centroid_A = A.mean(axis=0)
    centroid_B = B.mean(axis=0)
    
    A_centered = A - centroid_A
    B_centered = B - centroid_B

    H = A_centered.T @ B_centered

    # SVD
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Handle reflection
    if np.linalg.det(R) < 0:
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t = centroid_B - R @ centroid_A

    return R, t


def procrustes_ransac(A, B, num_iters=1000, inlier_thresh=0.5):
    errors_init = np.linalg.norm(A - B, axis=1)
    inlier_thresh = np.median(errors_init) * 0.5
    best_inliers = []
    best_R = None
    best_t = None
    N = A.shape[0]

    if N < 3:
        raise ValueError("Need at least 3 points for Procrustes")

    for _ in range(num_iters):
        # Randomly sample 3 points
        idx = np.random.choice(N, 3, replace=False)
        try:
            R, t = procrustes_rigid(A[idx], B[idx])
        except np.linalg.LinAlgError:
            continue

        A_transformed = (R @ A.T).T + t

        errors = np.linalg.norm(A_transformed - B, axis=1)

        # Determine inliers
        inliers = np.where(errors < inlier_thresh)[0]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_R = R
            best_t = t

    # Recompute R, t using all inliers
    if len(best_inliers) >= 3:
        best_R, best_t = procrustes_rigid(A[best_inliers], B[best_inliers])
    else:
        raise ValueError("Not enough inliers found")

    return best_R, best_t, best_inliers


if __name__ == "__main__":
    main()