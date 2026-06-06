import numpy as np
import scipy.io as sio
import sys
import os
from PIL import Image
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
    ref_kps, ref_descr = sift.detectAndCompute(ref_image, None) 

    ref_pointcloud, valid_indices = backproject(ref_kps, ref_depth, ref_intrinsics)
    ref_valid_descr = ref_descr[valid_indices]


    ## 2. For each image and depth in path_images_dir:
        # Compute point cloud for that image
        # Procrustes
        # RANSAC outlier removal, save final R and T to path_output_dir

    image_files = sorted(
        f for f in os.listdir(path_images_dir)
        if f.lower().endswith('.jpg')
    )

    depth_files = sorted(
        f for f in os.listdir(path_images_dir)
        if f.lower().endswith('.mat')
    )
    
    for imgname, depthname in zip(image_files, depth_files):
        print(imgname)
        print(depthname)
        
        img_path = os.path.join(path_images_dir, imgname)
        depth_path = os.path.join(path_images_dir, depthname)
        
        img = np.array(Image.open(img_path))
        mat = sio.loadmat(depth_path)
        depth = mat['depth']
        intrinsics = mat['K']

        kps_frame, descr_frame = sift.detectAndCompute(img, None)

        frame_pointcloud, frame_valid_indices = backproject(kps_frame, depth, intrinsics)
        frame_valid_descr = descr_frame[frame_valid_indices]

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(ref_valid_descr, frame_valid_descr, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        good_ref_pts = []
        good_frame_pts = []
        for match in good_matches:
            good_ref_pts.append(ref_pointcloud[match.queryIdx])
            good_frame_pts.append(frame_pointcloud[match.trainIdx])

        good_ref_pts = np.array(good_ref_pts)
        good_frame_pts = np.array(good_frame_pts)
            
        if len(good_ref_pts) < 3:
            print("Not enough good matches found")
            continue

        R, T = ransac(good_ref_pts, good_frame_pts)

        name, _ = os.path.splitext(imgname)
        number = name.split('_')[-1]

        output_name = f'transform_{number}.mat'
        output_path = os.path.join(path_output_dir, output_name)
        sio.savemat(output_path, {'R': R, 'T': T})



def backproject(keypoints, depth, K):
    points_3d = []
    valid_indices = []
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    for i, kp in enumerate(keypoints):
        u, v = int(round(kp.pt[0])), int(round(kp.pt[1]))
        h, w = depth.shape

        if 0 <= v < h and 0 <= u < w:
            z = depth[v, u]
            if z > 0:
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy

                points_3d.append([x, y, z])
                valid_indices.append(i)

    return np.array(points_3d), valid_indices

def procrustes(ref_pts, frame_pts):
    centroid_ref = np.mean(ref_pts, axis=0)
    centroid_frame = np.mean(frame_pts, axis=0)

    centered_ref_pts = ref_pts - centroid_ref
    centered_frame_pts = frame_pts - centroid_frame

    H = np.dot(centered_frame_pts.T, centered_ref_pts)

    U, S, Vt = np.linalg.svd(H)

    R = np.dot(Vt.T, U.T)

    if np.linalg.det(R) < 0: # verify if we got a rotation or a reflection
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)
    
    T = centroid_ref.reshape(3, 1) - np.dot(R, centroid_frame.reshape(3, 1))
    return R, T

def ransac(ref_pts, frame_pts):
    iterations = 1000
    most_inliers = -np.inf
    best_R = np.eye(3)
    best_T = np.zeros((3, 1))
    best_inliers_mask = None
    for i in range(iterations):
        random_indices = np.random.choice(ref_pts.shape[0], size=3, replace=False)
        random_ref_pts = ref_pts[random_indices]
        random_frame_pts = frame_pts[random_indices]

        new_R, new_T = procrustes(random_ref_pts, random_frame_pts)
        frame_proj_pts = (np.dot(new_R, frame_pts.T) + new_T).T
        errors = np.linalg.norm(ref_pts - frame_proj_pts, axis=1)

        current_inliers_mask = errors < 0.05
        inliers = np.sum(current_inliers_mask)

        if inliers > most_inliers:
            most_inliers = inliers
            best_R = new_R
            best_T = new_T
            best_inliers_mask = current_inliers_mask

    R = np.eye(3)
    T = np.zeros((3, 1))
    
    if most_inliers > 3:
        best_ref_pts = ref_pts[best_inliers_mask]
        best_frame_pts = frame_pts[best_inliers_mask]
        R, T = procrustes(best_ref_pts, best_frame_pts)

    return R, T
    

if __name__ == "__main__":
    main()