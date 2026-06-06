import numpy as np
import scipy.io as sio
import sys
import os
from PIL import Image
import cv2
from part1 import part1

### PART 1 ###
# Feature Detection, then calling Part1.py for feature matching, outlier removal with RANSAC and homography Estimation

def main():
    if len(sys.argv) != 5:
        print("Usage: python main1.py path_to_refimg path_images_dir path_feature_dir path_output_dir")
        sys.exit(1)

    path_to_refimg = sys.argv[1]
    path_images_dir = sys.argv[2]
    path_feature_dir = sys.argv[3]
    path_output_dir = sys.argv[4]

    os.makedirs(path_feature_dir, exist_ok=True)
    os.makedirs(path_output_dir, exist_ok=True)

    sift = cv2.SIFT_create()

    ref_img = np.array(Image.open(path_to_refimg))
    
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_RGB2GRAY)
    kp_ref, des_ref = sift.detectAndCompute(ref_gray, None)

    ref_feature_path = os.path.join(path_feature_dir, "reference.mat")
    save_features(kp_ref, des_ref, ref_feature_path)

    image_files = sorted(
        f for f in os.listdir(path_images_dir)
        if f.lower().endswith('.jpg')
    )

    for fname in image_files:
        print(fname)
        img_path = os.path.join(path_images_dir, fname)
        img = np.array(Image.open(img_path))

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        kp, des = sift.detectAndCompute(gray, None)

        name, _ = os.path.splitext(fname)
        feature_path = os.path.join(path_feature_dir, f"{name}.mat")

        save_features(kp, des, feature_path)
    
    ## CALL PART 1 
    part1(
        path1 = path_to_refimg,
        path2 = path_images_dir,
        path3 = path_feature_dir,
        path4 = path_output_dir
    )
    
    ## Warp images 
    for i, fname in enumerate(image_files):
        img_path = os.path.join(path_images_dir, fname)
        img = np.array(Image.open(img_path))
        name, _ = os.path.splitext(fname)
        number = name.split('_')[-1]
    
        # Load the corresponding homography
        homography_file = os.path.join(path_output_dir, f"homography_{number}.mat")
        mat = sio.loadmat(homography_file)
        H = mat["H"]
    
        # Get reference image size
        h_ref, w_ref = ref_gray.shape
        # Warp the current image using the homography
        warped = cv2.warpPerspective(img, H, (w_ref, h_ref))
    
        # Save the warped image
        warped_path = os.path.join(path_output_dir, f"warped_{fname}")
        Image.fromarray(warped).save(warped_path)
    
    print(f"Main 1 complete! Homographies found and images warped, stored in {path_output_dir}")

 
def save_features(kp, des, outkp_path):
    combined = np.concatenate(
        (
            np.array([k.pt for k in kp], dtype=np.float32).T,  # (2, N)
            des.T                                             # (D, N)
        ),
        axis=0
    )
    sio.savemat(outkp_path, {'kp': combined})



if __name__ == "__main__":
    main()