import os
import csv
import subprocess
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import sys



def main():

    #path_to_refimg = '../DATASETS/part2/taag3d/template'
    #path_images_dir = '../DATASETS/part2/taag3d/sequence'
    # path_to_refimg = 'template'
    # path_images_dir = 'sequence'
    # path_output_dir = 'transform_output'
    #TODO custom paths
    if len(sys.argv) != 4:
        print("Invalid arguments!")
        print("Usage: python main3.py path_to_refimgdir path_images_dir path_output_dir")
        sys.exit(1)

    path_to_refimg = sys.argv[1]
    path_images_dir = sys.argv[2]
    path_output_dir = sys.argv[3]

    os.makedirs(path_output_dir, exist_ok=True)

    group_path = '.'

    cmd = ["python3","main2.py",path_to_refimg,path_images_dir,path_output_dir]

    print(f"\nRunning command in {group_path}...")
    try:
        result = subprocess.run(
            cmd,
            cwd=group_path,
            capture_output=True,
            text=True,
            check=False
        )
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        # Print stdout to console
        print(stdout)
        if stderr:
            print(f"[stderr]\n{stderr}")

    except Exception as e:
        print(f"Error running {group_path}: {e}")

    print(f'Finished {group_path}')


    tim = cv2.cvtColor(
        cv2.imread(os.path.join(path_to_refimg,'templatergb.jpg')),
        cv2.COLOR_BGR2RGB
    )

    mat = sio.loadmat(os.path.join(path_to_refimg,'templatedepth.mat'))
    depth = mat['depth']
    K = mat['K']

    h, w = depth.shape
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]

    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    z = depth.reshape(-1)
    valid = z > 0

    x = (xs.reshape(-1) - cx) * z / fx
    y = (ys.reshape(-1) - cy) * z / fy

    ref_pts = np.stack([x, y, z], axis=1)[valid]
    ref_colors = tim[ys.reshape(-1)[valid], xs.reshape(-1)[valid]]


    plane, mask = get_best_plane(ref_pts)
    n, p = plane
    # print(np.mean(ref_pts, axis=0))

    ii = 0
    for fname in sorted(os.listdir(path_images_dir)):

        if not (fname.endswith('.jpg') or fname.endswith('.png')):
            continue

        ii += 1

        # --- read RGB ---
        im_rgb = cv2.imread(os.path.join(path_images_dir, fname))
        im = cv2.cvtColor(im_rgb, cv2.COLOR_BGR2RGB)

        # --- load per-frame transform ---
        data = sio.loadmat(os.path.join(group_path, path_output_dir,
                                        'transform_' + os.path.splitext(fname)[0].split('_')[1] + '.mat'))
        scale, R, T = data['scale'], data['R'], data['T'].reshape(3)

        # --- load depth and intrinsics ---
        mat = sio.loadmat(os.path.join(path_images_dir, os.path.splitext(fname)[0] + ".mat"))
        depth = mat['depth']
        K = mat['K']

        h, w = depth.shape
        fx, fy = K[0,0], K[1,1]
        cx, cy = K[0,2], K[1,2]

        # --- backproject to 3D ---
        ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        z = depth.reshape(-1)
        valid = z > 0

        x = (xs.reshape(-1) - cx) * z / fx
        y = (ys.reshape(-1) - cy) * z / fy

        pc = np.stack([x, y, z], axis=1)[valid]
        colors = im[ys.reshape(-1)[valid], xs.reshape(-1)[valid]]

        # --- transform into template frame ---
        pts = scale * (pc @ R.T) + T

        # --- plane filtering ---
        dist = np.abs((pts - p) @ n)
        mask = dist < 0.04  # adjust if needed

        pts = pts[mask]
        colors = colors[mask]

        if len(pts) == 0:
            print(f"Frame {ii}: no plane points")
            continue

        # --- project into template image ---
        img_pts, _ = cv2.projectPoints(
            pts.astype(np.float32),
            np.zeros((3,1)),
            np.zeros((3,1)),
            K.astype(np.float32),
            None
        )
        img_pts = img_pts.reshape(-1, 2)

        # --- dense rasterization with splatting ---
        template_equiv = tim
        # template_equiv = np.zeros_like(tim) ##over a black canvas
        zbuf = np.full((h, w), np.inf)

        # Splat size: 3x3 neighborhood
        splat = [-1, 0, 1]

        for (u, v), z_val, color in zip(img_pts, pts[:,2], colors):
            u = int(round(u))
            v = int(round(v))
            for du in splat:
                for dv in splat:
                    uu = u + du
                    vv = v + dv
                    if 0 <= uu < w and 0 <= vv < h:
                        if z_val < zbuf[vv, uu]:
                            zbuf[vv, uu] = z_val
                            template_equiv[vv, uu] = color

        

        plt.imsave(f"output/template_equiv_{ii-1}.png", template_equiv)
        print(f"Image {ii} completed")

## ransac plane fitting

def get_best_plane(tpc):
        
    max_tries = 1000
    error_treshold = 0.02

    best_plane = None
    best_inliers_count = 0

    points = tpc.reshape(-1, 3)  # shape (518*518, 3)
    for _ in range(max_tries):
        random_indices = np.random.choice(points.shape[0], size=3, replace=False)

        p1, p2, p3 = points[random_indices[0]], points[random_indices[1]], points[random_indices[2]]

        v1 = p2 - p1
        v2 = p3 - p1
        n = np.cross(v1, v2)
        if np.linalg.norm(n) == 0:
            continue  # points are collinear, skip iteration
        n = n / np.linalg.norm(n) 

        distances = np.abs( np.dot(points - p1, n) )  # Nx1 array

        current_inliers_mask = distances < error_treshold
        inliers_count = np.sum(current_inliers_mask)

        if inliers_count > best_inliers_count:
            best_inliers_count = inliers_count
            best_plane = (n, p1)
            inliers_mask = current_inliers_mask
    print(best_inliers_count)
    return best_plane, inliers_mask


if __name__ == "__main__":
    main()