#  RGB-D Document Tracking & Rectification

This repository contains a full pipeline for tracking, rectifying, and cleaning a planar document observed by a moving camera. It was developed as a final project for an Image Processing and Vision course and utilizes both 2D homography estimation and 3D RGB-D depth mappings.

The goal is to accurately map every video frame back to a flat "reference template" view, eventually filtering out occlusions (like hands, bodies, or non-planar objects) using robust geometric vision algorithms built from scratch.

![Final Result of Document Rectification](assets/movie_output.gif)

---

##  The Methodology Pipeline

The project is divided into three consecutive phases, moving from pure 2D approximations into robust 3D modeling.

### Part 1: 2D Homography Estimation from RGB Images
In the first phase, the problem is formulated in a 2D setting.
* **Feature Matching:** Used SIFT feature detection and k-nearest neighbors (k=2) combined with Lowe's Ratio Test to find correspondences between frames.
* **Robust Outlier Rejection:** Implemented a custom RANSAC loop (with dynamic iteration calculation) to estimate a 2D Homography Matrix using the Direct Linear Transform (DLT) algorithm.
* **Data Normalization:** Ensured numerical stability using SVD (Singular Value Decomposition) on centered and scaled inliers.

### Part 2: Incorporating Depth Information (RGB-D)
To handle complex camera angles and physical scaling, depth data (`.npy` and `.mat` files) is introduced to convert 2D features into 3D directions.
* **3D Point Extraction:** Extracted 3D points using camera intrinsic matrices ($fx$, $fy$, $cx$, $cy$).
* **Rigid Transformations:** Calculated rotation ($R$) and translation ($t$) matrices between point clouds using custom **Orthogonal Procrustes Analysis**.
* **Optical Flow:** Tracked features across sequential frames using the Lucas-Kanade method (`calcOpticalFlowPyrLK`), filtering out points that demonstrated excessive median motion to guarantee robust tracking.

### Part 3: Out-of-Plane Object Removal
The final module utilizes the rigid 3D transformations obtained in Part 2 to clean the document view.
* **3D Plane Fitting:** Built a 3D RANSAC algorithm to find the dominant plane (the document) in the reference point cloud.
* **Occlusion Masking:** Per-frame point clouds are rotated and translated into the template frame. Any point that falls outside an acceptable threshold distance from the computed document plane is classified as an occlusion (e.g., a hand) and stripped away.
* **Dense Rasterization:** The remaining valid 3D points are re-projected onto the 2D template using z-buffering and splatting, generating a clean, perspective-corrected image free of foreground obstructions.

---

##  Repository Structure

```text
├── assets/
│   ├── movie.gif                  # Input visualizations
│   └── movie_output.gif           # Result visualizations
├── notebooks/
│   └── check_protocol_task2.ipynb # 3D point cloud visual debugger
├── part1/
│   ├── data/                      # 2D sequence inputs
│   ├── main1.py                   # SIFT extraction and orchestration
│   └── part1.py                   # 2D Homography, RANSAC, and SVD engine
├── part2/
│   ├── data/                      # RGB-D sequence and template inputs
│   └── main2.py                   # 3D Procrustes Analysis and Rigid Transformations
├── part3/
│   ├── data/                      # Cleaned output destination
│   └── main3.py                   # 3D Plane fitting, occlusion removal, and splatting
├── utils/
│   └── make_gif.py                # Fast GIF generation utility
└── README.md
```

## Prerequisites
The scripts rely heavily on standard scientific computing libraries. Ensure you have the following installed:

```Bash
pip install numpy scipy opencv-python matplotlib pillow
```

## A Note on Datasets
The original RGB-D dataset used to evaluate Part 3 (featuring heavy foreground obstructions over a planar surface) was hosted on a closed university server and is not included in this public repository. 

The `plondres` (Praça de Londres) dataset provided in the `data/` folder serves as a structural placeholder. It allows the full 3D pipeline to execute without errors, demonstrating the architecture and mathematical implementations (Procrustes, RANSAC, Z-buffering), even though it lacks dramatic out-of-plane occlusions to filter.

## Execution
The code is modularized by project phase. For example, to run the final out-of-plane object removal pipeline (Part 3), use the following terminal command from the root directory:

```Bash
cd part3
python main3.py <path_to_reference_directory> <path_to_sequence_directory> <path_to_output_directory>
```

or simply

```Bash
cd part3
python main3.py
```

## Utilities
If you are iterating on the pipeline and want to visualize a sequence of output frames (such as the homography warping in Part 1 or the occlusion removal in Part 3), a standalone, highly optimized GIF generator is included.

Run it from the root directory, pointing it to your target image folder:
```bash
python utils/make_gif.py <path_to_image_folder> <output_filename.gif> [fps]

# Example:
# python utils/make_gif.py part1/data/capture_output assets/new_movie.gif 20
```

## Limitations & Future Work
* **Homography Chaining:** Currently, Part 1 matches all sequence frames directly to the reference image. In cases of extreme perspective shifts, SIFT feature matching degrades. Future iterations will implement sequential matching (matching $I_t$ to $I_{t-1}$) and matrix multiplication chaining to maintain robust tracking across wider baseline movements.