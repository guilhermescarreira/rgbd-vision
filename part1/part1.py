import numpy as np
import scipy
import scipy.io as sio
from scipy.spatial.distance import cdist
import os
#import cv2

def part1(path1, path2, path3, path4):
    '''
    path1: reference image (unused i think)
    path2: directory with other images (unused too)
    path3: directory with each image's features
    path4: output directory with homographies to each of the images
    '''
    os.makedirs(path4, exist_ok=True)

    referenceFeaturePath = os.path.join(path3, 'reference.mat')

    # Collect all feature files except the reference
    feature_files = sorted(
        f for f in os.listdir(path3)
        if f.endswith('.mat') and f != 'reference.mat'
    )

    for idx, fname in enumerate(feature_files):
        print(f'Image {idx + 1}')
        featurePath = os.path.join(path3, fname)
        name, _ = os.path.splitext(fname)
        number = name.split('_')[-1]

        output_name = f'homography_{number}.mat'
        output_path = os.path.join(path4, output_name)

        estimate_homography(featurePath, referenceFeaturePath, output_path)
    
    
def estimate_homography(featurePath, referenceFeaturePath, output_path):
    '''
    Estimates homography between NNNN image and reference image 
    '''
    
    refdata = sio.loadmat(referenceFeaturePath)
    refcombined = refdata['kp']
    refDescriptors = refcombined[2:, :].T
    refPts = refcombined[:2, :].T

    data = sio.loadmat(featurePath)
    combined = data['kp']
    descriptors = combined[2:, :].T
    pts = combined[:2, :].T


    ## Feature Matching
    distances = cdist(descriptors, refDescriptors, metric='euclidean')
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
            points_img1.append(pts[i])
            points_img2.append(refPts[best_index])
    '''
    bf = cv2.BFMatcher() ##TODO maybe usar FLANN aqui em vez de BFMatcher
    matches = bf.knnMatch(descriptors, refDescriptors, k=2)
    
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append([m])
            
    points_img1 = []
    points_img2 = []
    
    for match in good_matches:
        m = match[0] # because good_matches is a list of lists of only one element
        p1 = pts[m.queryIdx]
        p2 = refPts[m.trainIdx]
        points_img1.append(p1)
        points_img2.append(p2)
    '''
    ## Added on project update
    if len(points_img1) < 4:
        print("Not enough matches found")
        return

    ## Outlier Removal
    pts1 = np.float32(points_img1)
    pts2 = np.float32(points_img2)
    homogeneous_pts1 = np.vstack([pts1.T, np.ones((1, pts1.shape[0]))])

    n = 4
    P = 0.99
    p = 0.5
    max_iterations = 10000
    iteration = 0
    inlier_margin = 3

    most_inliers = -np.inf
    best_H = np.array([])
    while iteration < max_iterations:
        random_indexes = np.random.choice(len(pts1), 4, replace=False)
        rand_pts1 = pts1[random_indexes]
        rand_pts2 = pts2[random_indexes]
    
        A = []
        for [x1, y1], [x2, y2] in zip(rand_pts1, rand_pts2):
            A.append([x1, y1, 1, 0, 0, 0, -x2*x1, -x2*y1, -x2])
            A.append([0, 0, 0, x1, y1, 1, -y2*x1, -y2*y1, -y2])
        
        A = np.array(A)
        
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        H = Vt[-1].reshape(3, 3)
        
        # Project points of image1 into image2 using H (obtained from the previous step)
        new_pts1 = H.dot(homogeneous_pts1)
        new_pts1 = np.column_stack((new_pts1[0] / new_pts1[2] , new_pts1[1] / new_pts1[2]))
        
        # Calculate error, count inliers and keep H if it's the best so far
        diff = new_pts1 - pts2
        distances = np.linalg.norm(diff, axis=1)
        inliers = np.sum(distances < inlier_margin)
        if inliers > most_inliers:
            most_inliers = inliers
            best_H = H
            p = inliers / len(pts1)
            p = max(0.001, min(0.99, p))
            if (1 - p**n) > 0 and np.log(1 - p**n) != 0:
                max_iterations = int(np.log(1 - P) / np.log(1 - p**n))

        iteration += 1
    
    #print(most_inliers)
    #print(best_H)
    
    correct_pts1 = best_H.dot(homogeneous_pts1)
    correct_pts1 = np.column_stack((correct_pts1[0] / correct_pts1[2] , correct_pts1[1] / correct_pts1[2]))
    correct_matched_pts1 = []
    correct_matched_pts2 = []
    for [x1, y1], [corr_x1, corr_y1], [x2, y2] in zip(pts1, correct_pts1, pts2):
        distance = np.sqrt((x2 - corr_x1) ** 2 + (y2 - corr_y1) ** 2)
        if distance < inlier_margin:
            correct_matched_pts1.append([x1, y1])
            correct_matched_pts2.append([x2, y2])
    
    ## Normalization
    pts1_norm, T1 = normalize_points(correct_matched_pts1) # normalize to calculate SVD easily (with smaller values)
    pts2_norm, T2 = normalize_points(correct_matched_pts2)
    
    ## Homography Estimation via Least Squares
    A = []
    for [x1, y1], [x2, y2] in zip(pts1_norm, pts2_norm):
        A.append([x1, y1, 1, 0, 0, 0, -x2*x1, -x2*y1, -x2])
        A.append([0, 0, 0, x1, y1, 1, -y2*x1, -y2*y1, -y2])
    
    A = np.array(A)
    
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    H_norm = Vt[-1].reshape(3, 3)
    
    ## De-normalize Homography
    
    H = np.linalg.inv(T2) @ H_norm @ T1
    H = H / H[2, 2] # fixing h_33 to 1
    
    print("Homography: \n", H) 
    sio.savemat(output_path, {'H': H})



def normalize_points(pts):
    pts = np.asarray(pts)

    centroid = np.mean(pts, axis=0)
    pts_shifted = pts - centroid

    mean_dist = np.mean(np.sqrt(np.sum(pts_shifted**2, axis=1)))
    scale = np.sqrt(2) / mean_dist

    T = np.array([
        [scale, 0, -scale * centroid[0]],
        [0, scale, -scale * centroid[1]],
        [0, 0, 1]
    ])

    # Convert to homogeneous coordinates
    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
    pts_norm_h = (T @ pts_h.T).T

    return pts_norm_h[:, :2], T