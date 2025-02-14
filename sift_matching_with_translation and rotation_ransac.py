import cv2
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import time
import psutil  # For CPU & memory usage tracking

# Google Maps API Key (Replace with your valid API key)
GOOGLE_API_KEY = "xxx" 

# Constants
NES_ZIONA_LAT, NES_ZIONA_LON = 31.9292, 34.7986
ZOOM = 18  # Adjust based on desired detail
IMAGE_SIZE = (640, 480)  # Final image size
FOV = 50  # Camera Field of View in degrees
CAMERA_HEIGHT = 100  # Camera height in meters


def track_resources():
    """Returns current CPU and memory usage."""
    return psutil.cpu_percent(), psutil.virtual_memory().percent


def download_google_earth_image(lat, lon, zoom=ZOOM, size=IMAGE_SIZE):
    """Download an orthophoto from Google Earth."""
    url = f"https://maps.googleapis.com/maps/api/staticmap?center={lat},{lon}&zoom={zoom}&size={size[0]}x{size[1]}&maptype=satellite&key={GOOGLE_API_KEY}"
    response = requests.get(url)

    if response.status_code != 200:
        raise Exception(f"[ERROR] Failed to download image: {response.status_code}")

    image = Image.open(BytesIO(response.content))
    return image


def simulate_drone_view(image, fov=FOV, height=CAMERA_HEIGHT):
    """Simulate the drone's camera view by cropping based on FOV and height."""
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    h, w, _ = image_cv.shape

    # Compute cropping ratio
    crop_ratio = np.tan(np.radians(fov / 2)) * height * 0.01  # Scale to prevent over-cropping
    crop_width = max(int(w * (1 - crop_ratio)), 1)
    crop_height = max(int(h * (1 - crop_ratio)), 1)

    x_start = max((w - crop_width) // 2, 0)
    y_start = max((h - crop_height) // 2, 0)
    cropped = image_cv[y_start:y_start + crop_height, x_start:x_start + crop_width]

    resized = cv2.resize(cropped, IMAGE_SIZE)
    return resized


def rotate_image(image, angle):
    """Rotate the image by a given angle in degrees."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, rotation_matrix, (w, h))
    return rotated


def extract_sift_features(image):
    """Extract SIFT features from an image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors


def match_sift_features(des1, des2):
    """Match SIFT features using FLANN matcher."""
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
    return good_matches


def estimate_transform_ransac(kp1, kp2, matches):
    """Estimate rotation & translation using RANSAC, returns inliers and homography."""
    if len(matches) < 4:
        print("[ERROR] Not enough matches for RANSAC.")
        return None, None

    # Extract matched keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Compute homography using RANSAC
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Extract inliers (good matches)
    inliers = [m for i, m in enumerate(matches) if mask[i]]
    return H, inliers


# Performance tracking
start_time = time.time()
cpu_start, mem_start = track_resources()

# Step 1: Download images
t1 = time.time()
image1 = download_google_earth_image(NES_ZIONA_LAT, NES_ZIONA_LON)
image2 = download_google_earth_image(NES_ZIONA_LAT, NES_ZIONA_LON + 0.0005)
t2 = time.time()
print(f"[INFO] Image download time: {t2 - t1:.2f} sec")

# Step 2: Simulate drone view
t1 = time.time()
image1_cv = simulate_drone_view(image1)
image2_cv = simulate_drone_view(image2)
t2 = time.time()
print(f"[INFO] Drone view simulation time: {t2 - t1:.2f} sec")

# Step 3: Rotate the second image
t1 = time.time()
image2_rotated = rotate_image(image2_cv, 15)
t2 = time.time()
print(f"[INFO] Image rotation time: {t2 - t1:.2f} sec")

# Step 4: Extract SIFT features
t1 = time.time()
kp1, des1 = extract_sift_features(image1_cv)
kp2, des2 = extract_sift_features(image2_rotated)
t2 = time.time()
print(f"[INFO] SIFT feature extraction time: {t2 - t1:.2f} sec")

# Step 5: Match SIFT features
t1 = time.time()
matches = match_sift_features(des1, des2)
t2 = time.time()
print(f"[INFO] Feature matching time: {t2 - t1:.2f} sec")

# Step 6: Estimate rotation & translation using RANSAC
t1 = time.time()
H, inliers = estimate_transform_ransac(kp1, kp2, matches)
t2 = time.time()
print(f"[INFO] RANSAC estimation time: {t2 - t1:.2f} sec")

# Step 7: Draw matches (inliers only)
match_img = cv2.drawMatches(image1_cv, kp1, image2_rotated, kp2, inliers[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imshow("SIFT Feature Matching (Inliers)", match_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Print results
print(f"\n[INFO] Keypoints in Image 1: {len(kp1)}")
print(f"[INFO] Keypoints in Image 2: {len(kp2)}")
print(f"[INFO] Good Matches: {len(matches)}")
print(f"[INFO] Inlier Matches: {len(inliers)}")

if H is not None:
    print("\nEstimated Homography (Rotation & Translation):\n", H)
    dx = H[0, 2]
    dy = H[1, 2]
    rotation_angle = np.degrees(np.arctan2(H[1, 0], H[0, 0]))
    print(f"Estimated Translation: Δx = {dx:.2f}, Δy = {dy:.2f}")
    print(f"Estimated Rotation: {rotation_angle:.2f}°")

# Final resource tracking
cpu_end, mem_end = track_resources()
total_time = time.time() - start_time
print(f"\n[INFO] Total execution time: {total_time:.2f} sec")
print(f"[INFO] CPU usage: {cpu_end - cpu_start:.2f}%")
print(f"[INFO] Memory usage: {mem_end - mem_start:.2f}%")
