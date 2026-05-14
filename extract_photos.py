#!/usr/bin/env python3
"""
Extract individual photos from an A4 scanned image.
Detects rectangular photos (even rotated/diagonal) and saves them as separate files.

Usage:
    python extract_photos.py <scan_image> [--output-dir OUTPUT_DIR] [--min-area-ratio MIN] [--debug]

Example:
    python extract_photos.py scan.jpg --output-dir extracted/
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def order_points(pts):
    """Order points as: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect


def four_point_transform(image, pts):
    """Apply a perspective transform to extract and straighten a quadrilateral region."""
    rect = order_points(pts)
    tl, tr, br, bl = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = int(max(width_a, width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = int(max(height_a, height_b))

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1],
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (max_width, max_height))


def autocrop_to_content(image):
    """
    Crop to the actual photo content by detecting strong edges near each border.
    Scans inward from each side to find the first row/column with a significant
    gradient transition (= the real photo boundary).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Compute gradient magnitude (Sobel)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # Only search within the first/last 15% of each dimension
    search_h = max(int(h * 0.15), 10)
    search_w = max(int(w * 0.15), 10)

    # Threshold: a row/column is "content" if its mean gradient is above this
    # Use a fraction of the image's overall mean gradient
    global_mean = gradient.mean()
    edge_threshold = global_mean * 0.5

    # Find the peak gradient row/col near each edge, then crop just past it

    # Scan from top: find the row with the strongest gradient, crop past it
    top = 0
    row_means = gradient[:search_h, :].mean(axis=1)
    peak = int(np.argmax(row_means))
    if row_means[peak] > edge_threshold:
        top = peak + 2

    # Scan from bottom
    bottom = h
    row_means = gradient[h - search_h:, :].mean(axis=1)
    peak = int(np.argmax(row_means[::-1]))
    if row_means[search_h - 1 - peak] > edge_threshold:
        bottom = h - peak - 2

    # Scan from left
    left = 0
    col_means = gradient[:, :search_w].mean(axis=0)
    peak = int(np.argmax(col_means))
    if col_means[peak] > edge_threshold:
        left = peak + 2

    # Scan from right
    right = w
    col_means = gradient[:, w - search_w:].mean(axis=0)
    peak = int(np.argmax(col_means[::-1]))
    if col_means[search_w - 1 - peak] > edge_threshold:
        right = w - peak - 2

    # Sanity check: cropped region must be at least 50% of original
    if (right - left) < w * 0.5 or (bottom - top) < h * 0.5:
        return image

    return image[top:bottom, left:right]


def detect_photos(image, min_area_ratio=0.01, debug=False):
    """
    Detect rectangular photo regions in a scanned image.

    Args:
        image: BGR image (the full scan).
        min_area_ratio: Minimum area of a detected region relative to the full image.
        debug: If True, save a debug image with detected contours.

    Returns:
        List of extracted photo images.
    """
    h, w = image.shape[:2]
    total_area = h * w
    min_area = total_area * min_area_ratio
    # Photos shouldn't take up almost the entire scan
    max_area = total_area * 0.85

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use adaptive threshold + Canny for robust edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Try multiple approaches and combine edges
    # 1) Canny edge detection
    edges_canny = cv2.Canny(blurred, 30, 100)

    # 2) Adaptive threshold to catch low-contrast boundaries
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5
    )

    # Combine both edge maps
    edges = cv2.bitwise_or(edges_canny, thresh)

    # Dilate to close gaps in edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges = cv2.erode(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    photos = []
    boxes_used = []

    if debug:
        debug_img = image.copy()

    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue

        # Get the minimum area bounding rectangle (handles rotation)
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int32(box)

        rect_w, rect_h = rect[1]
        if rect_w == 0 or rect_h == 0:
            continue

        # Filter by aspect ratio — photos are roughly rectangular, not thin strips
        aspect = max(rect_w, rect_h) / min(rect_w, rect_h)
        if aspect > 5:
            continue

        # Check that the contour fills a reasonable portion of its bounding rect
        rect_area = rect_w * rect_h
        fill_ratio = area / rect_area if rect_area > 0 else 0
        if fill_ratio < 0.4:
            continue

        # Skip if this region overlaps significantly with an already-detected photo
        overlap = False
        for prev_box in boxes_used:
            # Simple overlap check via bounding rect intersection
            br1 = cv2.boundingRect(box)
            br2 = cv2.boundingRect(prev_box)
            x_overlap = max(0, min(br1[0] + br1[2], br2[0] + br2[2]) - max(br1[0], br2[0]))
            y_overlap = max(0, min(br1[1] + br1[3], br2[1] + br2[3]) - max(br1[1], br2[1]))
            overlap_area = x_overlap * y_overlap
            smaller_area = min(br1[2] * br1[3], br2[2] * br2[3])
            if smaller_area > 0 and overlap_area / smaller_area > 0.5:
                overlap = True
                break
        if overlap:
            continue

        # Extract the photo using perspective transform
        photo = four_point_transform(image, box.astype("float32"))

        # Crop to actual photo content using edge detection
        photo = autocrop_to_content(photo)

        # Skip very small resulting images
        ph, pw = photo.shape[:2]
        if ph < 50 or pw < 50:
            continue

        photos.append(photo)
        boxes_used.append(box)

        if debug:
            cv2.drawContours(debug_img, [box], 0, (0, 255, 0), 3)
            cx, cy = int(rect[0][0]), int(rect[0][1])
            cv2.putText(debug_img, f"#{len(photos)}", (cx - 20, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    if debug:
        debug_path = "debug_detection.jpg"
        cv2.imwrite(debug_path, debug_img)
        print(f"Debug image saved: {debug_path}")

    return photos


def process_single(input_path, output_dir, ext, min_area_ratio, debug):
    """Process a single scan image. Returns the number of photos extracted."""
    image = cv2.imread(str(input_path))
    if image is None:
        print(f"  Skipping (unreadable): {input_path}", file=sys.stderr)
        return 0

    print(f"Processing: {input_path} ({image.shape[1]}x{image.shape[0]})")

    photos = detect_photos(image, min_area_ratio=min_area_ratio, debug=debug)

    if not photos:
        print("  No photos detected.")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem

    for i, photo in enumerate(photos, start=1):
        out_path = output_dir / f"{stem}_photo_{i}.{ext}"
        params = []
        if ext == "jpg":
            params = [cv2.IMWRITE_JPEG_QUALITY, 95]
        elif ext == "png":
            params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
        cv2.imwrite(str(out_path), photo, params)
        ph, pw = photo.shape[:2]
        print(f"  Saved: {out_path} ({pw}x{ph})")

    return len(photos)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def main():
    parser = argparse.ArgumentParser(description="Extract photos from scanned A4 image(s).")
    parser.add_argument("input", help="Path to a scanned image or a directory of scans")
    parser.add_argument("--output-dir", "-o", default="extracted",
                        help="Output directory for extracted photos (default: extracted/)")
    parser.add_argument("--min-area-ratio", type=float, default=0.01,
                        help="Minimum photo area as ratio of total scan area (default: 0.01)")
    parser.add_argument("--format", choices=["jpg", "png"], default="jpg",
                        help="Output format (default: jpg)")
    parser.add_argument("--debug", action="store_true",
                        help="Save a debug image showing detected regions")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: path not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)

    if input_path.is_file():
        files = [input_path]
    else:
        files = sorted(
            f for f in input_path.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not files:
            print(f"No image files found in {input_path}", file=sys.stderr)
            sys.exit(1)
        print(f"Found {len(files)} image(s) in {input_path}\n")

    total_photos = 0
    for f in files:
        count = process_single(f, output_dir, args.format, args.min_area_ratio, args.debug)
        total_photos += count
        print()

    print(f"Done! {total_photos} photo(s) extracted from {len(files)} scan(s) to {output_dir}/")


if __name__ == "__main__":
    main()
