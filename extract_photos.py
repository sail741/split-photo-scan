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


def imread_unicode(path):
    """Read an image from a path that may contain non-ASCII characters (Windows fix)."""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path, image, params=None):
    """Write an image to a path that may contain non-ASCII characters (Windows fix)."""
    ext = Path(path).suffix
    success, buf = cv2.imencode(ext, image, params or [])
    if success:
        buf.tofile(str(path))


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
    Two-pass autocrop:
    1) Contour-based: removes white triangles in corners from rotation
    2) Gradient-based: fine-tunes each edge to the actual photo boundary
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # --- Pass 1: contour-based crop (handles corners and margins) ---
    _, white_mask = cv2.threshold(blurred, 210, 255, cv2.THRESH_BINARY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    _, sat_mask = cv2.threshold(sat, 15, 255, cv2.THRESH_BINARY)
    content = cv2.bitwise_or(cv2.bitwise_not(white_mask), sat_mask)
    # Light close to fill noise but not extend into margins
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    content = cv2.morphologyEx(content, cv2.MORPH_CLOSE, kernel, iterations=2)
    # Erode to pull boundary inward past interpolated edge pixels
    content = cv2.erode(content, kernel, iterations=2)

    contours, _ = cv2.findContours(content, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, cw, ch = cv2.boundingRect(largest)
        if cw >= w * 0.5 and ch >= h * 0.5:
            image = image[y:y+ch, x:x+cw]

    # --- Pass 1b: trim rows/columns that are mostly white ---
    h, w = image.shape[:2]
    gray2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    white_ratio = 0.5  # if >50% of a row/col is white, trim it
    top, bottom, left, right = 0, h, 0, w
    max_trim = int(min(h, w) * 0.08)
    for i in range(max_trim):
        if np.mean(gray2[top, left:right] > 220) > white_ratio:
            top += 1
        else:
            break
    for i in range(max_trim):
        if np.mean(gray2[bottom - 1, left:right] > 220) > white_ratio:
            bottom -= 1
        else:
            break
    for i in range(max_trim):
        if np.mean(gray2[top:bottom, left] > 220) > white_ratio:
            left += 1
        else:
            break
    for i in range(max_trim):
        if np.mean(gray2[top:bottom, right - 1] > 220) > white_ratio:
            right -= 1
        else:
            break
    if (right - left) >= w * 0.5 and (bottom - top) >= h * 0.5:
        image = image[top:bottom, left:right]

# --- Pass 2: gradient-based edge refinement ---
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(grad_x ** 2 + grad_y ** 2)

    search_h = max(int(h * 0.15), 10)
    search_w = max(int(w * 0.15), 10)
    global_mean = gradient.mean()
    edge_threshold = global_mean * 0.5

    top = 0
    row_means = gradient[:search_h, :].mean(axis=1)
    peak = int(np.argmax(row_means))
    if row_means[peak] > edge_threshold:
        top = peak + 2

    bottom = h
    row_means = gradient[h - search_h:, :].mean(axis=1)
    peak = int(np.argmax(row_means[::-1]))
    if row_means[search_h - 1 - peak] > edge_threshold:
        bottom = h - peak - 2

    left = 0
    col_means = gradient[:, :search_w].mean(axis=0)
    peak = int(np.argmax(col_means))
    if col_means[peak] > edge_threshold:
        left = peak + 2

    right = w
    col_means = gradient[:, w - search_w:].mean(axis=0)
    peak = int(np.argmax(col_means[::-1]))
    if col_means[search_w - 1 - peak] > edge_threshold:
        right = w - peak - 2

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
    max_area = total_area * 0.85

    # Downscale for detection (work at ~1800px max dimension)
    max_detect = 1800
    detect_scale = 1.0
    if max(h, w) > max_detect:
        detect_scale = max_detect / max(h, w)
    dh, dw = int(h * detect_scale), int(w * detect_scale)
    small = cv2.resize(image, (dw, dh), interpolation=cv2.INTER_AREA)

    # Add a white border so photos touching edges are fully enclosed
    border = 10
    bordered = cv2.copyMakeBorder(small, border, border, border, border,
                                  cv2.BORDER_CONSTANT, value=(255, 255, 255))

    gray = cv2.cvtColor(bordered, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect white background
    _, bg_mask = cv2.threshold(blurred, 220, 255, cv2.THRESH_BINARY)

    # Saturation channel: photos have color, white background doesn't
    hsv = cv2.cvtColor(bordered, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    _, sat_mask = cv2.threshold(sat, 20, 255, cv2.THRESH_BINARY)

    # Foreground = not white OR has color
    fg_mask = cv2.bitwise_or(cv2.bitwise_not(bg_mask), sat_mask)

    # Minimal close: fill pixel-level noise only, preserve inter-photo gaps
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    # Erode to widen gaps and separate close photos, then dilate back
    kernel_sep = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    fg_mask = cv2.erode(fg_mask, kernel_sep, iterations=4)
    fg_mask = cv2.dilate(fg_mask, kernel_sep, iterations=4)

    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Try to split oversized contours using separator line detection
    # (for photos touching each other with almost no gap)
    split_contours = []
    oversized_threshold = 0.4  # contours covering >40% of image are suspicious
    for c in contours:
        c_area = cv2.contourArea(c)
        bh_d, bw_d = fg_mask.shape
        if c_area > bh_d * bw_d * oversized_threshold:
            # Create mask for just this contour region
            c_mask = np.zeros_like(fg_mask)
            cv2.drawContours(c_mask, [c], 0, 255, -1)
            region = cv2.bitwise_and(c_mask, cv2.bitwise_not(fg_mask))
            # Find white channels inside this region using line detection
            line_len = max(dh, dw) // 12
            h_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (1, line_len))
            v_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (line_len, 1))
            # Use a lower threshold for oversized regions (catch shadowed gaps)
            _, inner_bg = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY)
            inner_bg = cv2.bitwise_and(inner_bg, c_mask)
            h_lines = cv2.morphologyEx(inner_bg, cv2.MORPH_OPEN, h_kern)
            v_lines = cv2.morphologyEx(inner_bg, cv2.MORPH_OPEN, v_kern)
            seps = cv2.bitwise_or(h_lines, v_lines)
            widen_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            seps = cv2.dilate(seps, widen_k, iterations=2)
            # Cut the contour with separators and find sub-contours
            split_mask = cv2.bitwise_and(fg_mask, cv2.bitwise_not(seps))
            split_mask = cv2.bitwise_and(split_mask, c_mask)
            sub_contours, _ = cv2.findContours(split_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(sub_contours) > 1:
                split_contours.extend(sub_contours)
            else:
                split_contours.append(c)
        else:
            split_contours.append(c)
    contours = split_contours

    # Scale contours back to original image coordinates
    contours = [((c - border) / detect_scale).astype(np.int32) for c in contours]

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
        if aspect > 4:
            continue

        # Filter out fragments: smallest side must be >= 15% of scan's smaller dim
        min_dim = min(rect_w, rect_h)
        if min_dim < min(h, w) * 0.15:
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
        imwrite_unicode(debug_path, debug_img)
        print(f"Debug image saved: {debug_path}")

    return photos


def process_single(input_path, output_dir, ext, min_area_ratio, debug):
    """Process a single scan image. Returns the number of photos extracted."""
    image = imread_unicode(input_path)
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
        imwrite_unicode(out_path, photo, params)
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
