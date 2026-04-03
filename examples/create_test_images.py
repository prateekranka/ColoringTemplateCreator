"""Create realistic pop art test images matching the user's three example styles:

1. Geometric collage dog with sunglasses - textured fills, minimal outlines
2. Pop art woman portrait - bold black outlines, halftone dot patterns
3. Abstract Matisse-style - no outlines, just overlapping color shapes

These are designed to stress-test the extraction pipeline with varying
outline styles, textures, and color approaches.
"""

import cv2
import numpy as np
from pathlib import Path

OUTPUT = Path(__file__).parent
OUTPUT.mkdir(exist_ok=True)


def add_texture(img, region_mask, noise_scale=30):
    """Add subtle texture/noise to a region to simulate paper/print texture."""
    noise = np.random.randint(-noise_scale, noise_scale, img.shape, dtype=np.int16)
    textured = img.astype(np.int16) + noise
    textured = np.clip(textured, 0, 255).astype(np.uint8)
    img[region_mask > 0] = textured[region_mask > 0]


def add_halftone(img, region_mask, dot_spacing=8, dot_radius=2):
    """Add halftone dot pattern to simulate comic print style."""
    h, w = img.shape[:2]
    for y in range(0, h, dot_spacing):
        for x in range(0, w, dot_spacing):
            if 0 <= y < h and 0 <= x < w and region_mask[y, x] > 0:
                brightness = np.mean(img[y, x])
                r = max(1, int(dot_radius * (1 - brightness / 255)))
                cv2.circle(img, (x, y), r, (180, 100, 100), -1)


def draw_geometric_dog(path: Path):
    """Style 1: Geometric collage dog with sunglasses.
    Textured colored patches, wavy patterns on ears, minimal outlines.
    Similar to the dachshund illustration the user provided."""
    h, w = 1200, 1000
    img = np.full((h, w, 3), 255, dtype=np.uint8)  # white background

    # Dog face - golden/amber color with texture
    face_pts = np.array([
        [350, 200], [650, 200], [700, 400], [680, 600],
        [600, 650], [400, 650], [320, 600], [300, 400]
    ], np.int32)
    face_mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(face_mask, [face_pts], 255)
    cv2.fillPoly(img, [face_pts], (60, 180, 220))  # golden/amber in BGR
    add_texture(img, face_mask, 15)

    # Nose bridge - darker golden
    nose_pts = np.array([
        [430, 280], [570, 280], [550, 500], [480, 550], [450, 500]
    ], np.int32)
    cv2.fillPoly(img, [nose_pts], (40, 160, 200))

    # Big black nose
    nose_center = (500, 530)
    cv2.ellipse(img, nose_center, (45, 35), 0, 0, 360, (15, 15, 15), -1)

    # Left ear - dark with wavy texture pattern
    ear_l_pts = np.array([
        [200, 200], [350, 150], [370, 250], [350, 500],
        [300, 600], [180, 550], [150, 400]
    ], np.int32)
    cv2.fillPoly(img, [ear_l_pts], (50, 55, 55))
    # Wavy lines on ear
    for y_off in range(0, 400, 12):
        y = 200 + y_off
        pts = []
        for x in range(160, 360, 4):
            wave_y = y + int(6 * np.sin(x * 0.08 + y_off * 0.1))
            pts.append([x, wave_y])
        if len(pts) > 1:
            pts_arr = np.array(pts, np.int32)
            mask_check = np.zeros((h, w), np.uint8)
            cv2.fillPoly(mask_check, [ear_l_pts], 255)
            cv2.polylines(img, [pts_arr], False, (80, 90, 90), 1)

    # Right ear
    ear_r_pts = np.array([
        [650, 150], [800, 200], [850, 400], [820, 550],
        [700, 600], [650, 500], [630, 250]
    ], np.int32)
    cv2.fillPoly(img, [ear_r_pts], (50, 55, 55))
    for y_off in range(0, 400, 12):
        y = 200 + y_off
        pts = []
        for x in range(640, 840, 4):
            wave_y = y + int(6 * np.sin(x * 0.08 + y_off * 0.1))
            pts.append([x, wave_y])
        if len(pts) > 1:
            pts_arr = np.array(pts, np.int32)
            cv2.polylines(img, [pts_arr], False, (80, 90, 90), 1)

    # Sunglasses - red frames with dark lenses
    # Left lens
    cv2.ellipse(img, (380, 310), (100, 70), -5, 0, 360, (40, 40, 200), -1)  # red frame
    cv2.ellipse(img, (380, 310), (80, 55), -5, 0, 360, (20, 20, 20), -1)    # dark lens
    # Right lens
    cv2.ellipse(img, (620, 310), (100, 70), 5, 0, 360, (40, 40, 200), -1)
    cv2.ellipse(img, (620, 310), (80, 55), 5, 0, 360, (20, 20, 20), -1)
    # Bridge
    cv2.line(img, (480, 300), (520, 300), (40, 40, 200), 8)
    # Frame outline
    cv2.ellipse(img, (380, 310), (100, 70), -5, 0, 360, (15, 15, 15), 3)
    cv2.ellipse(img, (620, 310), (100, 70), 5, 0, 360, (15, 15, 15), 3)

    # Sweater/body - colorful patchwork
    body_top = 650
    # Pink turtleneck
    cv2.rectangle(img, (350, body_top), (650, body_top + 80), (150, 130, 200), -1)
    # Vertical ribs on turtleneck
    for x in range(355, 650, 8):
        cv2.line(img, (x, body_top), (x, body_top + 80), (130, 110, 180), 1)

    # Patchwork sweater body
    patches = [
        ((250, body_top + 80), (400, body_top + 250), (200, 180, 60)),    # blue patch
        ((400, body_top + 80), (550, body_top + 250), (50, 190, 230)),    # yellow patch
        ((550, body_top + 80), (750, body_top + 250), (50, 50, 50)),      # dark patch
        ((250, body_top + 250), (500, h), (40, 40, 180)),                  # red patch
        ((500, body_top + 250), (750, h), (180, 200, 70)),                # teal patch
    ]
    for (x1, y1), (x2, y2), color in patches:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
        mask = np.zeros((h, w), np.uint8)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        add_texture(img, mask, 20)

    # Horizontal stripes on blue patch
    for y in range(body_top + 85, body_top + 250, 10):
        cv2.line(img, (250, y), (400, y), (220, 200, 80), 2)

    # Bow ties on dark patch
    for by in range(body_top + 110, body_top + 240, 50):
        for bx in range(580, 730, 50):
            pts = np.array([[bx, by], [bx+15, by-10], [bx+15, by+10]], np.int32)
            cv2.fillPoly(img, [pts], (200, 200, 200))
            pts2 = np.array([[bx+15, by], [bx+30, by-10], [bx+30, by+10]], np.int32)
            cv2.fillPoly(img, [pts2], (200, 200, 200))

    # Chevron pattern on teal area
    for y in range(body_top + 260, h, 20):
        for x in range(500, 750, 30):
            pts = np.array([[x, y], [x+15, y-10], [x+30, y]], np.int32)
            cv2.polylines(img, [pts], False, (160, 180, 50), 2)

    cv2.imwrite(str(path), img)
    print(f"Created: {path} ({w}x{h})")


def draw_popart_woman(path: Path):
    """Style 2: Pop art woman portrait with bold black outlines and halftone.
    Similar to the Lichtenstein-style woman the user provided."""
    h, w = 1200, 800
    # Blue background
    img = np.full((h, w, 3), (180, 60, 20), dtype=np.uint8)  # deep blue BGR

    # Skin area - peach with halftone dots
    face_pts = np.array([
        [250, 300], [300, 200], [400, 150], [500, 200], [550, 300],
        [560, 450], [540, 550], [520, 650], [500, 750],
        [400, 800], [300, 750], [280, 650], [260, 550], [240, 450]
    ], np.int32)
    face_mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(face_mask, [face_pts], 255)
    cv2.fillPoly(img, [face_pts], (190, 200, 240))  # peach skin
    add_halftone(img, face_mask, 6, 2)

    # Neck
    neck_pts = np.array([
        [350, 780], [450, 780], [470, 950], [430, 1000], [370, 1000], [330, 950]
    ], np.int32)
    neck_mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(neck_mask, [neck_pts], 255)
    cv2.fillPoly(img, [neck_pts], (190, 200, 240))
    add_halftone(img, neck_mask, 6, 2)

    # Hair - golden/yellow
    # Left bun
    cv2.circle(img, (250, 180), 90, (30, 170, 230), -1)
    cv2.circle(img, (250, 180), 90, (10, 10, 10), 4)
    # Right bun
    cv2.circle(img, (550, 180), 90, (30, 170, 230), -1)
    cv2.circle(img, (550, 180), 90, (10, 10, 10), 4)
    # Hair top
    hair_top = np.array([
        [250, 250], [270, 120], [350, 80], [400, 70], [450, 80],
        [530, 120], [550, 250], [500, 200], [400, 170], [300, 200]
    ], np.int32)
    cv2.fillPoly(img, [hair_top], (30, 170, 230))
    cv2.polylines(img, [hair_top], True, (10, 10, 10), 4)

    # Hair strands (curved lines within hair)
    for i in range(8):
        x_start = 280 + i * 30
        pts = []
        for t in range(20):
            x = x_start + int(10 * np.sin(t * 0.5 + i))
            y = 100 + t * 8
            pts.append([x, y])
        if pts:
            cv2.polylines(img, [np.array(pts, np.int32)], False, (20, 140, 190), 2)

    # Eyebrows - thick black
    brow_l = np.array([[300, 330], [380, 310], [385, 320], [310, 345]], np.int32)
    brow_r = np.array([[420, 310], [500, 330], [495, 345], [425, 320]], np.int32)
    cv2.fillPoly(img, [brow_l], (10, 10, 10))
    cv2.fillPoly(img, [brow_r], (10, 10, 10))

    # Eyes
    # Eye whites
    cv2.ellipse(img, (345, 380), (35, 20), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(img, (455, 380), (35, 20), 0, 0, 360, (255, 255, 255), -1)
    # Irises - blue
    cv2.circle(img, (345, 380), 12, (200, 80, 30), -1)
    cv2.circle(img, (455, 380), 12, (200, 80, 30), -1)
    # Pupils
    cv2.circle(img, (345, 380), 5, (10, 10, 10), -1)
    cv2.circle(img, (455, 380), 5, (10, 10, 10), -1)
    # Eye outlines
    cv2.ellipse(img, (345, 380), (35, 20), 0, 0, 360, (10, 10, 10), 3)
    cv2.ellipse(img, (455, 380), (35, 20), 0, 0, 360, (10, 10, 10), 3)
    # Eyeliner wings
    cv2.line(img, (310, 375), (295, 365), (10, 10, 10), 3)
    cv2.line(img, (490, 375), (505, 365), (10, 10, 10), 3)

    # Eye shadow - red/orange above eyes
    for eye_x in [345, 455]:
        shadow_pts = np.array([
            [eye_x - 40, 370], [eye_x - 30, 350], [eye_x, 340],
            [eye_x + 30, 350], [eye_x + 40, 370]
        ], np.int32)
        cv2.fillPoly(img, [shadow_pts], (50, 80, 220))

    # Nose
    cv2.line(img, (400, 420), (395, 500), (10, 10, 10), 2)
    cv2.line(img, (395, 500), (385, 510), (10, 10, 10), 2)
    cv2.line(img, (395, 500), (410, 510), (10, 10, 10), 2)

    # Lips - red
    lip_upper = np.array([
        [360, 580], [380, 560], [400, 570], [420, 560], [440, 580],
        [400, 590]
    ], np.int32)
    lip_lower = np.array([
        [360, 580], [400, 620], [440, 580], [400, 590]
    ], np.int32)
    cv2.fillPoly(img, [lip_upper], (30, 30, 210))
    cv2.fillPoly(img, [lip_lower], (30, 30, 210))
    cv2.polylines(img, [lip_upper], True, (10, 10, 10), 3)
    cv2.polylines(img, [lip_lower], True, (10, 10, 10), 3)

    # Earrings - gold geometric trapezoids
    # Left earring
    earring_l = np.array([[240, 500], [210, 550], [220, 650], [260, 650], [270, 550]], np.int32)
    cv2.fillPoly(img, [earring_l], (20, 160, 210))
    cv2.polylines(img, [earring_l], True, (10, 10, 10), 4)
    # Earring hole
    inner_l = np.array([[238, 540], [225, 570], [230, 620], [255, 620], [258, 570]], np.int32)
    cv2.fillPoly(img, [inner_l], (180, 60, 20))  # blue = background shows through
    cv2.polylines(img, [inner_l], True, (10, 10, 10), 3)

    # Right earring
    earring_r = np.array([[560, 500], [590, 550], [580, 650], [540, 650], [530, 550]], np.int32)
    cv2.fillPoly(img, [earring_r], (20, 160, 210))
    cv2.polylines(img, [earring_r], True, (10, 10, 10), 4)
    inner_r = np.array([[562, 540], [575, 570], [570, 620], [545, 620], [542, 570]], np.int32)
    cv2.fillPoly(img, [inner_r], (180, 60, 20))
    cv2.polylines(img, [inner_r], True, (10, 10, 10), 3)

    # Face outline - bold black
    cv2.polylines(img, [face_pts], True, (10, 10, 10), 5)
    cv2.polylines(img, [neck_pts], True, (10, 10, 10), 4)

    # Choker necklace
    cv2.line(img, (330, 790), (470, 790), (10, 10, 10), 6)

    # Red top/shirt at bottom
    shirt_pts = np.array([
        [200, 950], [330, 900], [470, 900], [600, 950],
        [650, 1200], [150, 1200]
    ], np.int32)
    cv2.fillPoly(img, [shirt_pts], (30, 30, 210))
    cv2.polylines(img, [shirt_pts], True, (10, 10, 10), 4)

    cv2.imwrite(str(path), img)
    print(f"Created: {path} ({w}x{h})")


def draw_abstract_matisse(path: Path):
    """Style 3: Abstract Matisse-style - overlapping color shapes, NO outlines.
    The hardest case for our pipeline: relies entirely on edge detection
    since there are no drawn black outlines."""
    h, w = 1200, 800
    img = np.full((h, w, 3), (220, 230, 240), dtype=np.uint8)  # warm off-white

    shapes = [
        # Background large shapes
        (np.array([[0, 0], [400, 0], [350, 600], [0, 500]]),
         (200, 140, 50)),   # blue
        (np.array([[300, 0], [800, 0], [800, 400], [400, 350]]),
         (30, 120, 220)),   # orange-red
        (np.array([[0, 400], [300, 300], [350, 700], [100, 800]]),
         (100, 180, 50)),   # teal-green
        (np.array([[400, 200], [800, 100], [800, 600], [500, 700]]),
         (40, 200, 240)),   # golden yellow

        # Mid-ground shapes
        (np.array([[200, 500], [450, 400], [500, 800], [250, 900]]),
         (150, 60, 130)),   # purple
        (np.array([[100, 700], [350, 650], [400, 1000], [150, 1100], [50, 950]]),
         (80, 170, 50)),    # green
        (np.array([[450, 600], [700, 500], [750, 900], [600, 1000], [400, 850]]),
         (20, 50, 200)),    # red

        # Abstract woman silhouette - dark hair
        (np.array([
            [300, 200], [350, 100], [400, 80], [450, 100], [480, 200],
            [470, 350], [450, 500], [420, 600], [380, 650],
            [340, 600], [320, 500], [310, 350]
        ]), (40, 35, 30)),  # near-black hair

        # Face/skin area
        (np.array([
            [350, 200], [420, 180], [440, 250], [450, 350],
            [430, 420], [400, 450], [370, 430], [350, 370], [340, 280]
        ]), (120, 140, 220)),  # warm skin tone

        # Lips
        (np.array([[370, 390], [400, 380], [420, 395], [400, 410], [380, 405]]),
         (130, 100, 200)),  # pink lips

        # Earring circle
    ]

    for pts, color in shapes:
        cv2.fillPoly(img, [pts], color)
        # Add brush texture
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        add_texture(img, mask, 25)

    # Earring - white circle
    cv2.circle(img, (330, 430), 20, (240, 240, 240), -1)

    # Leaf shapes on right side
    for i, (cx, cy, angle) in enumerate([(650, 300, 30), (700, 200, -20), (680, 400, 45)]):
        leaf_pts = []
        for t in np.linspace(0, 2 * np.pi, 30):
            r = 60 * (1 + 0.3 * np.cos(2 * t))
            x = int(cx + r * np.cos(t + np.radians(angle)))
            y = int(cy + r * 0.4 * np.sin(t + np.radians(angle)))
            leaf_pts.append([x, y])
        leaf_arr = np.array(leaf_pts, np.int32)
        cv2.fillPoly(img, [leaf_arr], (120, 130, 30))  # dark teal leaves

    # More bottom shapes
    cv2.ellipse(img, (600, 800), (100, 80), 20, 0, 360, (60, 200, 240), -1)  # yellow oval
    cv2.ellipse(img, (200, 1050), (120, 100), -10, 0, 360, (140, 100, 200), -1)  # pink

    # Small white circles (decorative)
    cv2.circle(img, (550, 650), 30, (240, 240, 240), -1)
    cv2.circle(img, (450, 250), 15, (240, 240, 240), -1)

    # Add overall brush-stroke texture effect
    for _ in range(200):
        x1 = np.random.randint(0, w)
        y1 = np.random.randint(0, h)
        length = np.random.randint(10, 40)
        angle = np.random.uniform(0, np.pi)
        x2 = int(x1 + length * np.cos(angle))
        y2 = int(y1 + length * np.sin(angle))
        color_var = img[min(y1, h-1), min(x1, w-1)].tolist()
        color_var = [max(0, min(255, c + np.random.randint(-20, 20))) for c in color_var]
        cv2.line(img, (x1, y1), (x2, y2), color_var, 1)

    cv2.imwrite(str(path), img)
    print(f"Created: {path} ({w}x{h})")


if __name__ == "__main__":
    draw_geometric_dog(OUTPUT / "geometric_dog.png")
    draw_popart_woman(OUTPUT / "popart_woman.png")
    draw_abstract_matisse(OUTPUT / "abstract_matisse.png")
    print("\nAll test images created.")
