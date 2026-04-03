"""Generate synthetic pop art test images that mimic the user's examples,
then run the coloring template pipeline on them."""

import cv2
import numpy as np
from pathlib import Path

OUTPUT = Path("/home/user/ColoringTemplateCreator/examples")
OUTPUT.mkdir(exist_ok=True)


def draw_pop_cat(path: Path):
    """Mimic the grumpy cat pop art: bold black outlines, flat color fills, text."""
    img = np.full((800, 600, 3), (230, 220, 210), dtype=np.uint8)  # beige bg

    # Colored background blocks (like the room behind the cat)
    cv2.rectangle(img, (50, 80), (550, 650), (200, 140, 120), -1)  # pink wall
    cv2.rectangle(img, (280, 80), (420, 650), (60, 180, 220), -1)  # yellow curtain
    cv2.rectangle(img, (420, 80), (550, 650), (40, 160, 200), -1)  # yellow-green

    # Cat body (white with black patches)
    cv2.ellipse(img, (300, 450), (120, 160), 0, 0, 360, (240, 240, 240), -1)
    cv2.ellipse(img, (300, 450), (120, 160), 0, 0, 360, (0, 0, 0), 3)  # outline

    # Cat head
    cv2.ellipse(img, (300, 320), (90, 75), 0, 0, 360, (240, 240, 240), -1)
    cv2.ellipse(img, (300, 320), (90, 75), 0, 0, 360, (0, 0, 0), 3)

    # Ears (triangles)
    pts_ear_l = np.array([[230, 260], [250, 310], [210, 310]], np.int32)
    pts_ear_r = np.array([[370, 260], [350, 310], [390, 310]], np.int32)
    cv2.fillPoly(img, [pts_ear_l], (20, 20, 20))
    cv2.fillPoly(img, [pts_ear_r], (20, 20, 20))
    cv2.polylines(img, [pts_ear_l, pts_ear_r], True, (0, 0, 0), 3)

    # Eyes (grumpy half-closed)
    cv2.ellipse(img, (270, 310), (20, 12), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(img, (330, 310), (20, 12), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(img, (270, 310), (20, 12), 0, 0, 360, (0, 0, 0), 2)
    cv2.ellipse(img, (330, 310), (20, 12), 0, 0, 360, (0, 0, 0), 2)
    cv2.circle(img, (275, 312), 6, (0, 100, 0), -1)
    cv2.circle(img, (335, 312), 6, (0, 100, 0), -1)

    # Nose + mouth
    pts_nose = np.array([[295, 335], (305, 335), (300, 342)], np.int32)
    cv2.fillPoly(img, [pts_nose], (180, 150, 160))
    cv2.polylines(img, [pts_nose], True, (0, 0, 0), 2)
    cv2.line(img, (300, 342), (300, 355), (0, 0, 0), 2)
    cv2.line(img, (300, 355), (280, 365), (0, 0, 0), 2)
    cv2.line(img, (300, 355), (320, 365), (0, 0, 0), 2)

    # Collar
    cv2.ellipse(img, (300, 390), (60, 15), 0, 0, 180, (180, 120, 0), -1)
    cv2.ellipse(img, (300, 390), (60, 15), 0, 0, 180, (0, 0, 0), 2)

    # Toilet paper roll (next to cat)
    cv2.ellipse(img, (430, 530), (30, 20), 0, 0, 360, (200, 160, 220), -1)
    cv2.ellipse(img, (430, 530), (30, 20), 0, 0, 360, (0, 0, 0), 2)
    cv2.rectangle(img, (415, 510), (445, 530), (200, 160, 220), -1)
    cv2.rectangle(img, (415, 510), (445, 530), (0, 0, 0), 2)

    # Text "ARE YOU POOPING?"
    cv2.putText(img, "ARE YOU", (140, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 4)
    cv2.putText(img, "POOPING?", (120, 750), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 4)

    # Border
    cv2.rectangle(img, (45, 75), (555, 700), (0, 0, 0), 3)

    cv2.imwrite(str(path), img)
    print(f"Created: {path}")


def draw_pool_ghost(path: Path):
    """Mimic the ghost by the pool: dark sky, simple white blob, clean outlines."""
    img = np.full((1200, 600, 3), (40, 45, 45), dtype=np.uint8)  # dark sky

    # Pool deck
    cv2.rectangle(img, (0, 600), (600, 1200), (180, 140, 150), -1)  # pink deck
    cv2.rectangle(img, (0, 600), (600, 1200), (0, 0, 0), 2)

    # Pool water
    pts_pool = np.array([[0, 750], [600, 750], [600, 1100], [0, 1100]], np.int32)
    cv2.fillPoly(img, [pts_pool], (180, 160, 100))  # blue water
    cv2.polylines(img, [pts_pool], True, (0, 0, 0), 3)

    # Water ripples
    cv2.ellipse(img, (300, 900), (80, 15), 0, 0, 360, (255, 255, 255), 2)
    cv2.ellipse(img, (150, 850), (40, 8), 0, 0, 360, (255, 255, 255), 2)

    # Pool ladder
    cv2.line(img, (80, 720), (80, 850), (200, 200, 200), 3)
    cv2.line(img, (110, 720), (110, 850), (200, 200, 200), 3)
    cv2.line(img, (80, 760), (110, 760), (200, 200, 200), 3)
    cv2.line(img, (80, 800), (110, 800), (200, 200, 200), 3)
    # Ladder outlines
    cv2.line(img, (80, 720), (80, 850), (0, 0, 0), 1)
    cv2.line(img, (110, 720), (110, 850), (0, 0, 0), 1)

    # Ghost character sitting at pool edge
    cv2.ellipse(img, (350, 680), (60, 90), 0, 0, 360, (240, 240, 240), -1)
    cv2.ellipse(img, (350, 680), (60, 90), 0, 0, 360, (0, 0, 0), 3)
    # Ghost eyes (dots)
    cv2.circle(img, (335, 660), 4, (0, 0, 0), -1)
    cv2.circle(img, (365, 660), 4, (0, 0, 0), -1)

    # Ghost legs in water
    cv2.line(img, (330, 770), (330, 810), (240, 240, 240), 8)
    cv2.line(img, (370, 770), (370, 810), (240, 240, 240), 8)
    cv2.line(img, (330, 770), (330, 810), (0, 0, 0), 2)
    cv2.line(img, (370, 770), (370, 810), (0, 0, 0), 2)

    # Lounge chair
    cv2.rectangle(img, (400, 550), (560, 600), (140, 100, 60), -1)  # wooden base
    cv2.rectangle(img, (400, 520), (560, 560), (160, 120, 80), -1)  # blue cushion
    cv2.rectangle(img, (400, 520), (560, 600), (0, 0, 0), 2)

    # Beach ball in water
    cv2.circle(img, (400, 920), 25, (80, 80, 220), -1)
    cv2.circle(img, (400, 920), 25, (0, 0, 0), 2)
    cv2.line(img, (375, 920), (425, 920), (0, 0, 0), 2)

    # Moon
    cv2.circle(img, (100, 150), 40, (50, 170, 230), -1)  # orange
    cv2.circle(img, (100, 150), 40, (0, 0, 0), 2)
    cv2.circle(img, (90, 145), 3, (0, 0, 0), -1)  # face dot
    cv2.circle(img, (110, 145), 3, (0, 0, 0), -1)

    # Bottles/drinks on deck
    cv2.rectangle(img, (420, 630), (435, 680), (40, 80, 120), -1)
    cv2.rectangle(img, (420, 630), (435, 680), (0, 0, 0), 2)
    cv2.rectangle(img, (445, 650), (460, 680), (60, 140, 180), -1)
    cv2.rectangle(img, (445, 650), (460, 680), (0, 0, 0), 2)

    cv2.imwrite(str(path), img)
    print(f"Created: {path}")


def draw_airport_ghost(path: Path):
    """Mimic the ghost at the airport: Gate 10 Delayed, sitting on bench."""
    # White border padding
    img = np.full((700, 700, 3), (255, 255, 255), dtype=np.uint8)

    # The scene is inset
    x0, y0, x1, y1 = 50, 100, 650, 650

    # Dark green/teal sky through window
    cv2.rectangle(img, (x0, y0), (x1, y1), (50, 70, 60), -1)

    # Floor
    cv2.rectangle(img, (x0, 400), (x1, y1), (140, 120, 100), -1)  # teal floor

    # Window struts (golden/amber)
    cv2.line(img, (250, y0), (300, 400), (40, 160, 200), 4)
    cv2.line(img, (450, y0), (500, 400), (40, 160, 200), 4)
    # Window sill
    cv2.line(img, (x0, 400), (x1, 400), (140, 100, 60), 4)

    # Runway through window
    pts_runway = np.array([[320, 250], [380, 250], [550, 400], [200, 400]], np.int32)
    cv2.fillPoly(img, [pts_runway], (80, 110, 90))
    cv2.polylines(img, [pts_runway], True, (0, 0, 0), 2)

    # Sunset/moon through window
    cv2.circle(img, (400, 300), 30, (50, 170, 230), -1)
    cv2.circle(img, (400, 300), 30, (0, 0, 0), 2)

    # Gate sign
    cv2.rectangle(img, (80, 200), (230, 260), (80, 50, 30), -1)
    cv2.rectangle(img, (80, 200), (230, 260), (0, 0, 0), 2)
    cv2.putText(img, "GATE 10", (90, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(img, "DELAYED", (90, 252), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 255), 1)

    # Bench
    cv2.rectangle(img, (280, 480), (450, 500), (180, 180, 180), -1)
    cv2.rectangle(img, (280, 480), (450, 500), (0, 0, 0), 2)
    # Bench legs
    cv2.line(img, (300, 500), (300, 540), (0, 0, 0), 2)
    cv2.line(img, (430, 500), (430, 540), (0, 0, 0), 2)

    # Ghost on bench
    cv2.ellipse(img, (360, 440), (45, 70), 0, 0, 360, (240, 240, 240), -1)
    cv2.ellipse(img, (360, 440), (45, 70), 0, 0, 360, (0, 0, 0), 3)
    cv2.circle(img, (348, 425), 3, (0, 0, 0), -1)
    cv2.circle(img, (372, 425), 3, (0, 0, 0), -1)

    # Suitcase
    cv2.rectangle(img, (460, 470), (520, 540), (50, 180, 220), -1)
    cv2.rectangle(img, (460, 470), (520, 540), (0, 0, 0), 2)
    cv2.line(img, (480, 470), (480, 455), (0, 0, 0), 2)
    cv2.line(img, (500, 470), (500, 455), (0, 0, 0), 2)
    cv2.line(img, (480, 455), (500, 455), (0, 0, 0), 2)

    # Food cart
    cv2.rectangle(img, (100, 420), (200, 460), (140, 140, 140), -1)
    cv2.rectangle(img, (100, 420), (200, 460), (0, 0, 0), 2)
    cv2.circle(img, (120, 465), 8, (0, 0, 0), 2)
    cv2.circle(img, (180, 465), 8, (0, 0, 0), 2)

    # Scene border
    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 0), 3)

    # Shadow under bench
    cv2.ellipse(img, (365, 545), (80, 15), 0, 0, 360, (30, 30, 30), -1)

    cv2.imwrite(str(path), img)
    print(f"Created: {path}")


if __name__ == "__main__":
    draw_pop_cat(OUTPUT / "pop_cat.png")
    draw_pool_ghost(OUTPUT / "pool_ghost.png")
    draw_airport_ghost(OUTPUT / "airport_ghost.png")
    print("\nAll test images created.")
