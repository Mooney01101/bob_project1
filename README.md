# bob_project1
import cv2
import numpy as np
import time
import threading
import traceback

from PIL import Image
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

from nanoowl.tree import Tree
from nanoowl.tree_predictor import TreePredictor
from nanoowl.owl_predictor import OwlPredictor


# ============================================================
# CONFIGURATION
# ============================================================

ENGINE_PATH = "/opt/nanoowl/data/owl_image_encoder_patch32.engine"

CAMERA_DEVICES = [0, 1, 2]

SERVER_HOST = "0.0.0.0"
SERVER_PORT = 7860

NANOOWL_THRESHOLD = 0.08

JPEG_QUALITY = 80


# ============================================================
# MOTION CONFIGURATION
# ============================================================

MOTION_THRESHOLD = 4          # Pen-specific center-shift pixel threshold
STATIONARY_REQUIRED = 8       # Consecutive stationary frames needed
SCENE_MOTION_THRESHOLD = 1.0  # Motion threshold considered as "Zero Motion"


# ============================================================
# NANOOWL SETUP
# ============================================================

print()
print("=" * 60)
print(" NANOOWL PEN DROP DETECTOR")
print("=" * 60)
print()

print("Loading NanoOWL...")
print("Engine:", ENGINE_PATH)

try:

    owl_predictor = OwlPredictor(
        image_encoder_engine=ENGINE_PATH
    )

    predictor = TreePredictor(
        owl_predictor=owl_predictor
    )

except Exception:

    print()
    print("❌ FAILED TO LOAD NANOOWL")
    print()

    traceback.print_exc()

    raise

print("✅ NanoOWL loaded successfully.")
print()


# ============================================================
# PROMPT
# ============================================================

PROMPT = "[a pen / a stylus / a marker / a ball] [a hand] [a floor]"

tree = Tree.from_prompt(PROMPT)

print("NanoOWL prompt:")
print(PROMPT)
print()

print("Tree labels:")

for index, label in enumerate(tree.labels):

    print(
        f"  {index}: {label}"
    )

print()


# ============================================================
# SEQUENCE TRACKING
# ============================================================

sequence_steps = [
    "Object in hand",
    "Object falling / floating",
    "Object on floor (Motion Zero)"
]

completed_steps = [
    False,
    False,
    False
]

current_step = 0

sequence_lock = threading.Lock()


# ============================================================
# MOTION TRACKING (PEN & CAMERA SCENE)
# ============================================================

last_pen_center = None
stationary_frames = 0
last_gray_frame = None  # Used for frame-differencing scene motion

motion_lock = threading.Lock()


# ============================================================
# CAMERA SETUP
# ============================================================

print("Opening camera...")

camera_lock = threading.Lock()


def get_camera():

    """Try opening available video devices using V4L2."""

    for dev in CAMERA_DEVICES:

        print(
            f"Trying /dev/video{dev} ..."
        )

        cap = cv2.VideoCapture(
            dev,
            cv2.CAP_V4L2
        )

        if not cap.isOpened():

            print(
                f"   ❌ Could not open /dev/video{dev}"
            )

            cap.release()

            continue

        cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            640
        )

        cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            480
        )

        ret, frame = cap.read()

        if ret and frame is not None:

            print(
                f"   ✅ Camera successfully opened "
                f"on /dev/video{dev}"
            )

            print(
                f"   Frame size: "
                f"{frame.shape[1]}x{frame.shape[0]}"
            )

            return cap

        print(
            "   ❌ Camera opened but could not "
            "read a frame"
        )

        cap.release()

    return None


cap = get_camera()

if cap is None:

    raise RuntimeError(
        "Could not open any camera device "
        "(/dev/video0, /dev/video1, /dev/video2)"
    )


# ============================================================
# GET CAMERA FRAME
# ============================================================

def get_frame():

    with camera_lock:

        ret, frame = cap.read()

    if not ret:

        return None

    return frame


# ============================================================
# RESET
# ============================================================

def reset_sequence():

    global current_step
    global completed_steps
    global last_pen_center
    global stationary_frames
    global last_gray_frame

    with sequence_lock:

        completed_steps = [
            False,
            False,
            False
        ]

        current_step = 0

    with motion_lock:

        last_pen_center = None
        stationary_frames = 0
        last_gray_frame = None

    print()
    print("🔄 Sequence reset")
    print()


# ============================================================
# NANOOWL DETECTION
# ============================================================

def run_nanoowl(frame):

    """
    Run NanoOWL on one OpenCV BGR frame.

    Returns:
        output
        labels_detected
    """

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    pil_image = Image.fromarray(
        rgb
    )

    output = predictor.predict(
        image=pil_image,
        tree=tree,
        threshold=NANOOWL_THRESHOLD
    )

    labels_detected = []

    for detection in output.detections:

        try:

            for label_index in detection.labels:

                label_index = int(
                    label_index
                )

                if (
                    0 <= label_index
                    < len(tree.labels)
                ):

                    label = tree.labels[
                        label_index
                    ]

                    label = str(
                        label
                    ).strip().lower()

                    labels_detected.append(
                        label
                    )

        except Exception as e:

            print(
                "Detection label extraction error:",
                repr(e)
            )

            print(
                "Detection:",
                detection
            )

    return (
        output,
        labels_detected
    )


# ============================================================
# FIND PEN / BALL CENTER
# ============================================================

def get_pen_center(output):

    """
    Find the bounding-box center of detected pen, stylus, marker, or ball.

    Returns:
        (x, y) or None
    """

    for detection in output.detections:

        try:

            detection_labels = []

            for label_index in detection.labels:

                label_index = int(
                    label_index
                )

                if (
                    0 <= label_index
                    < len(tree.labels)
                ):

                    label = str(
                        tree.labels[
                            label_index
                        ]
                    ).strip().lower()

                    detection_labels.append(
                        label
                    )

            is_target_detection = any(
                keyword in label
                for label in detection_labels
                for keyword in [
                    "pen",
                    "stylus",
                    "marker",
                    "ball"
                ]
            )

            if not is_target_detection:

                continue

            box = detection.box

            x1, y1, x2, y2 = map(
                float,
                box
            )

            center_x = int(
                (x1 + x2) / 2
            )

            center_y = int(
                (y1 + y2) / 2
            )

            return (
                center_x,
                center_y
            )

        except Exception as e:

            print(
                "Bounding-box error:",
                repr(e)
            )

    return None


# ============================================================
# UPDATE MOTION TRACKING (OBJECT AND CAMERA SCENE)
# ============================================================

def update_motion(frame, pen_center):

    """
    Computes both object-specific motion and global camera frame motion.

    Returns:
        pen_motion, pen_is_stationary, scene_motion
    """

    global last_pen_center
    global stationary_frames
    global last_gray_frame

    with motion_lock:

        # --- 1. Compute Frame Motion via Differencing ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if last_gray_frame is None:

            scene_motion = 0.0

        else:

            frame_delta = cv2.absdiff(last_gray_frame, gray)
            scene_motion = float(np.mean(frame_delta))

        last_gray_frame = gray

        # --- 2. Compute Tracked Object Motion ---
        if pen_center is None:

            stationary_frames = 0

            last_pen_center = None

            return (
                None,
                False,
                scene_motion
            )

        if last_pen_center is None:

            last_pen_center = pen_center

            stationary_frames = 0

            return (
                None,
                False,
                scene_motion
            )

        dx = pen_center[0] - last_pen_center[0]
        dy = pen_center[1] - last_pen_center[1]

        pen_motion = float(
            np.sqrt(dx * dx + dy * dy)
        )

        if pen_motion <= MOTION_THRESHOLD:

            stationary_frames += 1

        else:

            stationary_frames = 0

        last_pen_center = pen_center

        pen_is_stationary = (
            stationary_frames >= STATIONARY_REQUIRED
        )

        return (
            pen_motion,
            pen_is_stationary,
            scene_motion
        )


# ============================================================
# TEST NANOOWL BEFORE STARTING WEB SERVER
# ============================================================

print()
print("=" * 60)
print(" TESTING NANOOWL INFERENCE")
print("=" * 60)
print()

test_frame = get_frame()

if test_frame is None:

    raise RuntimeError(
        "Camera opened but returned no test frame."
    )

print(
    "Camera test frame received. "
    "Running NanoOWL..."
)

try:

    test_output, test_labels = run_nanoowl(
        test_frame
    )

    print()
    print(
        "✅ NanoOWL inference completed."
    )

    print()
    print("Detected labels:")

    if len(test_labels) == 0:

        print("  NONE")

    else:

        for label in test_labels:

            print(
                " ",
                label
            )

    print()

except Exception:

    print()
    print(
        "❌ NANOOWL INFERENCE FAILED"
    )

    print()

    traceback.print_exc()

    raise

print("=" * 60)
print(" NANOOWL TEST COMPLETE")
print("=" * 60)
print()


# ============================================================
# WEB SERVER
# ============================================================

class CamHandler(BaseHTTPRequestHandler):

    def log_message(
        self,
        format,
        *args
    ):

        return

    def do_GET(self):

        if self.path == "/":

            self.send_response(200)

            self.send_header(
                "Content-Type",
                "text/html; charset=utf-8"
            )

            self.send_header(
                "Cache-Control",
                "no-cache"
            )

            self.end_headers()

            html = """
<!DOCTYPE html>

<html>

<head>

<title>NanoOWL Object Drop Detector</title>

<meta
    name="viewport"
    content="width=device-width, initial-scale=1.0"
>

<style>

body {
    background-color: #121212;
    color: white;
    text-align: center;
    font-family: Arial, sans-serif;
    margin: 0;
    padding: 20px;
}

h1 {
    color: #00ff88;
}

img {
    width: 90%;
    max-width: 900px;
    border: 3px solid #333;
    border-radius: 10px;
}

button {
    background-color: #333;
    color: white;
    border: none;
    padding: 12px 25px;
    margin-top: 15px;
    border-radius: 6px;
    font-size: 16px;
    cursor: pointer;
}

button:hover {
    background-color: #555;
}

</style>

</head>

<body>

<h1>
🟢 NanoOWL Object Drop Detector
</h1>

<p>
Object in Hand → Falling → Ball / Object on Floor (Zero Motion)
</p>

<img src="/cam.mjpg">

<br>

<button onclick="location.href='/reset'">
Reset Sequence
</button>

</body>

</html>
"""

            self.wfile.write(
                html.encode("utf-8")
            )

            return

        # ====================================================
        # RESET
        # ====================================================

        if self.path == "/reset":

            reset_sequence()

            self.send_response(303)

            self.send_header(
                "Location",
                "/"
            )

            self.end_headers()

            return

        # ====================================================
        # CAMERA STREAM
        # ====================================================

        if self.path == "/cam.mjpg":

            self.stream_camera()

            return

        self.send_response(404)

        self.end_headers()

    # ========================================================
    # CAMERA STREAM LOOP
    # ========================================================

    def stream_camera(self):

        self.send_response(200)

        self.send_header(
            "Content-Type",
            "multipart/x-mixed-replace; boundary=frame"
        )

        self.send_header(
            "Cache-Control",
            "no-cache"
        )

        self.send_header(
            "Pragma",
            "no-cache"
        )

        self.end_headers()

        global current_step
        global completed_steps

        while True:

            try:

                # ============================================
                # GET FRAME
                # ============================================

                frame = get_frame()

                if frame is None:

                    time.sleep(0.05)

                    continue

                # ============================================
                # NANOOWL
                # ============================================

                output, labels_detected = run_nanoowl(
                    frame
                )

                # ============================================
                # DETECTION FLAGS
                # ============================================

                has_object = any(
                    keyword in label
                    for label in labels_detected
                    for keyword in [
                        "pen",
                        "stylus",
                        "marker",
                        "ball"
                    ]
                )

                has_hand = any(
                    "hand" in label
                    for label in labels_detected
                )

                has_floor = any(
                    "floor" in label
                    for label in labels_detected
                )

                # ============================================
                # GET POSITIONS & SCENE MOTION
                # ============================================

                pen_center = get_pen_center(
                    output
                )

                pen_motion, pen_is_stationary, scene_motion = (
                    update_motion(
                        frame,
                        pen_center
                    )
                )

                # ============================================
                # SEQUENCE LOGIC
                # ============================================

                with sequence_lock:

                    # ----------------------------------------
                    # STEP 1: Object in Hand
                    # ----------------------------------------

                    if current_step == 0:

                        if has_object and has_hand:

                            completed_steps[0] = True

                            current_step = 1

                            print(
                                "\n>>> "
                                "STEP 1 COMPLETE: "
                                "OBJECT IN HAND\n"
                            )

                    # ----------------------------------------
                    # STEP 2: Object Falling
                    # ----------------------------------------

                    elif current_step == 1:

                        if (
                            has_object
                            and not has_hand
                            and not has_floor
                        ):

                            completed_steps[1] = True

                            current_step = 2

                            print(
                                "\n>>> "
                                "STEP 2 COMPLETE: "
                                "OBJECT FALLING\n"
                            )

                    # ----------------------------------------
                    # STEP 3: Motion is zero / Ball on floor
                    # ----------------------------------------

                    elif current_step == 2:

                        # Checks if camera scene motion is zero OR object stationary
                        if (scene_motion <= SCENE_MOTION_THRESHOLD) or pen_is_stationary:

                            completed_steps[2] = True

                            current_step = 3

                            print(
                                "\n>>> "
                                "STEP 3 COMPLETE: "
                                "BALL / OBJECT ON FLOOR (ZERO MOTION)\n"
                            )

                            print(
                                ">>> "
                                "SEQUENCE COMPLETE <<<\n"
                            )

                # ============================================
                # DRAW NANOOWL OUTPUT OR GREENSCREEN
                # ============================================

                with sequence_lock:

                    is_complete = (current_step == 3)

                if is_complete:

                    # --- SOLID GREEN SCREEN WHEN COMPLETE ---
                    frame[:] = (0, 255, 0)

                    # UI Panel on Green Screen
                    cv2.rectangle(
                        frame,
                        (10, 10),
                        (600, 220),
                        (0, 0, 0),
                        -1
                    )

                    cv2.putText(
                        frame,
                        "SEQUENCE COMPLETE!",
                        (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"Motion level: {scene_motion:.2f} (ZERO)",
                        (30, 85),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

                    for idx, step_name in enumerate(sequence_steps):

                        y = 125 + idx * 28

                        cv2.putText(
                            frame,
                            f"[X] {step_name}",
                            (30, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2
                        )

                else:

                    try:

                        frame = predictor.draw_tree_output(
                            frame,
                            output
                        )

                    except Exception as e:

                        print("Draw error:", repr(e))

                    # ============================================
                    # NORMAL CAMERA HUD PANEL
                    # ============================================

                    cv2.rectangle(
                        frame,
                        (10, 10),
                        (600, 225),
                        (0, 0, 0),
                        -1
                    )

                    cv2.putText(
                        frame,
                        "OBJECT DROP DETECTOR",
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2
                    )

                    detection_text = (
                        "Detected: " + ", ".join(labels_detected) if labels_detected else "Detected: none"
                    )

                    cv2.putText(
                        frame,
                        detection_text[:70],
                        (20, 58),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (255, 255, 255),
                        1
                    )

                    motion_text = f"Object Motion: {pen_motion:.1f}px" if pen_motion is not None else "Object Motion: waiting"

                    cv2.putText(
                        frame,
                        motion_text,
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (255, 255, 255),
                        1
                    )

                    scene_color = (0, 255, 0) if scene_motion <= SCENE_MOTION_THRESHOLD else (255, 255, 255)

                    cv2.putText(
                        frame,
                        f"Camera Scene Motion: {scene_motion:.2f}",
                        (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        scene_color,
                        1
                    )

                    with sequence_lock:

                        step = current_step

                        steps = completed_steps.copy()

                    for idx, step_name in enumerate(sequence_steps):

                        y = 135 + idx * 24

                        if steps[idx]:

                            color = (0, 255, 0)

                            prefix = "[X]"

                        elif idx == step:

                            color = (0, 255, 255)

                            prefix = "[>]"

                        else:

                            color = (120, 120, 120)

                            prefix = "[ ]"

                        cv2.putText(
                            frame,
                            f"{prefix} {step_name}",
                            (20, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            2
                        )

                # ============================================
                # ENCODE & STREAM JPEG
                # ============================================

                success, encoded = cv2.imencode(
                    ".jpg",
                    frame,
                    [
                        cv2.IMWRITE_JPEG_QUALITY,
                        JPEG_QUALITY
                    ]
                )

                if not success:

                    continue

                jpg = encoded.tobytes()

                self.wfile.write(
                    b"--frame\r\n"
                )

                self.wfile.write(
                    b"Content-Type: image/jpeg\r\n"
                )

                self.wfile.write(
                    f"Content-Length: {len(jpg)}\r\n\r\n"
                    .encode()
                )

                self.wfile.write(
                    jpg
                )

                self.wfile.write(
                    b"\r\n"
                )

                self.wfile.flush()

                time.sleep(0.01)

            except (
                BrokenPipeError,
                ConnectionResetError,
                ConnectionAbortedError
            ):

                print(
                    "Browser disconnected."
                )

                break

            except Exception:

                print(
                    "\n❌ CAMERA STREAM ERROR"
                )

                traceback.print_exc()

                print()

                break


# ============================================================
# THREADED SERVER
# ============================================================

class ThreadedHTTPServer(
    ThreadingMixIn,
    HTTPServer
):

    daemon_threads = True

    allow_reuse_address = True


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print()

    print("=" * 60)

    print(
        " NANOOWL OBJECT DROP DETECTOR"
    )

    print("=" * 60)

    print()

    print(
        "Open in browser:",
        f"http://<JETSON-IP>:{SERVER_PORT}"
    )

    print(
        "NanoOWL threshold:",
        NANOOWL_THRESHOLD
    )

    print(
        "Motion threshold:",
        MOTION_THRESHOLD,
        "pixels"
    )

    print(
        "Scene motion zero-threshold:",
        SCENE_MOTION_THRESHOLD
    )

    print(
        "Stationary frames required:",
        STATIONARY_REQUIRED
    )

    print(
        "Press Ctrl+C to stop.\n"
    )

    server = ThreadedHTTPServer(
        (
            SERVER_HOST,
            SERVER_PORT
        ),
        CamHandler
    )

    try:

        server.serve_forever()

    except KeyboardInterrupt:

        print(
            "\nStopping server..."
        )

    finally:

        server.server_close()

        if cap:

            cap.release()

        print(
            "Server stopped."
        )
