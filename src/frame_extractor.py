import cv2
import os
import json

def extract_frames(video_path, output_dir, interval=1):
    """
    Extract one frame from the video every `interval` seconds.

    Parameters:
        video_path (str): Path to input video.
        output_dir (str): Directory where frames will be saved.
        interval (int): Time interval between extracted frames.

    Returns:
        list: Metadata containing frame paths and timestamps.
    """

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Unable to open the video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        raise ValueError("Unable to determine video FPS.")

    duration = total_frames / fps

    print(f"Video FPS       : {fps:.2f}")
    print(f"Total frames    : {total_frames}")
    print(f"Duration        : {duration:.2f} seconds")
    print(f"Sampling rate   : 1 frame every {interval} second(s)")
    print()

    frame_metadata = []

    current_time = 0
    frame_number = 0

    while current_time < duration:

        cap.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)

        success, frame = cap.read()

        if not success:
            break

        filename = f"frame_{frame_number:05d}.jpg"
        frame_path = os.path.join(output_dir, filename)

        cv2.imwrite(frame_path, frame)

        frame_metadata.append({
            "frame_number": frame_number,
            "timestamp": round(current_time, 2),
            "frame_path": frame_path
        })

        print(
            f"Extracted: {filename} "
            f"| Timestamp: {current_time:.2f}s"
        )

        frame_number += 1
        current_time += interval

    cap.release()

    print()
    print(f"Successfully extracted {len(frame_metadata)} frames.")

    # Save frame metadata
    metadata_path = os.path.join(
        "results",
        "frame_metadata.json"
    )

    os.makedirs("results", exist_ok=True)

    with open(metadata_path, "w") as f:
        json.dump(frame_metadata, f, indent=4)

    print(f"Metadata saved to: {metadata_path}")

    return frame_metadata


if __name__ == "__main__":

    video_path = video_path = r"C:\Users\Admin\Documents\SSSC AI ML Cohort\Final Capstone Project - Vision seek AI\Data\Traffic Video.mp4"
    output_dir = "frames"

    extract_frames(
        video_path=video_path,
        output_dir=output_dir,
        interval=1
    )
