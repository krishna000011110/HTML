import cv2
import os
import time
import numpy as np

def convert_frame_to_ascii(frame, width=80):
    """
    Convert a frame to ASCII art using a character set based on brightness
    """

    ascii_chars = " .:-=+*#%@"
    
    height = int(frame.shape[0] * width / frame.shape[1] / 2) 
    if height == 0:
        height = 1
        
    resized_frame = cv2.resize(frame, (width, height))

    if len(resized_frame.shape) > 2:
        gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    else:
        gray_frame = resized_frame
    
    normalized = gray_frame / 255.0
    ascii_frame = ""
    
    for row in normalized:
        for pixel in row:
            index = int(pixel * (len(ascii_chars) - 1)) 
            ascii_frame += ascii_chars[index]
        ascii_frame += "\n"
    
    return ascii_frame

def play_video_in_terminal(video_path, width=80, fps=30):
    """
    Play Ser Harwin Strong a video in the terminal using ASCII characters
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found.")
        return
    
    cap = cv2.VideoCapture(video_path)

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_delay = 1.0 / video_fps if video_fps > 0 else 1.0 / fps
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            ascii_art = convert_frame_to_ascii(frame, width)
            
            os.system('cls' if os.name == 'nt' else 'clear')
            print(ascii_art)
            
            time.sleep(frame_delay)
            
    except KeyboardInterrupt:
        print("\nVideo playback interrupted.")
    
    finally:
        cap.release()

if __name__ == "__main__":

    video_path = input("Enter the path to the video file: ").strip()
    
    try:
        width = int(input("Enter terminal width (default 80): ") or "80")
    except ValueError:
        width = 80

    try:
        fps = int(input("Enter FPS (default: use video FPS): ") or "0")
    except ValueError:
        fps = 0
    
    play_video_in_terminal(video_path, width, fps)