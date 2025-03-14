#!/usr/bin/env python3
"""
Script to convert videos to audio and trim them to a maximum of 30 seconds.
This is useful for preparing videos for speech recognition with Yandex SpeechKit,
which has a limit of 30 seconds for short audio recognition.
"""

import os
import argparse
import subprocess
import tempfile
from pathlib import Path
from tqdm import tqdm

def get_video_duration(video_path):
    """
    Get the duration of a video in seconds using FFmpeg.
    
    Args:
        video_path (str): Path to the video file.
        
    Returns:
        float: Duration of the video in seconds.
    """
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        video_path
    ]
    
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        duration = float(result.stdout.strip())
        return duration
    except (subprocess.SubprocessError, ValueError) as e:
        print(f"Error getting duration for {video_path}: {e}")
        return 0.0

def convert_video_to_audio(video_path, output_path, max_duration=30, format='mp3', sample_rate=48000):
    """
    Convert a video to audio and trim it to a maximum duration.
    
    Args:
        video_path (str): Path to the video file.
        output_path (str): Path to save the output audio file.
        max_duration (int, optional): Maximum duration in seconds. Defaults to 30.
        format (str, optional): Output audio format. Defaults to 'mp3'.
        sample_rate (int, optional): Sample rate in Hz. Defaults to 48000.
        
    Returns:
        bool: True if conversion was successful, False otherwise.
    """
    # Get video duration
    duration = get_video_duration(video_path)
    
    if duration <= 0:
        print(f"Could not determine duration for {video_path}")
        return False
    
    # Determine if we need to trim the video
    if duration > max_duration:
        print(f"Video {os.path.basename(video_path)} is {duration:.2f} seconds, trimming to {max_duration} seconds")
        trim_option = ['-t', str(max_duration)]
    else:
        print(f"Video {os.path.basename(video_path)} is {duration:.2f} seconds, no trimming needed")
        trim_option = []
    
    # Convert video to audio
    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-vn',  # No video
        '-ar', str(sample_rate),  # Sample rate
        '-ac', '1',  # Mono
        '-f', format
    ] + trim_option + [
        output_path
    ]
    
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except subprocess.SubprocessError as e:
        print(f"Error converting {video_path} to audio: {e}")
        return False

def process_videos(input_dir, output_dir, max_duration=30, format='mp3', sample_rate=48000):
    """
    Process all videos in a directory, converting them to audio and trimming if necessary.
    
    Args:
        input_dir (str): Directory containing video files.
        output_dir (str): Directory to save output audio files.
        max_duration (int, optional): Maximum duration in seconds. Defaults to 30.
        format (str, optional): Output audio format. Defaults to 'mp3'.
        sample_rate (int, optional): Sample rate in Hz. Defaults to 48000.
        
    Returns:
        list: List of paths to the output audio files.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(list(Path(input_dir).glob(f'*{ext}')))
    
    if not video_files:
        print(f"No video files found in {input_dir}")
        return []
    
    print(f"Found {len(video_files)} video files")
    
    # Process each video
    output_files = []
    
    for video_path in tqdm(video_files, desc="Processing videos"):
        video_name = os.path.basename(video_path)
        output_name = f"{os.path.splitext(video_name)[0]}.{format}"
        output_path = os.path.join(output_dir, output_name)
        
        success = convert_video_to_audio(
            video_path=str(video_path),
            output_path=output_path,
            max_duration=max_duration,
            format=format,
            sample_rate=sample_rate
        )
        
        if success:
            output_files.append(output_path)
    
    return output_files

def main():
    parser = argparse.ArgumentParser(description='Convert videos to audio and trim them to a maximum duration.')
    parser.add_argument('--input-dir', required=True, help='Directory containing video files')
    parser.add_argument('--output-dir', required=True, help='Directory to save output audio files')
    parser.add_argument('--max-duration', type=int, default=30, help='Maximum duration in seconds (default: 30)')
    parser.add_argument('--format', default='mp3', choices=['mp3', 'wav', 'ogg'], help='Output audio format (default: mp3)')
    parser.add_argument('--sample-rate', type=int, default=48000, help='Sample rate in Hz (default: 48000)')
    
    args = parser.parse_args()
    
    output_files = process_videos(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        max_duration=args.max_duration,
        format=args.format,
        sample_rate=args.sample_rate
    )
    
    print(f"Successfully processed {len(output_files)} videos")
    
    if output_files:
        print("\nOutput audio files:")
        for file_path in output_files:
            print(f"  - {file_path}")

if __name__ == "__main__":
    main() 