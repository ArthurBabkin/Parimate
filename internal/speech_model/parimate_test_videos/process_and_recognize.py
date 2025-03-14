#!/usr/bin/env python3
"""
Script to process videos, convert them to audio, and recognize speech using Yandex SpeechKit.
"""

import os
import argparse
import json
import sys
import tempfile
import subprocess
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import YandexSpeechKit
from video_processor import process_videos

# Load environment variables from .env file
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)

def convert_audio_format(audio_file_path, target_format='lpcm', sample_rate=16000, max_duration=29, start_time=0):
    """
    Convert audio file to a different format using FFmpeg and trim it to reduce file size.
    
    Args:
        audio_file_path (str): Path to the audio file.
        target_format (str, optional): Target audio format. Defaults to 'lpcm'.
        sample_rate (int, optional): Sample rate in Hz. Defaults to 16000.
        max_duration (int, optional): Maximum duration in seconds. Defaults to 29.
        start_time (int, optional): Start time in seconds. Defaults to 0.
        
    Returns:
        str: Path to the converted audio file.
    """
    # Create a temporary file for the converted audio
    temp_dir = tempfile.gettempdir()
    temp_audio_file = os.path.join(temp_dir, f"converted_{os.path.basename(audio_file_path)}_{start_time}_{max_duration}.{target_format}")
    
    # Convert audio using FFmpeg with lower sample rate and trim to reduce file size
    cmd = [
        'ffmpeg',
        '-i', audio_file_path,
        '-ss', str(start_time),  # Start time
        '-ar', str(sample_rate),  # Lower sample rate
        '-ac', '1',  # Mono
        '-t', str(max_duration),  # Trim to max_duration seconds
        '-f', target_format,
        temp_audio_file
    ]
    
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return temp_audio_file
    except subprocess.SubprocessError as e:
        print(f"Error converting {audio_file_path} to {target_format}: {e}")
        return None

def get_audio_duration(audio_path):
    """
    Get the duration of an audio file in seconds using FFmpeg.
    
    Args:
        audio_path (str): Path to the audio file.
        
    Returns:
        float: Duration of the audio file in seconds.
    """
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        audio_path
    ]
    
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        duration = float(result.stdout.strip())
        return duration
    except (subprocess.SubprocessError, ValueError) as e:
        print(f"Error getting duration for {audio_path}: {e}")
        return 0.0

def recognize_speech_in_audio_files(audio_files, auto_detect_language=False, language_code='ru-RU', format='mp3'):
    """
    Recognize speech in audio files using Yandex SpeechKit.
    
    Args:
        audio_files (list): List of paths to audio files.
        auto_detect_language (bool, optional): Whether to automatically detect the language. Defaults to False.
        language_code (str, optional): Language code to use if auto_detect_language is False. Defaults to 'ru-RU'.
        format (str, optional): Audio format. Defaults to 'mp3'.
        
    Returns:
        dict: Dictionary mapping audio file paths to recognition results.
    """
    # Get API key and folder ID from environment variables
    api_key = os.environ.get('YC_API_KEY')
    folder_id = os.environ.get('YC_FOLDER_ID')
    
    # Check if API key and folder ID are set
    if not api_key:
        raise ValueError("YC_API_KEY environment variable is not set.")
    
    if not folder_id:
        raise ValueError("YC_FOLDER_ID environment variable is not set.")
    
    # Initialize the speech recognition model
    speech_model = YandexSpeechKit(api_key=api_key, folder_id=folder_id)
    
    # Recognize speech in each audio file
    results = {}
    temp_files = []
    
    # Maximum duration for audio (in seconds)
    max_duration = 29
    
    for audio_path in tqdm(audio_files, desc="Recognizing speech"):
        try:
            # Get the duration of the audio file
            duration = get_audio_duration(audio_path)
            
            if duration <= 0:
                print(f"Could not determine duration for {audio_path}")
                continue
            
            # Convert audio to a supported format and trim to max_duration if needed
            if duration > max_duration:
                print(f"Audio {os.path.basename(audio_path)} is {duration:.2f} seconds, trimming to {max_duration} seconds")
            else:
                print(f"Converting {os.path.basename(audio_path)} from {format} to lpcm...")
                
            converted_audio_path = convert_audio_format(
                audio_path, 
                target_format='wav', 
                sample_rate=16000,
                max_duration=max_duration
            )
            
            if not converted_audio_path:
                raise ValueError(f"Failed to convert {audio_path} to lpcm format.")
            
            temp_files.append(converted_audio_path)
            recognition_format = 'lpcm'
            
            if auto_detect_language:
                result = speech_model.auto_detect_language_and_recognize(
                    audio_file_path=converted_audio_path,
                    format=recognition_format,
                    sample_rate_hertz=16000,
                    method='http'
                )
                
                results[audio_path] = {
                    'detected_language': result['detected_language'],
                    'language_name': result['language_name'],
                    'text': result['text'],
                    'confidence_scores': result['confidence_scores']
                }
            else:
                result = speech_model.recognize_short_audio_http(
                    audio_file_path=converted_audio_path,
                    format=recognition_format,
                    sample_rate_hertz=16000,
                    language_code=language_code
                )
                
                results[audio_path] = {
                    'language': language_code,
                    'text': result.get('result', '')
                }
        except Exception as e:
            print(f"Error recognizing speech in {audio_path}: {e}")
            results[audio_path] = {
                'error': str(e)
            }
    
    # Clean up temporary files
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    return results

def save_results_to_json(results, output_file):
    """
    Save recognition results to a JSON file.
    
    Args:
        results (dict): Dictionary mapping audio file paths to recognition results.
        output_file (str): Path to save the JSON file.
    """
    # Convert results to a more readable format
    formatted_results = {}
    
    for audio_path, result in results.items():
        audio_name = os.path.basename(audio_path)
        formatted_results[audio_name] = result
    
    # Save to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(formatted_results, f, ensure_ascii=False, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Process videos and recognize speech using Yandex SpeechKit.')
    parser.add_argument('--input-dir', required=True, help='Directory containing video files')
    parser.add_argument('--output-dir', default='processed_audio', help='Directory to save output audio files')
    parser.add_argument('--results-file', default='recognition_results.json', help='File to save recognition results')
    parser.add_argument('--max-duration', type=int, default=30, help='Maximum duration in seconds (default: 30)')
    parser.add_argument('--format', default='mp3', choices=['mp3', 'wav', 'ogg'], help='Output audio format (default: mp3)')
    parser.add_argument('--auto-detect-language', action='store_true', help='Automatically detect the language')
    parser.add_argument('--language', default='ru-RU', help='Language code to use if auto-detect-language is False (default: ru-RU)')
    parser.add_argument('--skip-processing', action='store_true', help='Skip video processing and only perform speech recognition')
    
    args = parser.parse_args()
    
    # Process videos if not skipped
    if not args.skip_processing:
        print("Processing videos...")
        output_files = process_videos(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            max_duration=args.max_duration,
            format=args.format,
            sample_rate=48000
        )
    else:
        print("Skipping video processing...")
        # Get all audio files in the output directory
        output_files = []
        for ext in [f'.{args.format}']:
            output_files.extend([str(f) for f in Path(args.output_dir).glob(f'*{ext}')])
        
        if not output_files:
            print(f"No audio files found in {args.output_dir}. Exiting.")
            return
        
        print(f"Found {len(output_files)} audio files for recognition.")
    
    # Recognize speech
    print("\nRecognizing speech...")
    results = recognize_speech_in_audio_files(
        audio_files=output_files,
        auto_detect_language=args.auto_detect_language,
        language_code=args.language,
        format=args.format
    )
    
    # Save results
    save_results_to_json(results, args.results_file)
    print(f"\nResults saved to {args.results_file}")
    
    # Print a summary
    print("\nSummary:")
    for audio_path, result in results.items():
        audio_name = os.path.basename(audio_path)
        if 'error' in result:
            print(f"  - {audio_name}: Error: {result['error']}")
        elif 'detected_language' in result:
            confidence_ru = result['confidence_scores'].get('ru-RU', 0)
            confidence_en = result['confidence_scores'].get('en-US', 0)
            print(f"  - {audio_name}: {result['language_name']} ({result['detected_language']}): {result['text'][:50]}...")
            print(f"    Confidence scores: ru-RU: {confidence_ru}, en-US: {confidence_en}")
        else:
            print(f"  - {audio_name}: {result['text'][:50]}...")

if __name__ == "__main__":
    main() 