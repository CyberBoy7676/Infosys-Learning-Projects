# Import necessary libraries
import whisper   # OpenAI's Whisper library for speech-to-text processing
from moviepy.editor import VideoFileClip  # To extract audio from video
import os        # For file handling (optional)
import torch     # To manage device selection (GPU/CPU)

# Check if GPU is available and set the device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the Whisper model on the selected device
# Options: tiny, base, small, medium, large
# Choose based on system capability and speed requirements
model = whisper.load_model("base", device=device)

# Function to extract audio from video
def extract_audio_from_video(video_path, audio_output_path):
    """
    Extracts audio from a video file and saves it as a separate audio file.

    Parameters:
        video_path (str): Path to the video file.
        audio_output_path (str): Path to save the extracted audio.
    """
    try:
        # Load the video file
        video = VideoFileClip(video_path)

        # Extract and save the audio
        video.audio.write_audiofile(audio_output_path)
        print(f"Audio extracted and saved at: {audio_output_path}")
    except Exception as e:
        print(f"An error occurred while extracting audio: {e}")

# Function to transcribe speech to text
def transcribe_audio(file_path):
    """
    Transcribes audio to text using OpenAI's Whisper.

    Parameters:
        file_path (str): Path to the audio file.
        
    Returns:
        str: Transcribed text.
    """
    try:
        # Check if the file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        # Perform transcription
        print("Transcribing audio... This may take a moment.")
        result = model.transcribe(file_path)

        # Return the transcribed text
        return result['text']
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Main logic to process video and transcribe speech
if __name__ == "__main__":
    # Specify the path to the video file (e.g., .mp4, .mkv)
    video_file = "sample_video.mp4"  # Replace with your video file path
    
    # Path to save the extracted audio
    audio_file = "extracted_audio.mp3"  # Path for saving the extracted audio
    
    # Step 1: Extract audio from the video
    extract_audio_from_video(video_file, audio_file)
    
    # Step 2: Transcribe the extracted audio to text
    transcription = transcribe_audio(audio_file)
    
    # Print the transcription result
    if transcription:
        print("\nTranscription:\n")
        print(transcription)
