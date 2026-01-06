# Video Frame Interpolation with Lightweight U-Net

## What is Video Frame Interpolation?

Video frame interpolation is a technique that generates new **intermediate frames** between two existing consecutive frames in a video. This creates the illusion of smoother motion and effectively increases the frame rate (e.g., from 30 FPS to 60 FPS) without changing the playback speed.

Why is it useful?
- Turns choppy low-FPS videos into buttery-smooth high-FPS ones
- Enables convincing slow-motion from normal-speed footage
- Improves visual quality for old videos, games, or compressed content

Traditional methods (like simple blending) often produce ghosting or blurry artifacts. Modern deep learning approaches, like this U-Net model, learn complex motion patterns to synthesize realistic in-between frames with sharp details and accurate object movement.

## Project Overview
This is a simple U-Net model trained on the Vimeo-90K dataset. It takes two consecutive frames as input and predicts a high-quality middle frame, enabling smooth video frame rate upsampling while preserving motion continuity and minimizing visual artifacts.

## Live Demo
Try the CPU-only Gradio app deployed on Hugging Face Spaces:  
[Live Gradio Demo](https://huggingface.co/spaces/Awguhst/Frame-Interpolation) 

Upload any video, interpolate frames, and download the smoother version with original audio preserved.

## Side-by-Side Video Comparison
Here's a simple side-by-side comparison of an original low-FPS clip (left) next to the interpolated smoother version (right), demonstrating the dramatic improvement in motion fluidity:

<video controls width="100%">
  <source src="demo/video.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

(Example from a motion interpolation demo showing the "soap opera effect" in slow motion.)
