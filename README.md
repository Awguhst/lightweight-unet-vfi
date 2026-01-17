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

## Visual Comparison: Original vs. Interpolated

Below is a side-by-side comparison demonstrating the effect of the frame interpolation model.  
The interpolated video generates a smooth intermediate frame between consecutive frames, resulting in noticeably smoother motion.

| Original Video | Frame Interpolated Video |
|----------------------------------|--------------------------|
| ![Original Video](assets/original.gif) | ![Frame Interpolated Video](assets/interpolated.gif) |

## Live Demo
Try the CPU-only Gradio app deployed on Hugging Face Spaces:  
[Live Gradio Demo](https://huggingface.co/spaces/Awguhst/Frame-Interpolation) 

Upload any video, interpolate frames, and download the smoother version with original audio preserved.
