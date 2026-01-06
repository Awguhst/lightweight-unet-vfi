import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quantization import quantize_dynamic
from torchvision import transforms
import cv2
import torch
from PIL import Image
import numpy as np
import subprocess
import gradio as gr 

class DoubleConv(nn.Module):
    """(convolution => [BN] => GELU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Pad to match spatial dimensions
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                       diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.tanh = nn.Tanh()  
    def forward(self, x):
        return self.tanh(self.conv(x))

class FrameInterpolationModel(nn.Module):
    def __init__(self, n_channels=6, n_out_channels=3):
        super(FrameInterpolationModel, self).__init__()
       
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)  
       
        self.up1 = Up(256, 128)
        self.up2 = Up(128, 64)
       
        self.outc = OutConv(64, n_out_channels)
    def forward(self, x):
        x1 = self.inc(x)  # 64
        x2 = self.down1(x1)  # 128
        x3 = self.down2(x2)  # 256 
       
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
       
        return self.outc(x)
    
transform = transforms.Compose([
    transforms.ToTensor(),                         
    transforms.Normalize(mean=[0.5, 0.5, 0.5],       
                         std=[0.5, 0.5, 0.5])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = FrameInterpolationModel().to(device)  
model.load_state_dict(torch.load('frame_interpolation_best.pth', map_location=torch.device('cpu')))
model = quantize_dynamic(model, {torch.nn.Conv2d, torch.nn.Linear}, dtype=torch.qint8)
model = torch.compile(model, mode="max-autotune")
model.eval()

def get_video_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 1000:  # Safety check for invalid values
        print("Warning: Detected FPS is invalid. Falling back to 30 FPS.")
        fps = 30.0
    else:
        print(f"Detected original FPS: {fps:.2f}")
    cap.release()
    return fps

def load_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))
    cap.release()
    if len(frames) == 0:
        raise ValueError("No frames loaded. Check if the video path is correct and file is accessible.")
    return frames

def process_video(video_path):
    if video_path is None:
        raise ValueError("No video uploaded.")
    
    print("Starting processing...")
    
    # Load frames
    print("Loading video frames...")
    original_frames = load_video_frames(video_path)
    total_original = len(original_frames)
    print(f"Loaded {total_original} frames from the video.")
    
    # Detect original FPS
    original_fps = get_video_fps(video_path)
    print(f"Original FPS: {original_fps:.2f}")
    
    # Generate interpolated frames 
    print("Starting frame interpolation...")
    interpolated_frames = []
    total_pairs = len(original_frames) - 1  
    
    with torch.no_grad():
        for i in range(total_pairs):
            f1 = transform(original_frames[i]).unsqueeze(0).to(device)
            f2 = transform(original_frames[i+1]).unsqueeze(0).to(device)
            input_pair = torch.cat([f1, f2], dim=1)
            pred = model(input_pair)
            
            # Convert to PIL Image
            img = (pred[0].clamp(-1, 1).cpu() + 1) / 2
            img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            interpolated_frames.append(Image.fromarray(img))
            
            # Progress update – print every 10 pairs or at the end
            if (i + 1) % 10 == 0 or i == total_pairs - 1:
                percentage = (i + 1) / total_pairs * 100
                print(f"Interpolated {i+1}/{total_pairs} frame pairs ({percentage:.1f}%)")
    
    print("Interpolation complete! Building final frame sequence...")
    
    # Build final smooth sequence
    final_frames = []
    for i in range(len(original_frames) - 1):
        final_frames.append(original_frames[i])
        final_frames.append(interpolated_frames[i])
    final_frames.append(original_frames[-1])
    
    total_final = len(final_frames)
    print(f"Final video will have {total_final} frames (2x smoother).")
    
    # Save video at normal speed
    def save_video_normal_speed(frames, output_path, original_fps):
        if len(frames) == 0:
            raise ValueError("No frames to save.")
        width, height = frames[0].size  # PIL: (width, height)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_fps = original_fps * 2  # To maintain real-time speed
        out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))
        
        for frame in frames:
            frame_bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        print(f"Video saved → {output_path} at {output_fps:.2f} FPS (original speed preserved)")
    
    # Save the result
    silent_video_path = 'interpolated_video.mp4'
    save_video_normal_speed(final_frames, silent_video_path, original_fps)
    
    final_output_path = 'interpolated_video_with_audio.mp4'
    subprocess.run([
        'ffmpeg', '-i', silent_video_path, '-i', video_path,
        '-c:v', 'copy', '-c:a', 'aac',
        '-map', '0:v:0', '-map', '1:a:0?',
        '-shortest', '-y', final_output_path
    ], check=True)
    
    print("Done! Your buttery-smooth normal-speed video is ready.")
    return final_output_path

with gr.Blocks(title="Video Frame Interpolation") as iface:
    gr.Markdown(
        """
        <div style="text-align: center;">
            <h1>🎥 Video Frame Interpolation</h1>
            <p>Upload a video and get a smoother version with <strong>doubled frame rate</strong> using AI frame interpolation.</p>
        </div>
        """
    )
    
    with gr.Row():
        with gr.Column():
            input_video = gr.Video(
                label="Upload Video (max ~20s recommended)",
                sources=["upload"]  
            )
            submit_btn = gr.Button("Generate Smoother Video", variant="primary")
        
        with gr.Column():
            output_video = gr.Video(label="")
    
    gr.Examples(
        examples=[
            ["bird.mp4"],
            ["jellyfish.mp4"],
            ["chicken.mp4"]
        ],
        inputs=input_video,
        label="Click an example to try it!"
    )
    
    submit_btn.click(
        fn=process_video,  
        inputs=input_video,
        outputs=output_video,
        queue=True
    )

iface.launch()