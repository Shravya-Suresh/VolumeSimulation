import tkinter as tk
from tkinter import ttk, filedialog
from PIL import ImageTk, Image
import pygame
import os
import time
import threading
import torch
import torch.nn as nn
import torchaudio
from torchaudio import transforms as T
import sounddevice as sd
import sys

class MusicPlayer:
    def __init__(self, root):
        self.root = root
        self.root.title("Music Player with Audio Classification")

        # Initialize Pygame mixer
        pygame.mixer.init(frequency=44100, size=-16, channels=2)

        self.default_volume = 1.0  # full volume
        self.reduced_volume = 0.2  # reduced volume when speaking

        pygame.mixer.music.set_volume(self.default_volume)

        
        # Current song and playback state
        self.current_song = None
        self.playing = False
        self.slider_update_id = None
        self.bg_image = None
        self.cover_image = None
        self.classification_active = False
        
        # Audio classification setup
        #self.model = self.load_model_from_checkpoint("/Users/mac/Downloads/content/saved_models_new20250721_184805/best_model_weights.pth")
        self.model = self.load_model_from_checkpoint ("/Users/mac/Desktop/VolumeSimulation/AirPod-Project/content/saved_models_new20250721_184805/best_model_weights.pth")
        self.label_map = {0: "Singing -> No volume reduced", 1: "Speaking -> Volume reduced", 2: "Noise -> No volume reduced"}
        
        # GUI Setup
        self.setup_ui()
        
        # Start the classification thread
        self.start_classification_thread()
    
    def setup_ui(self):
        # Set window size
        self.root.geometry("600x400")
        
        # Create background canvas
        self.bg_canvas = tk.Canvas(self.root)
        self.bg_canvas.pack(fill="both", expand=True)
        
        # Load background image
        try:
            #img_path = "/Users/mac/Desktop/AirPod-Project/bg.webp"
            img_path = "bg.webp"
            self.load_background_image(img_path)
        except Exception as e:
            print(f"Error loading background image: {e}")
            self.bg_canvas.config(bg="#2c3e50")
        
        # Create upload button frame
        self.upload_frame = ttk.Frame(self.bg_canvas)
        self.upload_frame_window = self.bg_canvas.create_window(
            300, 200,
            window=self.upload_frame
        )
        
        # Create 3D effect upload button
        self.upload_btn = tk.Canvas(
            self.upload_frame, 
            width=150, 
            height=50, 
            bg="white", 
            bd=0, 
            highlightthickness=0,
            relief="raised"
        )
        self.upload_btn.pack()
        
        # Draw the upload button with 3D effect
        self.upload_btn.create_rectangle(
            5, 5, 145, 45, 
            fill="#f0f0f0", 
            outline="#cccccc", 
            width=2
        )
        self.upload_btn.create_text(
            75, 25, 
            text="Upload Music", 
            font=("Arial", 12, "bold"), 
            fill="#333333"
        )
        
        # Add shadow effect
        self.upload_btn.create_rectangle(
            5, 45, 145, 47, 
            fill="#e0e0e0", 
            outline=""
        )
        self.upload_btn.create_rectangle(
            145, 5, 147, 45, 
            fill="#e0e0e0", 
            outline=""
        )
        
        self.upload_btn.bind("<Button-1>", self.on_upload_click)
        self.upload_btn.bind("<ButtonRelease-1>", self.on_upload_release)
        
        # Create playback controls frame
        self.playback_frame = ttk.Frame(self.bg_canvas)
        
        # Make frames semi-transparent
        self.style = ttk.Style()
        self.style.configure("TFrame", background="")
        self.upload_frame.config(style="TFrame")
        self.playback_frame.config(style="TFrame")

        self.header_frame = ttk.Frame(self.playback_frame)
        self.header_frame.pack(fill="x", pady=(5, 0))  # Top padding of 5px
        
        # Close button in top-right
        self.close_btn = tk.Button(
        self.header_frame,
        text="✕",  # This is the cross symbol
        font=("Arial", 14, "bold"),  # Increased font size to 14
        command=self.go_back_to_upload,
        bg="white",  # White background
        fg="black",  # Black cross
        bd=0,  # No border
        activebackground="#e0e0e0",  # Light gray when clicked
        activeforeground="black",  # Keep cross black when clicked
        relief="flat"  # Flat appearance
    )
        self.close_btn.pack(side="right", padx=5, pady=5)

        
        # Create a frame for the cover image
        self.cover_frame = ttk.Frame(self.playback_frame)
        self.cover_frame.pack(pady=(10, 0))
        
        # Create a canvas for the cover image
        self.cover_canvas = tk.Canvas(
            self.cover_frame, 
            width=300,
            height=500,
            bg="#333333",
            bd=0,
            highlightthickness=0
        )
        self.cover_canvas.pack()
        
        # Default placeholder text for cover image
        self.cover_canvas.create_text(
            150, 25,
            text="Now Playing",
            font=("Arial", 10),
            fill="white"
        )
        
        # Create a frame for the time labels and seek bar
        top_frame = ttk.Frame(self.playback_frame)
        top_frame.pack(pady=10)
        
        # Left time label
        self.time_label_left = ttk.Label(top_frame, text="00:00", width=6, background="", foreground="white")
        self.time_label_left.pack(side="left", padx=5)
        
        # Seek Bar
        self.seek_bar = ttk.Scale(
            top_frame,
            from_=0,
            to=100,
            orient="horizontal",
            command=self.on_seek
        )
        self.seek_bar.pack(side="left", expand=True, fill="x", padx=5)
        
        # Right time label
        self.time_label_right = ttk.Label(top_frame, text="00:00", width=6, background="", foreground="white")
        self.time_label_right.pack(side="left", padx=5)
        
        # Frame for the play/pause button
        button_frame = ttk.Frame(self.playback_frame)
        button_frame.pack()
        
        # Play/Pause Button
        self.play_pause_btn = tk.Canvas(button_frame, width=40, height=40, bg="white", bd=0, highlightthickness=0)
        self.play_pause_btn.pack(pady=10)
        
        # ▶️ (Play icon - triangle)
        self.draw_play_icon()  
        
        self.play_pause_btn.bind("<Button-1>", self.toggle_play)
        
        # Bind window resize event
        self.root.bind("<Configure>", self.on_window_resize)
        
        # Add classification status label
        self.classification_label = ttk.Label(
            self.playback_frame,
            text="Audio Classification: Active",
            foreground="green",
            background="",
            font=("Arial", 10)
        )
        self.classification_label.pack(pady=5)
    
    def go_back_to_upload(self):
        # Stop any playback
        if self.playing:
            pygame.mixer.music.stop()
            self.playing = False
            if self.slider_update_id:
                self.root.after_cancel(self.slider_update_id)
        
        # Reset current song
        self.current_song = None
        
        # Clear playback frame
        if hasattr(self, 'playback_frame_window'):
            self.bg_canvas.delete(self.playback_frame_window)
        
        # Recreate upload interface
        self.upload_frame = ttk.Frame(self.bg_canvas)
        self.upload_frame_window = self.bg_canvas.create_window(
            self.root.winfo_width()/2, self.root.winfo_height()/2,
            window=self.upload_frame
        )
        
        # Rebuild upload button
        self.upload_btn = tk.Canvas(
            self.upload_frame,
            width=150,
            height=50,
            bg="white",
            bd=0,
            highlightthickness=0,
            relief="raised"
        )
        self.upload_btn.pack()
        
        # Redraw button graphics
        self.upload_btn.create_rectangle(
            5, 5, 145, 45,
            fill="#f0f0f0",
            outline="#cccccc",
            width=2
        )
        self.upload_btn.create_text(
            75, 25,
            text="Upload Music",
            font=("Arial", 12, "bold"),
            fill="#333333"
        )
        
        # Rebind events
        self.upload_btn.bind("<Button-1>", self.on_upload_click)
        self.upload_btn.bind("<ButtonRelease-1>", self.on_upload_release)

    def load_background_image(self, img_path):
        self.bg_image_original = Image.open(img_path)
        self.resize_background()
    
    def resize_background(self):
        win_width = self.root.winfo_width()
        win_height = self.root.winfo_height()
        
        if win_width < 1 or win_height < 1:
            return
        
        resized_img = self.bg_image_original.resize((win_width, win_height), Image.LANCZOS)
        self.bg_image = ImageTk.PhotoImage(resized_img)
        self.bg_canvas.delete("bg")
        self.bg_canvas.create_image(0, 0, image=self.bg_image, anchor="nw", tags="bg")
    
    def on_window_resize(self, event):
        self.resize_background()
        
        if hasattr(self, 'upload_frame_window'):
            self.bg_canvas.coords(self.upload_frame_window, self.root.winfo_width()/2, self.root.winfo_height()/2)
        
        if hasattr(self, 'playback_frame_window'):
            self.bg_canvas.coords(self.playback_frame_window, self.root.winfo_width()/2, self.root.winfo_height()/2)
    
    def on_upload_click(self, event):
        self.upload_btn.create_rectangle(
            5, 5, 145, 45, 
            fill="#e0e0e0", 
            outline="#cccccc", 
            width=2
        )
        self.upload_btn.create_text(
            75, 25, 
            text="Upload Music", 
            font=("Arial", 12, "bold"), 
            fill="#333333"
        )
    
    def on_upload_release(self, event):
        self.upload_btn.create_rectangle(
            5, 5, 145, 45, 
            fill="#f0f0f0", 
            outline="#cccccc", 
            width=2
        )
        self.upload_btn.create_text(
            75, 25, 
            text="Upload Music", 
            font=("Arial", 12, "bold"), 
            fill="#333333"
        )
        
        file_path = filedialog.askopenfilename(
            title="Select Music File",
            filetypes=[("Audio Files", "*.mp3 *.wav *.ogg"), ("All Files", "*.*")]
        )
        
        if file_path:
            self.load_music(file_path)
            self.show_playback_controls()
    
    def show_playback_controls(self):
    # Remove upload frame completely
        if hasattr(self, 'upload_frame_window'):
            self.bg_canvas.delete(self.upload_frame_window)
            del self.upload_frame
            del self.upload_btn
        
        # Force complete UI update
        self.root.update_idletasks()
        self.root.update()
        
        # Show playback frame centered
        self.playback_frame_window = self.bg_canvas.create_window(
            self.root.winfo_width()/2, 
            self.root.winfo_height()/2,
            window=self.playback_frame
        )
        
        # Ensure proper focus and rendering
        self.playback_frame.update_idletasks()
        self.playback_frame.lift()
        self.root.focus_force()
        
        self.load_cover_image()

    
    def load_cover_image(self):
        self.cover_canvas.delete("all")
        
        try:
            #cover_path = "/Users/mac/Desktop/AirPod-Project/small_bg.jpeg"
            cover_path = "small_bg.jpeg"
            img = Image.open(cover_path)
            img = img.resize((300, 300), Image.LANCZOS)
            self.cover_img = ImageTk.PhotoImage(img)
            self.cover_canvas.create_image(0, 0, image=self.cover_img, anchor="nw")
        except:
            self.cover_canvas.create_text(
                150, 25,
                text="Now Playing: " + os.path.basename(self.current_song),
                font=("Arial", 10),
                fill="white"
            )
    
    def draw_play_icon(self):
        self.play_pause_btn.delete("all")
        self.play_pause_btn.create_polygon(
            10, 5,
            10, 35,
            30, 20,
            fill="black",
            outline="black"
        )
    
    def draw_pause_icon(self):
        self.play_pause_btn.delete("all")
        self.play_pause_btn.create_rectangle(
            10, 5,
            15, 35,
            fill="black",
            outline="black"
        )
        self.play_pause_btn.create_rectangle(
            25, 5,
            30, 35,
            fill="black",
            outline="black"
        )
    
    def toggle_play(self, event=None):
        if not self.current_song:
            print("No song loaded.")
            return
            
        if not self.playing:
            print("Playing song:", self.current_song)
            pygame.mixer.music.play()
            self.draw_pause_icon()
            self.playing = True
            self.update_seek_bar()
        else:
            print("Pausing song.")
            pygame.mixer.music.pause()
            self.draw_play_icon()
            self.playing = False
            if self.slider_update_id:
                self.root.after_cancel(self.slider_update_id)

    def update_seek_bar(self):
        if self.playing:
            current_pos = pygame.mixer.music.get_pos() / 1000
            self.seek_bar.set(current_pos)
            
            song_length = self.seek_bar.cget("to")
            self.time_label_left.config(text=time.strftime('%M:%S', time.gmtime(current_pos)))
            self.time_label_right.config(text=time.strftime('%M:%S', time.gmtime(song_length)))
            
            self.slider_update_id = self.root.after(100, self.update_seek_bar)
    
    def on_seek(self, value):
        if self.current_song and self.playing:
            pygame.mixer.music.set_pos(float(value))
    
    def load_music(self, file_path):
        if not os.path.exists(file_path):
            print("Error: File not found!")
            return
        
        self.current_song = file_path
        pygame.mixer.music.load(file_path)
        song_length = pygame.mixer.Sound(file_path).get_length()
        self.seek_bar.config(to=song_length)
        self.time_label_right.config(text=time.strftime('%M:%S', time.gmtime(song_length)))

    # Audio Classification Functions
    def create_waveforms_and_standard_waveform(self, filepath, sample_rate=44100):
        try:
            waveform, org_samplerate = torchaudio.load(filepath)
        except RuntimeError as e:
            print(f"Failed to load {filepath}. Is FFmpeg installed? Error: {e}")
            sys.exit(1)

        if org_samplerate != sample_rate:
            resampler = T.Resample(orig_freq=org_samplerate, new_freq=sample_rate)
            waveform = resampler(waveform)
        return waveform, sample_rate

    def create_single_channel(self, waveform):
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform

    def create_melspectrogram(self, waveform, sample_rate=44100):
        mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=512,
            n_mels=64
        )
        mel_spec = mel_transform(waveform)
        return mel_spec

    def trim_or_pad(self, waveform, max_duration=5, sample_rate=44100):
        max_len = max_duration * sample_rate
        if waveform.shape[1] > max_len:
            waveform = waveform[:, :max_len]
        else:
            padding = max_len - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        return waveform, sample_rate

    def preprocess(self, data_path, sample_rate=44100, duration=5):
        waveform, sample_rate = self.create_waveforms_and_standard_waveform(data_path, sample_rate)
        waveform = self.create_single_channel(waveform)
        waveform, sample_rate = self.trim_or_pad(waveform, duration, sample_rate)
        mel_spec = self.create_melspectrogram(waveform, sample_rate)
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-6)
        mel_spec = mel_spec.unsqueeze(0)  
        return mel_spec

    class AudioCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ELU(),
                nn.MaxPool2d(2),
                nn.Dropout2d(0.3),

                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ELU(),
                nn.MaxPool2d(2),
                nn.Dropout2d(0.3),

                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ELU(),
                nn.MaxPool2d(2),
                nn.Dropout2d(0.3),

                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ELU(),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ELU(),
                nn.Dropout(0.5),
                nn.Linear(128, 3)
            )

        def forward(self, x):
            x = self.conv(x)
            x = self.fc(x)
            return x

    def load_model_from_checkpoint(self, checkpoint_path):
        model = self.AudioCNN()
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint)
        model.eval()
        return model

    def record_and_predict_every_5s(self):
        print("Audio classification started. Predictions will appear in terminal...")
        sample_rate = 44100
        duration = 5
        
        while self.classification_active:
            try:
                audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
                sd.wait()

                waveform = torch.tensor(audio.T, dtype=torch.float32)
                waveform = waveform / (waveform.abs().max() + 1e-9)

                if waveform.abs().mean() < 0.01:
                    timestamp = time.strftime("%H:%M:%S")
                    print(f"[{timestamp}] → Too quiet, skipping")
                    continue

                torchaudio.save("temp.wav", waveform, sample_rate)
                mel_spec = self.preprocess("temp.wav")
                
                with torch.no_grad():
                    output = self.model(mel_spec)
                    pred = torch.argmax(output, dim=1).item()
                    label = self.label_map.get(pred, "Unknown")
                    timestamp = time.strftime("%H:%M:%S")
                    print(f"[{timestamp}] → Predicted: {label}")

                    # 🔉 Adjust volume based on prediction
                    if label == "Speaking":
                        pygame.mixer.music.set_volume(self.reduced_volume)
                    else:
                        pygame.mixer.music.set_volume(self.default_volume)


            except Exception as e:
                print(f"Error during classification: {e}")
                time.sleep(1)
    
    def start_classification_thread(self):
        self.classification_active = True
        classification_thread = threading.Thread(target=self.record_and_predict_every_5s, daemon=True)
        classification_thread.start()
    
    def on_close(self):
        self.classification_active = False
        
        # Clean up temporary audio files
        temp_files = ["temp.wav"]  # Add any other temp files you create
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    print(f"Deleted temporary file: {temp_file}")
            except Exception as e:
                print(f"Error deleting temporary file {temp_file}: {e}")
        
        # Ensure pygame shuts down cleanly
        pygame.mixer.quit()
        
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = MusicPlayer(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()