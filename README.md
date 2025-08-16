#Volume Simulation 

Description:

This project is inspired by the intelligent volume adjustment feature in Apple AirPods, which lowers the music volume when voice input is detected. However, traditional implementations often mistake singing along for speech and reduce the volume which will not give a good listening experince. 
Hence, this project is built to predict if input is noise, speech or singing and based on that, if input is singing or noise, the volume will note reduce but if it is speaking only then the volume will reduce.


Dataset:

The dataset consits of three folder, each having cleaned audios of sining, speech and noise to train a CNN model and save the best model to use it to predict the input while listening to music
The dataset links are as follows 

Singing: https://drive.google.com/file/d/1LJAxozDu7r9jFe5qtdL55J7yJsyvt4QV/view?usp=sharing

Speech: https://drive.google.com/file/d/1dY4LjmzLQLIJxLdT_UDhFzKhVsKsH1ff/view?usp=sharing

Noise: https://drive.google.com/file/d/1rMORIyHCEB0l4CMoBl33rDLRJlM1qV0a/view?usp=sharing


Methodology:

The project workflow can be divided into two main phases: Model Training and Real-time Prediction & Volume Control.

1. Model Training

Data Preprocessing:

All audio files were standardized by converting them to mono, resampling to 44.1 kHz, and trimming or padding them to a fixed duration of 5 seconds.
Mel spectrograms with 64 mel bins were computed for each audio sample, then normalized to ensure consistent scale across the dataset.

Model Architecture:

A Convolutional Neural Network (CNN) was designed to classify audio into three categories: singing, speech, and noise.
The architecture consists of two convolutional layers (with ReLU activation and max pooling), followed by fully connected layers for classification.
Cross-entropy loss was used, and training was performed using the Adam optimizer.
The model with the highest validation accuracy was saved for deployment.

2. Real-time Prediction & Volume Control

Live Audio Capture:

A microphone stream is continuously monitored using the sounddevice library. Incoming audio is segmented into 5-second windows and preprocessed in the same way as the training data.

Inference:

The trained CNN model predicts the class of each segment in real time.

Dynamic Volume Adjustment:

If the detected class is speech, the system executes an AppleScript command to lower the system volume while the music is playing.
If the detected class is singing or noise, the volume remains unchanged to avoid disrupting the listening experience.
