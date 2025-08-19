Volume Simulation 

Description:

This project is inspired by the intelligent volume adjustment feature in Apple AirPods, which lowers the music volume when voice input is detected. However, traditional implementations often mistake singing along for speech and reduce the volume which will not give a good listening experince. 
Hence, this project is built to predict if input is noise, speech or singing and based on that, if input is singing or noise, the volume will note reduce but if it is speaking only then the volume will reduce.


Dataset:

The dataset consits of three folder, each having cleaned audios of sining, speech and noise to train a CNN model and save the best model to use it to predict the input while listening to music.
The dataset links are as follows 

Singing: https://drive.google.com/file/d/1LJAxozDu7r9jFe5qtdL55J7yJsyvt4QV/view?usp=sharing

Speech: https://drive.google.com/file/d/1dY4LjmzLQLIJxLdT_UDhFzKhVsKsH1ff/view?usp=sharing

Noise: https://drive.google.com/file/d/1rMORIyHCEB0l4CMoBl33rDLRJlM1qV0a/view?usp=sharing


Methodology:

The project workflow can be divided into two main phases: Model Training and Real-time Prediction & Volume Control.

1. Model Training

a. Data Preprocessing: All audio files were standardized by converting them to mono, resampling to 44.1 kHz, and trimming or padding them to a fixed duration of 5 seconds. Mel spectrograms with 64 mel bins were computed for each audio sample, then normalized to ensure consistent scale across the dataset.

b. Model Architecture: A Convolutional Neural Network (CNN) was designed to classify audio into three categories: singing, speech, and noise.
The architecture consists of two convolutional layers (with ReLU activation and max pooling), followed by fully connected layers for classification.
Cross-entropy loss was used, and training was performed using the Adam optimizer.
The model with the highest validation accuracy was saved for deployment.

2. Real-time Prediction & Volume Control

a. Live Audio Capture: A microphone stream is continuously monitored using the sounddevice library. Incoming audio is segmented into 5-second windows and preprocessed in the same way as the training data.

b. Inference: The trained CNN model predicts the class of each segment in real time.

c. Dynamic Volume Adjustment: If the detected class is speech, the system executes an AppleScript command to lower the system volume while the music is playing. If the detected class is singing or noise, the volume remains unchanged to avoid disrupting the listening experience.

Results:

The proposed CNN model achieved an accuracy of 93.16% on the test dataset. This indicates that the system can reliably distinguish between singing, speech, and background noise in most cases. The evaluation was conducted on unseen audio samples after preprocessing into mel spectrograms, ensuring that the model’s performance reflects real-world applicability. While accuracy is the primary reported metric, qualitative testing also showed that the model responded promptly and consistently during live audio simulation.

Drawbacks and future scope:

1. This project works best for enligh langauges and mostly causes errors for other languages.
2. Better results can be obtained by using larger datasets.
3. If the model predicts singing, the system can be made more accurate to make sure that the singing input is the same song as the song being played.
4. The system will work well only when a listening device is used, if the song is played on speaker, then the model will listen to the song and predict input as singing, future model can use a loopback filter or virtual audio routing to exclude the device’s own playback from the microphone input.

Applications:

1. Smart headphones and earphones.
2. Video conferencing noise suppression with very slight change in code.
3. Real-time voice/music mixing tools.

## 🎥 Demo

▶ [Watch the Demo Video](https://drive.google.com/file/d/15p-LMeNbUBdl0f0drQRK1IcZYpIammoV/view?usp=drive_link)

Due to the limitaion of not being able to record what is being played, only input voice is heard in the video, but when the method is followed and a bluetooth device is used, the audio hearing of song and prediction of the sound as input in microphone will happen.

