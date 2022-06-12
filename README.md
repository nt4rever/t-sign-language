# Sign Language Detector
A simple CNN project for detecting american sign language.
Here, I have implemented CNN (Convolution Neural Network) using Keras.

### Tools Used
1. Python 3.7
2. OpenCV 4
3. Tensorflow 2.8.2
4. Keras 2.8.0
5. FastApi[all]
6. Scikit-learn
7. Matplotlib

### Running this project
1. Install Python 3, Opencv 4, Tensorflow, Keras...
2. First Train the model.
    ```
    python cnn_model.py
    ```
2. Now to test the model you just need to run recognise.py . To do so just open the terminal and run following command.
    ```
    python recognize.py
    ```
    Adjust the hsv values from the track bar to segment your hand color.

3. To create your own data set.
    ```
    python capture.py
    ```
4. Run FastApi server.
    ```
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
    ```