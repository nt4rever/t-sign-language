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

### Dataset
https://www.kaggle.com/datasets/tanvilla/asl-dataset
### Running this project
1. Install Python 3, Opencv 4, Tensorflow, Keras...
2. First Train the model.
    ```
   app/training-tsign.ipynb
   ```
2. Now to test the model you just need to run recognise.py . To do so just open the terminal and run following command.
    ```
    python app/recognize.py
    ```
    Adjust the hsv values from the track bar to segment your hand color.

3. To create your own data set.
    ```
    python app/capture.py
    ```
4. Run FastApi server.
    ```
    python app/api.py
    ```