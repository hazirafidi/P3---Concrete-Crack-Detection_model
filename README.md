# P3 - Concrete Crack Detection
 Concrete crack detection model training via Transfer Learning

## 1. Project Summary
This project is carried out to implement Deep Learning on Model Training for Image Classification via Transfer Learning. The objective of this project is to train a model that be able to detect concrete crack images. The dataset is retrieved from [here] (https://shrdc-my.sharepoint.com/personal/kong_kah_chun_shrdc_org_my/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fkong%5Fkah%5Fchun%5Fshrdc%5Forg%5Fmy%2FDocuments%2FDeep%20Learning%20TTT%2FImage%20Segmentation%2FData%2Fdata%2Dscience%2Dbowl%2D2018%2Ezip&parent=%2Fpersonal%2Fkong%5Fkah%5Fchun%5Fshrdc%5Forg%5Fmy%2FDocuments%2FDeep%20Learning%20TTT%2FImage%20Segmentation%2FData&ga=1)

## 2. IDE and Framework 
The project is built with Google Colab Notebook as the main IDE. The main frameworks used in this project are TensorFlow, Numpy, Matplotlib, OpenCV and Scikit-learn.
 
## 3. Methodology

The methodolgy of this project is inpired by Tensorflow Image Segmentation Tutorial. You can find the documentation in [here](https://www.tensorflow.org/tutorials/images/transfer_learning).

 
### 3.1 Input Pipeline

The dataset files contains a Negative folder and Positive folder, in the format of images. The images are splitted into train and validation data in the ratio of 80:20. Then, the validation data is further split into test data for model evaluation purpose. The input images are preprocessed with tensorflow preprocess_input method to ensure the input images are fit and right to the pre-trained model. Data augmentation is applied for the train dataset to increase variety of the images.


### 3.2 Model Pipeline 
The model architecture can be illustrated as in the figure below.
 
 ![image](https://user-images.githubusercontent.com/100177902/163772961-250d3caf-838a-424e-9b9d-aae1e631c2c8.png)
 
The base model use in this project is VGG16. The feature extraction from base model VGG16 is extracted and combined with own classifier, dropout and output layer as can be refer in figure above. In summary, the model consist of two components, which is the feature extraction layer and Fully Connected classification layer which will translate the image into its respective class.

![image](https://user-images.githubusercontent.com/100177902/163769994-ff4fc536-5da8-47bd-a014-bf5f020e8421.png)

The model is trained with initial epochs of 10. Early stopping is also applied in the model training to avoid overfitting. At the first Feature Extraction training, there is no early stopping. Then Fine-tune model is implemented to increase performance even further or "fine-tunes" the weights of the top layer of the VGG19 model alongside the classifier added. The training continued from the initial epochs with addition of 10 new epochs. The training stops at epoch 18, with a training accuracy of 99% and validation accuracy of 99%. The model training graphs for both Feature Extraction and Fine-Tune stages are shown in figures below.
 
 ![image](https://user-images.githubusercontent.com/100177902/163774559-e2ffce3f-6636-4e4d-8528-902470ea14d0.png)
 ![image](https://user-images.githubusercontent.com/100177902/163774585-6bc27e48-c8ae-42ac-8804-f033dcd0bb42.png)


## 4. Result 
The model is evaluated with test data, which is shown in figure below.
 
![image](https://user-images.githubusercontent.com/100177902/163774707-4511529f-4fad-4422-8a09-f9b4eff2d20f.png)

## 5. Conclusion
Transfer learning is widely use in deep learning model training approach as it is convenient and less resourceful consumption since the model is consructed based on the pre-trained model. With this method, we can save lot of time and energy as well as other resources. What we need is abundant of quality data so that the model performance is high. The model trained in this project managed to achieve both 99% accuracy and validation accuracy. Overall rhe model performance is very excellent.
