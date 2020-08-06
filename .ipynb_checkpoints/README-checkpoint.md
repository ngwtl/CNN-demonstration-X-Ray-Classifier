## Capstone - Transfer learning to assist in pneumonia diagnosis using chest X-ray images


### Introduction and problem statement
---
Lung conditions such as pneumonia, nodules that may lead to lung cancer and infiltrates  are debilitating and potentially life-threatening conditions widespread in all societies. This is especially true against the backdrop of COVID-19 where [15%](https://www.webmd.com/lung/ards-acute-respiratory-distress-syndrome) of all COVID-19 patients develop severe, life-threatening pneumonia-related complications.

In the non-COVID-19 world, pneumonia accounts for over 15% of all deaths of children under 5 years old internationally. In 2015, 920,000 children under the age of 5 died from the disease. In the United States, pneumonia accounts for over 500,000 visits to emergency departments and over 50,000 deaths in 2015 , keeping the ailment on the list of top 10 causes of death in the country.
In Singapore, statistics are even more compelling where pneumonia accounted for [20.6](https://www.healthxchange.sg/heart-lungs/lung-conditions/pneumonia-causes-symptoms#:~:text=Pneumonia%2C%20a%20serious%20inflammatory%20condition,Ministry%20of%20Health%20(MOH).) per cent of deaths in Singapore in 2018, just behind cancer, the top killer, which caused 28.8 per cent of deaths.

<img src="https://github.com/ngwtl/X-Ray-classifier/blob/master/Images/x-ray%20samples.png" width="600"/>

Although lung conditions are common, accurately diagnosing pneumonia is a tall order. It requires review of a chest radiograph (CXR) by highly trained specialists and confirmation through clinical history, vital signs and laboratory exams. Pneumonia usually manifests as an area or areas of increased opacity on CXR. However, the diagnosis of pneumonia on CXR is complicated because of a number of other conditions in the lungs such as fluid overload (pulmonary edema), bleeding, volume loss (atelectasis or collapse), lung cancer, or post-radiation or surgical changes. Outside of the lungs, fluid in the pleural space (pleural effusion) also appears as increased opacity on CXR. When available, comparison of CXRs of the patient taken at different time points and correlation with clinical symptoms and history are helpful in making the diagnosis. Though onerous, CXR analysis is still one of the quickest and most efficient methods of first level screening compared to other more time-consuming radiography methods such as CT scans and MRIs.


The COVID-19 pandemic has exacerbated the resource scarcity within hospitals and there is a strong need to develop a quicker and more efficient way of screening and diagnosing life-threatening pneumonia and lung conditions in order to prevent hospital resources from being overwhelmed. This project attempts to address the issue of speedy lung condition diagnosis by utilising transfer learning of pre-trained convolutional neural networks (CNNs) on CXRs to develop a classification model. This model can then be deployed as a web application in order to develop a diagnostic tool to assist in speedy lung condition diagnosis. 




### Executive Summary
---
An image classifier based on the InceptionNetV3 architecture was trained using 30,000 images taken from the RSNA pneumonia lung challenge dataset. The images were classfied into 3 categories:
1) Pneumonia
2) Normal
3) No pneumonia, but likely other complications.

The model, which achieved an accuracy score of 72% was deployed using streamlit as a web app image classifier that can be used as a diagnostic tool for physicians. 

### Data
---

<img src="https://github.com/ngwtl/X-Ray-classifier/blob/master/Images/dataset%20info.png" width="600"/>


X-Ray data was obtained from multiple different sources including the [NIH's](https://nihcc.app.box.com/v/ChestXray-NIHCC) open source dataset, [JSRT](http://db.jsrt.or.jp/eng.php) lung nodule dataset, the [Guangzhou Women's and Children's hospital](https://www.qmenta.com/covid-19-kaggle-chest-x-ray-normal/) dataset, the [Covid-19 open-source dataset](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge) and the [RSNA](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge) Pneumonia Detection Challenge. 

Due to the imbalanced nature of the data, only the RSNA data was used to train the model. Images of other different lung anomalies were chosen from the other different datasets in order to act as a holdout set for deployment training.
Normalised counts of classes establised the baseline score of 39% for the three classes of `No Lung Opactity/ Not Normal`,`Normal` and `Lung Opacity`.

If using the code for deployment, please download the [RSNA](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge) Pneumonia Detection Challenge dataset for code application. 

### Modelling
---
<img src="https://github.com/ngwtl/X-Ray-classifier/blob/master/Model%20results/model_performance.png" width="600"/>



Transfer learning takes what a model has learned while solving one problem and applies it towards a new application. For instance, the knowledge that a neural network that has already been pre-trained to recognise objects such as cats or dogs can be used to help do a better job in reading X-ray images. 

Using Keras's in-built functions and weights, 6 different convolutional newtwrok architectures (VGG16, Xception, InceptionResnetV2, ResNet50, Densenet121 and InceptionnetV3) were trained with preset "ImageNet" weights to determine which was the best candidate to take to production.

| Model 	| Accuracy (%) 	|
|-	|-	|
| InceptionnetV3 	| 72.08 	|
| Densenet121 	| 67.4 	|
| Resnet50 	| 65.2 	|
| InceptionResnetV2 	| 46.875 	|
| Xception 	| 42.23 	|
| VGG16 	| 37.5 	|

By comparing the above scores, the final convolutional network architecture chosen for deployment is InceptionNetV3 due to its performance.

Full models with weights and architectures can be downloded from [here](https://drive.google.com/drive/folders/1o4f_wEdg8dkKS_6vFnryg_pEMYzNrbv0?usp=sharing).
### Deployment
---
The fully-trained InceptionNetV3 model was deployed as a webapp using [streamlit](https://www.streamlit.io/). 
The deployed model (named Dr. Glava Tikvah) consists of a simple image uploader that allows a diagnostician or physician to upload CXRs.
The app returns a simple diagnosis (0: Pneumonia, 1: No Pneumonia but something else, 3: Normal) and their respective predicted probabilities. The webapp can be run locally and is entitled `app.py` in the `code` folder of this repository. Streamlit is required to run the app.


<img src="https://github.com/ngwtl/X-Ray-classifier/blob/master/demo.gif" width="500"/>

### Summary of Findings & Recommendations

Using RSNA data, a model based on the InceptionNetV3 architecture was trained to an accuracy of 71% in the classifying of CXR images. This was done with minimal image pre-processing and with pre-trained weights from "ImageNet".

Deployment was successful using Streamlit as a wrapper as a demonstration for. webapp that can act as a helpful diagnosis tool for physicians, just in case of human error. 

There are certain limitations to the model. For instance, the current accuracy is not good enough for the deployment to act as an independent diagnostic tool. All predictions must still be clinically correlated. 
Also, the quality of the prediction is vastly influenced by the quality of the X-ray image fed into the webapp. Skill of radiologist plays an important factor in the functioning of the webapp. 

Given the current limitations of the model further work can be done to enhance the effectiveness of the webapp. These include:

- Training the model on larger datasets with different kinds of pneumonia.

- Implementing a grad-cam or heatmap imaging system in order to allow the user to visualise areas of interest.

- Experiment further with alternaive methods of image processing, such as playing with the transparency or subtracting backgrounds in orer to isolate the lungs.

