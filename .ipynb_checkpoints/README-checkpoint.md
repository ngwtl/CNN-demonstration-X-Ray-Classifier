## Capstone - Transfer learning to assist in pneumonia diagnosis using chest X-ray images
---


### Introduction and problem statement

Lung conditions such as pneumonia, growths and nodules that may lead to lung cancer and infiltrates in the lung are debilitating and potentially life-threatening conditions widespread in all societies. This is especially true against the backdrop of COVID-19 where [15%](https://www.webmd.com/lung/ards-acute-respiratory-distress-syndrome) of all COVID-19 patients develop severe, life-threatening pneumonia-related complications.

In the non-COVID-19 world, pneumonia accounts for over 15% of all deaths of children under 5 years old internationally. In 2015, 920,000 children under the age of 5 died from the disease. In the United States, pneumonia accounts for over 500,000 visits to emergency departments and over 50,000 deaths in 2015 , keeping the ailment on the list of top 10 causes of death in the country.
In Singapore, statistics are even more compelling where pneumonia accounted for [20.6](https://www.healthxchange.sg/heart-lungs/lung-conditions/pneumonia-causes-symptoms#:~:text=Pneumonia%2C%20a%20serious%20inflammatory%20condition,Ministry%20of%20Health%20(MOH).) per cent of deaths in Singapore in 2018, just behind cancer, the top killer, which caused 28.8 per cent of deaths.

Although lung conditions are common, accurately diagnosing pneumonia is a tall order. It requires review of a chest radiograph (CXR) by highly trained specialists and confirmation through clinical history, vital signs and laboratory exams. Pneumonia usually manifests as an area or areas of increased opacity on CXR. However, the diagnosis of pneumonia on CXR is complicated because of a number of other conditions in the lungs such as fluid overload (pulmonary edema), bleeding, volume loss (atelectasis or collapse), lung cancer, or post-radiation or surgical changes. Outside of the lungs, fluid in the pleural space (pleural effusion) also appears as increased opacity on CXR. When available, comparison of CXRs of the patient taken at different time points and correlation with clinical symptoms and history are helpful in making the diagnosis. Though onerous, CXR analysis is still one of the quickest and most efficient methods of first level screening compared to other more time-consuming radiography methods such as CT scans and MRIs.


The COVID-19 pandemic has exacerbated the resource scarcity within hospitals and there is a strong need to develop a quicker and more efficient way of screening and diagnosing life-threatening pneumonia and lung conditions in order to prevent hospital resources from being overwhelmed. This project attempts to address the issue of speedy lung condition diagnosis by utilising transfer learning of pre-trained convolutional neural networks (CNNs) on CXRs to develop a classification model. This model can then be deployed as a web application in order to develop a diagnostic tool to assist in speedy lung condition diagnosis. 




### Executive Summary
For this project, we will consider 4 datasets, namely, train, test, spray and weather.

Once the datasets are imported, we will explore each feature. Feature engineering comes next as we transform the date and weather features. Categorical features are also transformed to dummy variables.

Finally, we will train our model using GridSearch, of which, the best model will be used for our Kaggle submission.


### Modelling
---
<img src="https://github.com/ngwtl/X-Ray-classifier/blob/master/Model%20results/model_performance.png" width="500"/>



Transfer learning takes what a model has learned while solving one problem and applies it towards a new application. For instance, the knowledge that a neural network that has already been pre-trained to recognise objects such as cats or dogs can be used to help do a better job in reading X-ray images. 

Using XGBoost (our best performing model), we achieved an ROC_AUC of **0.8511**. GridSearchCV and cross_val_score were used to tune the respective classifier with the optimised hyperparameters and then check if the learning algorithm is capable of yielding high mean score and low variances as we would want the selected learning algorithm to be capable of producing similar performance on unseen data. In this case it will be the test set provided by kaggle.

By comparing the above scores, the final learning algorithm that is choosen is XGBoost. Using the best parameters selected are `colsample_bytree = 0.2`, `gamma = 0.03`, `learning_rate = 0.1`, `max_depth = 3`, `reg_alpha = 0`, `reg_lambda = 1`, `subsample = 0.5`.
    
### Deployment
<img src="https://github.com/ngwtl/X-Ray-classifier/blob/master/demo.gif" width="500"/>

### Summary of Findings & Recommendations

Examination of the the total costs of spraying the whole of Chicago compared against the benefits show that the costs far outweigh the benefits in monetary terms. At best, accounting for inflation or even a pessimistic outcome of having 50% more WNV infections, the total monetary benefit to Chicago as a society may only be 16%-25% of the total cost of spraying.

However, our model does not take into account any non-monetary benefits to reducing the mosquito population. These include the emotional costs from loss of life, the reduction in the need for enhanced testing for suspected WNV cases and public confidence in the government.

From previous geospatial analysis of spray data, there is a distinct lack of evidence to support the claim that mosquito spraying had any effect on the reduction of WNV-infected mosquitos. Furthermore, the spray data pointed towards highly fragmented and haphazardous spraying operations that did not seem to be driven by the evidence if the presence and severity of WNV mosquito infestations. Traps such as the T900 trap at O'Hare International airport which proved to capture the most WNV-infected mosquitos by far were not sprayed.

Given the high costs required to conduct spraying operations, we hence recommend the following action points:

- Re-examine the effectiveness of spraying Zenivex™ E4 as a means to control the mosquito population. Evidence points towards the ineffectiveness of the chemical and it is likely that other kinds of non-toxic mosquito sprays should be explored.

- Re-direct mosquito spraying operations in a more organised and evidence-driven manner whereby severe hotspots such as O'Hare International Airport are sprayed first at the beginning of summer in order to prevent large populations of mosquitos forming. In addition, spraying operations should be accurately logged and routes planned to make sure mosquito breeding sites are properly covered.

- Examine new ways of controlling the mosquito population that may arguably cost less than spraying the whole of Chicago. Innovative ways of doing so may include 'anti-mosquito' campaigns done in places such as Singapore or encouraging citizens of Chicago to get rid of stagnant water sources.


### Folder Organisation:
---
    |__ code
    |   |__ 01_eda.ipynb   
    |   |__ 02_modeling.ipynb    
    |__ assets
    |   |__ train.csv
    |   |__ test.csv
    |   |__ processed.zip    
    |   |__ weather.csv
    |   |__ spray.csv
    |   |__ mapdata_copyright_openstreetmap_contributors.txt
    |__ chicago_geodata
    |   |__ geo_export_55fcb48c-7621-4c8a-999c-9fb3c86e8950.dbf
    |   |__ geo_export_55fcb48c-7621-4c8a-999c-9fb3c86e8950.prj
    |   |__ geo_export_55fcb48c-7621-4c8a-999c-9fb3c86e8950.shp
    |   |__ geo_export_55fcb48c-7621-4c8a-999c-9fb3c86e8950.shx
    |__ planning_doc.xlsx
    |__ group_presentation.pdf
    |__ README.md