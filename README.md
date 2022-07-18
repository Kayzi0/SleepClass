# SleepClass
Sleep Stage Classification and Data Augmentation.

## Quickstart : Running the model

To run the model:
  - get the data from https://cloud.imi.uni-luebeck.de/s/MqMQCWWdDr8ZRcL and extract it into the folder
  - run `preprocessing.ipynb`
  - run `augment.ipynb`
  - if you want to train a network yourself, run `train.ipynb`. Adjust the `epochs` if necessary.
  - run `test.ipynb`
## Approach
We use a fairly general-purpose deep-learning approach to perform the classification. Research has shown that CNNs are very good at extracting features themselves, so very little preprocessing and no feature engineering was needed. For augmentation, simple gaussian noise augmentation was used. We also compared the performance differences of using weighted classes.

## Preparing the data
For our approach, we decided to use the 10Hz downsampled data, as we initially planned to use a VAE for augmentation, and the lower frequency would have allowed the very time-intensive training of a generative network to be a bit faster. In the end we discarded the approach due to time constraints, but stuck to the 10Hz data. 

This means that no further downsampling or resampling was needed and the data just had to be extracted into properly formatted arrays and the sliding windows had to be created. As non-overlapping windows were suggested, simply creating views of a full time series implicitly created these windows for us. For more detail on the preprocessing, refer, to `preprocessing.ipynb`.

## Data Augmentation and class weights
For data augmentation, we used a simple gaussian noise augmentation. We only augmented data with the N1 label, as that was the significantly underrepresented class in the data, in order to make the distribution more even. 

Additionally, during training we compared using equal weights and inverse square weights in order to reduce the effects of class imbalance. 

## Training
For training, an 80-20 split into training and validation data was used. The training loop is a fairly run of the mill PyTorch training loop. 
For the loss, a CrossEntropyLoss with inverse square adjusted weights was used, as this is most suitable for a multi-class classification.
The optimizer is a standard Adam optimizer with initial `lr = 0.001` and a learning rate scheduler `ReduceLROnPlateu`, which dynamically adjusts the learning rate once the loss hits a plateau.

## Peformance
Due to resource constraints we were not able to train a network for a longer amount of time. The training performance was fairly satisfactory, with the best run (weighted loss and augmented data) netting an accuracy and an f1 score of 83% during validation. The augmentation improved the accuracy by around 3%, as our best run without the augmented data yielded about 80%. The loss functions can be inspected in `train.ipynb`. There seems to either be some sort of error in our code, or the network seems to generalise incredibly poorly, as the testing results (seen below) are way worse than the validation. We have not been able to find out what is causing this issue, also due to time constraints.

Nevertheless, even in the possibly faulty testing, the impact of even this simple data augmentation can be seen. More elaborate techniques, such as generative models would most likely provide better results.


## Results of Testing

### No augmentation, no weighted loss
Final accuracy: 0.2477 \
mean avg precision 0.2167 \
f1 score: 0.1833

### No augmentation, weighted loss
Final accuracy: **0.3349** \
mean avg precision 0.2251 \
f1 score: 0.2560

### Augmentation, no weighted loss
Final accuracy: 0.3303 \
mean avg precision 0.2083 \
f1 score: 0.2946

### Augmentation, weighted loss
Final accuracy: 0.3303 \
mean avg precision **0.2303** \
f1 score: **0.3154**