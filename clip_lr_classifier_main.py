import torch
import os
import clip
import shutil
import numpy as np
from clip_lr_classifier_utils import UnknownForest, get_features, test_perform
from PIL import Image
from sklearn.linear_model import LogisticRegression

# Load the clip model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)


# create dataset:
    # train/test:   .../db_forest_balanced/data_good/img1.jpg, img2.jpg, ...
    #               .../db_forest_balanced/data_bad/img1.jpg, img2.jpg, ...
    # inference:    .../db_forest/folder1, folder2, .../img1.jpg, img2.jpg, ...

db_path_train = './clip_classification/db_forest_balanced_train' # 80%db_forest_balanced
db_path_test = './clip_classification/db_forest_balanced_test'   # 20%db_forest_balanced
db_path_inference = './clip_classification/db_forest'
db_path_inference_bad = './clip_classification/db_forest_inference/bad_1'
db_path_inference_good = './clip_classification/db_forest_inference/good_0'


#%% 

# create datasets        
UnknownForest_train = UnknownForest(db_path_train, 224)    
UnknownForest_test = UnknownForest(db_path_test, 224) 

# calculate image features using clip
train_features, train_labels = get_features(UnknownForest_train)
test_features, test_labels = get_features(UnknownForest_test)

# train 

# Perform logistic regression
classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
classifier.fit(train_features, train_labels)

# training accuracy
predictions_train = classifier.predict(train_features)
accuracy_train = np.mean((train_labels == predictions_train).astype(np.float)) * 100.
print(f' >>>>>>>> Logistic Regression Training Accu (no reg.): {accuracy_train:.3f}')

# test

predictions_test = classifier.predict(test_features)
test_perform (test_labels, predictions_test)

# inference (apply the classifier to unlabeled data and copy them to class categories: bad_1 and good_0)

infer_probabilities = []
counter = 1

for folder in sorted(os.listdir(os.path.join(db_path_inference))):
    for file in os.listdir(os.path.join(db_path_inference, folder)):
        
        file_path = os.path.join(os.path.join(db_path_inference, folder), file)
        image = preprocess(Image.open(file_path)).unsqueeze(0).to(device)
        # build features for images without labels
        with torch.no_grad():
            infer_features = model.encode_image(image)
        # Evaluate using the logistic regression classifier
        infer_predictions = classifier.predict(infer_features) #.cpu()) # numpy array of size 1
        infer_probabilities.append(infer_predictions)
        
        
        # copy classified images into a new dir
        new_img_name = os.path.join(file_path.split('/')[-1].split('.')[-2] + '_' + str(counter))
        image_suffix = file_path.split('/')[-1].split('.')[-1]
        
        if int(infer_predictions) == 1:
            shutil.copy(file_path, os.path.join(db_path_inference_bad, new_img_name) + '.' + image_suffix)
        elif int(infer_predictions) == 0:
            shutil.copy(file_path, os.path.join(db_path_inference_good, new_img_name) + '.' + image_suffix)
        counter += 1










