import os
import clip
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import PrecisionRecallDisplay

# Load the clip model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)


class UnknownForest(Dataset):
    
    def __init__(self, db_path, resize):
        
        super().__init__()
        
        self.db_path = db_path
        self.data = []
        self.resize = resize
        # self.img_dim = (512, 512) # original images: (896, 512)
    
        # create image_paths and labels
        for folder in sorted(os.listdir(os.path.join(self.db_path))):
            for file in os.listdir(os.path.join(self.db_path, folder)):
                file_path = os.path.join(os.path.join(self.db_path, folder), file)
                class_name = file_path.split('/')[-2]
                self.data.append([file_path, class_name])
        self.class_map = {'data_good': 0, 'data_bad':1}
        
    def  __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        img, class_name = self.data[idx] #str, str
        label = self.class_map[class_name] # int

        # transforming the image using pytorch 'transforms' function
        tf = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),  # str path -> img data
            # size = (height, weight)
            # size = a => smaller edge of the image will be matched to this number, use a sequence of length 1: [size, ]
                # height > width: (size * height / width, size)
            # transforms.Resize(size)
            transforms.Resize((int(self.resize * 1.25), int(self.resize * 1.25))),
            # transforms.Resize((self.img_dim)),
            transforms.RandomRotation(15),
            transforms.CenterCrop(self.resize),
            # transforms.CenterCrop((self.img_dim)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # from imagenet
        ])
        
        img = tf(img)
        label = torch.tensor(label)

        return img, label
    
    
def get_features(dataset):
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size=100)):
            features = model.encode_image(images.to(device))

            all_features.append(features)
            all_labels.append(labels)

    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()


# test performance metrics
def test_perform (test_labels, predictions_test):
    
    # accuracy
    acc_s = np.mean((test_labels == predictions_test).astype(float)) * 100.
    
    # ROC score
    roc_s = roc_auc_score(test_labels, predictions_test)

    # Recall score 
    # 'micro': calculate metrics globally by counting the total true positives, false negatives and false positives.
    # 'macro': calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
    # 'weighted': calculate metrics for each label, and find their average weighted by support (the number of true instances for each label). 
    #             This alters ‘macro’ to account for label imbalance; Weighted recall is equal to accuracy.
    recall_s_micro = recall_score(test_labels, predictions_test, average='micro')
    recall_s_macro = recall_score(test_labels, predictions_test, average='macro')
    recall_s_weighted = recall_score(test_labels, predictions_test, average='weighted')
    recall_s = recall_score(test_labels, predictions_test, average= None)
    
    # Average Precision (AP) score 
    ap_s = average_precision_score(test_labels, predictions_test)
    
    print(f' >>>>>>>> Logistic Regression (Test Data) (no reg.): \n Accur. = {acc_s} \n ROC.Score = {roc_s} \n Recall.Score [goodImg (0) as good, badImg (1) as bad] = {recall_s} \n AveragePrecison.Score = {ap_s}')
    
    # plot: ROCAUC  
    display = RocCurveDisplay.from_predictions(test_labels, predictions_test, name="Logistic Reg.")
    _ = display.ax_.set_title("2-class ROC curve")
    # plot: Precison Recall
    display = PrecisionRecallDisplay.from_predictions(test_labels, predictions_test, name="Logistic Reg.")
    _ = display.ax_.set_title("2-class Precision-Recall curve")
