import os
import clip
import torch
import cv2
import itertools
from PIL import Image

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
 
db_img = '~/data'
db_img_bad = './clip_classification/data_bad'
db_img_good = './clip_classification/data_good'



#%%
db_img_folders = os.listdir(db_img)

list_probabilities = []
list_probable = []
list_bad_images = []

for folder_name in db_img_folders:
    
    folder_path = os.path.join(db_img,folder_name)
    img_list = os.listdir(folder_path)
    
    for img in img_list:
        
        img_path = os.path.join(folder_path,img)

        # image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
        image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
        text = clip.tokenize(["forest painting framed in white", "forest painting on a white wall", "side by side photos of forest"]).to(device)
        text = clip.tokenize(["collage of forest photos on white background", "two forest photos on a white wall", "side by side photos of forest"]).to(device)
        text = clip.tokenize(["collage of forest photos on white background", "two forest photos on a white wall", "white background"]).to(device)
        text = clip.tokenize(["forest photos on white background", "forest photos on white wall", "nature photos on white background", 
                              "jungle photos on white background", "jungle photos on white wall", "nature photos on white wall"]).to(device) 
        
        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
            
            logits_per_image, logits_per_text = model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        
        print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
    
        list_probabilities.append(probs)
        
        probs_list = probs.tolist()[0]  # [0.337890625, 0.1781005859375, 0.484130859375]
        if any (p > 0.90 for p in probs_list):
            # print(f'this is probable: {probs_list}')
            list_bad_images.append(img_path)
            # print(f' \n bad image is in: {img_path} \n ')
            
            tmp = cv2.imread(img_path)
            cv2.imwrite(os.path.join(db_img_bad ,img_path.split('/')[-2] + '_' +img) , tmp)
        else:
            tmp = cv2.imread(img_path)
            cv2.imwrite(os.path.join(db_img_good,img_path.split('/')[-2] + '_' +img) , tmp)


            list_probable.append(probs_list)

        
        



        
        
        

