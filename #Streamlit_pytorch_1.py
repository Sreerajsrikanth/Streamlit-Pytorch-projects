#Streamlit_pytorch_1

import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch

from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2,FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes
weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT ##dictionairy
categories = weights.meta["categories"]
img_preprocess = weights.transforms() ##before it is given pre trained model

def load_model():
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.8)
    model.eval()
    return model
model = load_model()

def make_prediction(img):
    img_processed = img_preprocess(img)
    prediction = model(img_processed.unsqueeze(0))
    prediction = prediction[0]
    prediction["labels"] = [categories[label] for label in prediction["labels"]]
    return prediction

def create_image_with_bboxes(img,prediction):
    img_tensor = torch.tensor(img)
    img_with_boxes = draw_bounding_boxes(img_tensor,boxes=prediction["boxes"], labels=prediction["labels"],colors=["red" if label=="person" else "green" for label in prediction["labels"]],width=2)
    img_with_bboxes_np = img_with_boxes.detach().numpy().transpose(1,2,0)
    return img_with_bboxes_np

##Dashboard
st.title("Object Detector :dog: ")
upload = st.file_uploader(label = 'upload image here',type=["png","jpg","jpeg"])

if upload:
    img = Image.open(upload)
    prediction = make_prediction(img) ##Dictionairy
    img_with_bbox=create_image_with_bboxes(np.array(img).transpose(2,0,1), prediction)

    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111)
    plt.imshow(img_with_bbox)
    plt.xticks([],[])
    plt.yticks([],[])
    ax.spines[["top","bottom","right","left"]].set_visible(False)

    st.pyplot(fig, use_container_width=True)
    del prediction["boxes"]
    st.header("Predicted Probabilities")
    st.write(prediction)