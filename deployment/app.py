import gradio as gr
from fastai import *
from pathlib import Path, WindowsPath
from fastai.vision.all import *
import pickle

#Path._flavour = WindowsPath._flavour

with open("model_convnext_small_v1.pkl", "rb") as f:
    model = pickle.load(f)

#model = load_learner('model_convnext_small_v1.pkl')
vehicle_classes = model.dls.vocab

def vehicle_classifier(image) -> dict:
    #img = image.copy().resize((224, 224))
    img_pil = Image.fromarray(image)
    pred, idx, probs = model.predict(img_pil)
    return dict(zip(vehicle_classes, map(float, probs)))

input = gr.Image(image_mode="RGB")
output = gr.Label()
examples = ['0.jpg', '1.jpg', '2.jpg', '3.jpg']

demo = gr.Interface(fn=vehicle_classifier, inputs=input, outputs=output, examples=examples)
demo.launch()



