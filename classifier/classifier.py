import numpy as np
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import load_img,img_to_array # type: ignore

def preprocess_single_img(imgpath,target_size=(60,60)):
    image = load_img(imgpath,target_size=target_size)
    
    img_array = img_to_array(image)
    
    img_array = img_array / 255.0
    
    img_array = np.expand_dims(img_array,axis=0) # (1,60,60,3)
    
    return img_array

def classify(imagepath,modelpath,class_names):
    model = load_model(modelpath)
    
    preprocessed_img = preprocess_single_img(imagepath)
    
    prediction = model.predict(preprocessed_img)
    # [ [0.925554 0.85255] ]
    
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score =  str(str(int(prediction[0][index] * 100)) + '%')
    
    return class_name,confidence_score
    
    
    