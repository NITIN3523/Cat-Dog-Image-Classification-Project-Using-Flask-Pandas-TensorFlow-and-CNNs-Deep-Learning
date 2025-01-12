from flask import *
from classifier import classify
import os

app = Flask(__name__)
app.secret_key = '!@#$%^&*()'

@app.route('/', methods = ['GET', 'POST'])
def classification_html():
    if request.method == 'POST':
        
        file = request.files['file']    
        
        imgpath = 'static/image_01.jpg'    
        file.save(imgpath)
        
        class_names = {0:'Cat', 1:'Dog'}        
        modelpath = 'Model/Cat_Dog_classfication_model.keras'
        class_name, confidence_score = classify(imgpath,modelpath,class_names)

        os.remove(imgpath)
        
        return jsonify({
            'Classification_Result' : class_name,
            'Classification_Confidence': confidence_score
        })
        
    return render_template('Classification.html')
    
if __name__ == '__main__':
    app.run(debug=True)
