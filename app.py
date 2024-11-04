## Importing the necessary libraries
from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import cv2
import os
import json
import re

## Creating the Flask app
app = Flask(__name__)

## Configuring the upload and model folders
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODEL_FOLDER'] = 'model'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        ##Handle model file upload
        model_file = request.files['model']
        model_path = os.path.join(app.config['MODEL_FOLDER'], secure_filename(model_file.filename))
        model_file.save(model_path)
        
        ##Load model to retrieve layer names
        model = load_model(model_path)
        layer_names = [layer.name for layer in model.layers if '_conv' in layer.name]
        
        ##Handle label dictionary upload
        label_dict = request.form['label_dict']
        
        ##Handle image upload
        image_file = request.files['image']
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(image_file.filename))
        image_file.save(image_path)
        
        ##Handle image dimensions
        input_width = int(request.form['input_width'])
        input_height = int(request.form['input_height'])
        
        ##Pass the data to the layer selection page
        return render_template('select_layers.html', model_path=model_path, image_path=image_path,
                               label_dict=label_dict, input_width=input_width, 
                               input_height=input_height, layers=layer_names)
    return render_template('upload.html')

@app.route('/predict')
def predict():
    model_path = request.args.get('model_path')
    image_path = request.args.get('image_path')
    try:
        CLASS_TO_LABEL = json.loads(request.args.get('label_dict'))
        LABEL_TO_CLASS = {value: key for key, value in CLASS_TO_LABEL.items()}
    except json.JSONDecodeError as e:
        return f"Error decoding label_dict JSON: {e}", 400
    #CLASS_TO_LABEL = json.loads(request.args.get('label_dict'))
    input_width = int(request.args.get('input_width'))
    input_height = int(request.args.get('input_height'))
    selected_layers = request.args.getlist('layer_names')
    model = load_model(model_path)
    original_img = cv2.imread(image_path)
    img = image.load_img(image_path, target_size=(input_width, input_height))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    preds = model.predict(x)
    predicted_class = np.argmax(preds[0])
    predicted_label = LABEL_TO_CLASS[int(predicted_class)]
    #print("Predicted label:", predicted_label)
    #predicted_label = CLASS_TO_LABEL[int(predicted_class)]
    
    gradcam_images = []

    for layer_name in selected_layers:
        last_conv_layer = model.get_layer(layer_name)
        grad_model = tf.keras.models.Model([model.input], [last_conv_layer.output, model.output])
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(x)
            loss = predictions[:, predicted_class]
        
        grads = tape.gradient(loss, conv_outputs)[0]
        weights = tf.reduce_mean(grads, axis=(0, 1))
        cam = np.sum(weights * conv_outputs[0], axis=-1)
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input_width, input_height))
        cam = cam / cam.max() if cam.max() != 0 else cam
    
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        original_img_resized = cv2.resize(image.img_to_array(img) / 255.0, (input_width, input_height))
        superimposed_img = cv2.addWeighted(original_img_resized.astype('float32'), 0.6,
                                           heatmap.astype('float32') / 255, 0.4, 0)
    
        gradcam_filename = f'gradcam_{layer_name}.png'
        cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], gradcam_filename), superimposed_img * 255)
        gradcam_images.append((gradcam_filename, layer_name))
        #gradcam_images.append(gradcam_filename)
        print("Grad-CAM images data:", gradcam_images)

    #return render_template('result.html', images=gradcam_images, predicted_label=predicted_label)
    return render_template('result.html', images=gradcam_images,predicted_label=predicted_label)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)