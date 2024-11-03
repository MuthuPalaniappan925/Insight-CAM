import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from flask import Flask, request, render_template, send_from_directory

app = Flask(__name__)

uploads_dir = 'uploads'
if not os.path.exists(uploads_dir):
    os.makedirs(uploads_dir)

model = VGG16(weights='imagenet')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file uploaded', 400
    
    file = request.files['file']
    layer_names = request.form.getlist('layers')
    img_path = os.path.join(uploads_dir, file.filename)
    file.save(img_path)
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    

    preds = model.predict(x)
    predicted_class = np.argmax(preds[0])
    
    gradcam_images = []
    
    for layer_name in layer_names:
        last_conv_layer = model.get_layer(layer_name)
        grad_model = tf.keras.models.Model([model.input], [last_conv_layer.output, model.output])
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(x)
            loss = predictions[:, predicted_class]
        
        grads = tape.gradient(loss, conv_outputs)[0]
        
        weights = tf.reduce_mean(grads, axis=(0, 1))
        cam = np.sum(weights * conv_outputs[0], axis=-1)
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam / cam.max() if cam.max() != 0 else cam
    
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        original_img = image.img_to_array(img) / 255.0  
        superimposed_img = cv2.addWeighted(original_img.astype('float32'), 0.6,
                                            heatmap.astype('float32') / 255, 0.4, 0)
    
        gradcam_filename = f'gradcam_{layer_name}.png'
        cv2.imwrite(os.path.join(uploads_dir, gradcam_filename), superimposed_img * 255)
        gradcam_images.append(gradcam_filename)

    return render_template('results.html', images=gradcam_images)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(uploads_dir, filename)

if __name__ == '__main__':
    app.run(debug=True)