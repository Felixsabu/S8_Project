from flask import Flask, render_template, request, redirect, url_for, jsonify
import numpy as np
import pandas as pd
import os
import pickle
import cv2
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.models import load_model
from ultralytics import YOLO
from PIL import Image
from werkzeug.utils import secure_filename
from tensorflow.keras.saving import register_keras_serializable

app = Flask(__name__)

# Set the path for the current directory
# web = os.path.dirname(os.path.abspath(__file__))

# Load the pre-trained models with exception handling
# try:
#     with open('Model_SVM_C2.pkl', 'rb') as pickle_file:
#         new_data = pickle.load(pickle_file)

#     dental_model_path = os.path.join(web, "/content/drive/MyDrive/colab_notebook/FinalApp/dental_model.h5")
#     severity_model_path = os.path.join(web, "/content/drive/MyDrive/colab_notebook/FinalApp/tooth_cavity_detection_model.h5")

#     # Register custom function for feature extraction
#     @register_keras_serializable()
#     def feature_extractor_model():
#         return hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/3", 
#                               input_shape=(224, 224, 3), trainable=False)

#     # Load models with custom objects
#     dental_model = tf.keras.models.load_model(dental_model_path, 
#                                               custom_objects={'feature_extractor_model': feature_extractor_model})
#     severity_model = load_model(severity_model_path)

#     # # Load YOLO model
#     # yolo_model = YOLO("best.pt")

#     if dental_model:
#       print("dental model")
#     else:
#       print("no dental")
#     if severity_model:
#       print("severity model")
#     else:
#       print("no severity")
#     if yolo_model:
#       print("yolo model")
#     else:
#       print("no yolo")

# except Exception as e:
#     print(f"Error loading models: {e}")
#     dental_model, severity_model, yolo_model = None, None, None


# Routes
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("appindex.html")

@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
      web = os.path.dirname(os.path.abspath(__file__))
      with open('Model_SVM_C2.pkl', 'rb') as pickle_file:
        new_data = pickle.load(pickle_file)

      dental_model_path = os.path.join(web, "/content/drive/MyDrive/colab_notebook/FinalApp/dental_model.h5")
      severity_model_path = os.path.join(web, "/content/drive/MyDrive/colab_notebook/FinalApp/tooth_cavity_detection_model.h5")

      # Register custom function for feature extraction
      @register_keras_serializable()
      def feature_extractor_model():
          return hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/3", 
                              input_shape=(224, 224, 3), trainable=False)

      # Load models with custom objects


      
      # dental_model = tf.keras.models.load_model(dental_model_path, 
      #                                         custom_objects={'feature_extractor_model': feature_extractor_model})
      severity_model = load_model(severity_model_path)

    # # Load YOLO model
    # yolo_model = YOLO("best.pt")

      # if dental_model:
      #   print("dental model")
      # else:
      #   print("no dental")
      # if severity_model:
      #   print("severity model")
      # else:
      #   print("no severity")
    
      if 'my_image' not in request.files:
          return "No file uploaded", 400

      imagefile = request.files['my_image']
      if imagefile.filename == '':
          return "No selected file", 400

      # Secure the filename and save it
      filename = secure_filename(imagefile.filename)
      image_path = os.path.join("static", filename)
      imagefile.save(image_path)

      # Predict dental condition
      # cavity = predict_dental_condition(image_path)
      severity = predict_severity(image_path, severity_model) #if cavity == "has cavity" else cavity

      # Process image for color analysis
      img = cv2.imread(image_path)
      height, width, _ = img.shape

      start_x = int(0.8 * width)
      end_x = min(start_x + int(0.1 * width), width)
      start_y = int(0.8 * height)
      end_y = min(start_y + int(0.1 * height), height)

      if end_x <= width and end_y <= height:
          S5RGB_imgA1A1_1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
          S5meanrefA1_1 = S5RGB_imgA1A1_1[start_y:end_y, start_x:end_x]
          S5meanA1_2 = S5RGB_imgA1A1_1[start_y:end_y, start_x + int(0.15 * width):start_x + int(0.25 * width)]

          S5refA1_1 = np.reshape(S5meanrefA1_1, (-1, 3))
          S5refA1_2 = np.reshape(S5meanA1_2, (-1, 3))

          Rref, Gref, Bref = S5refA1_1.mean(axis=0)
          R, G, B = S5refA1_2.mean(axis=0)

          list_total2 = [[Rref, Gref, Bref, R, G, B]]
          predic_model = new_data.predict(list_total2)
          model = predic_model[0]

          out_model = pd.read_csv('OUTPUT.csv')
          test = out_model[out_model['ชื่อเฉดไกด์'] == model]

          return render_template(
               "appindex.html",
              test1=str(test['ชื่อเฉดไกด์'].iloc[0]),
              test2=severity,
              test3=str(test['โทนสี'].iloc[0]),
              test4=str(test['ชื่อเฉดไกด์ที่ใกล้เคียง'].iloc[0]),
              test5=str(test['เทียบเท่าเฉดไกด์ 3D Master'].iloc[0]),
              image_target=image_path
          )
    else:
        return "Image size is too small for the required processing.", 400

@app.route("/detect", methods=["POST"])
def detect():
    if 'image_file' not in request.files:
        return "No file uploaded", 400

    buf = request.files["image_file"]
    boxes = detect_objects_on_image(buf.stream)
    return jsonify(boxes)

def detect_objects_on_image(buf):
  # Load YOLO model
    yolo_model = YOLO("best.pt")
    
    results = yolo_model.predict(Image.open(buf))
    result = results[0]
    output = []
    for box in result.boxes:
        x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]
        class_id = box.cls[0].item()
        prob = round(box.conf[0].item(), 2)
        prob_percentage = f"{prob * 100:.2f}%"
        if result.names[class_id] in ["Caries", "Deep Caries"]:
            output.append([x1, y1, x2, y2, result.names[class_id], prob_percentage])
    return output

def predict_dental_condition(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = dental_model.predict(img_array)
    predicted_class = np.argmax(predictions)

    return "has cavity" if predicted_class == 0 else "no cavity"

def predict_severity(image_path, model):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128)) / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    severity = np.argmax(prediction, axis=1)
    severity_map = {0: 'healthy', 1: 'mild', 2: 'moderate', 3: 'severe'}
    return severity_map[severity[0]]

@app.route('/display/<image_target>')
def display_image(image_target):
    return redirect(url_for('static', filename=image_target), code=301)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
