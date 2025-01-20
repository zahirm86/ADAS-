from flask import Flask, render_template, request, redirect, url_for, flash , send_from_directory
from werkzeug.utils import secure_filename
import cv2
from ultralytics import YOLO
import numpy as np
import os
import torch
import tensorflow as tf
import time 
# Check if GPU is available, if not, use CPU
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





# Load pretrained YOLOv8n models



app = Flask(__name__)



app.secret_key = os.urandom(24)
app.debug = True  # Enable debug mode
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi'}



os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use only GPU 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


modelA = YOLO('models/Pedestrians-Vehicles.pt').to(device)
modelB = YOLO('models/Traffic_Signs.pt').to(device)




def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')




@app.route('/static/images/<path:filename>')
def serve_image(filename):
    return send_from_directory('static/images', filename)







'''
# Route to serve  CSS
@app.route('/static/css/bootstrap.css')
def serve_bootstrap_css():
    return send_from_directory('static/css', 'bootstrap.css')

# Route to serve custom SCSS
@app.route('/static/css/style.scss')
def serve_custom_scss():
    return send_from_directory('static/css', 'style.scss')

# Route to serve responsive CSS
@app.route('/static/css/responsive.css')
def serve_responsive_css():
    return send_from_directory('static/css', 'responsive.css')

# Route to serve Font Awesome CSS
@app.route('/static/css/font-awesome.min.css')
def serve_font_awesome_css():
    return send_from_directory('static/css', 'font-awesome.min.css')



# Route to serve Bootstrap.js
@app.route('/static/js/bootstrap.js')
def serve_bootstrap_js():
    return send_from_directory('static/js', 'bootstrap.js')

# Route to serve custom.js
@app.route('/static/js/custom.js')
def serve_custom_js():
    return send_from_directory('static/js', 'custom.js')

# Route to serve jQuery
@app.route('/static/js/jquery-3.4.1.min.js')
def serve_jquery():
    return send_from_directory('static/js', 'jquery-3.4.1.min.js')
'''


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Handle file upload
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            process_video(filepath)
            return redirect(url_for('upload_file'))  # Redirect to the upload page after processing
    else:
        # Handle GET request (e.g., render upload form)
        return render_template('index.html')

    
    
    
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    timestamp = int(time.time())
    output_filename = f'processed_video_{timestamp}.mp4'
    output_path = os.path.join('uploads', output_filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
 
    
   
    trapezium_bottom_width = width
    trapezium_top_width = int(width / 2)
    trapezium_height = int(height / 4)
    trapezium_bottom_left = (0, height)
    trapezium_bottom_right = (trapezium_bottom_width, height)
    trapezium_top_left = (int((width - trapezium_top_width) / 2), height - trapezium_height)
    trapezium_top_right = (int((width + trapezium_top_width) / 2), height - trapezium_height)
    trapezium_pts = np.array([trapezium_bottom_left, trapezium_bottom_right, trapezium_top_right, trapezium_top_left], np.int32)
    trapezium_pts = trapezium_pts.reshape((-1, 1, 2))
    
    
    
    
    

    try:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            frame_A = frame.copy()
            frame_B = frame.copy()

            vehicle_count = 0
            pedestrian_count = 0

            resultsA = modelA.predict(frame_A, classes=[0,1,2,3,5,6,7], conf=0.5, verbose=False)
            proximity_alert = False

            for result in resultsA:
                boxes = result.boxes.xyxy
                for box, _, cls in zip(boxes, result.boxes.conf, result.boxes.cls):
                    b = box.cpu().numpy().astype(int)
                    c = int(cls.cpu().numpy())
                    if c in [1, 2, 3, 5, 6, 7]:
                        vehicle_count += 1
                    elif c == 0:
                        pedestrian_count += 1
                    cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
                    cv2.putText(frame, modelA.names[c], (b[0], b[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    if (b[3] >= trapezium_top_left[1] and b[3] >= trapezium_top_right[1] and
                        trapezium_top_left[0] <= (b[0] + b[2]) / 2 <= trapezium_top_right[0]):
                        proximity_alert = True

            cv2.polylines(frame, [trapezium_pts], isClosed=True, color=(255, 0, 0), thickness=2)

            cv2.putText(frame, f"Total Vehicles: {vehicle_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
           # Original text with border
            #cv2.putText(frame, f"Total Vehicles: {vehicle_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Border text with bold effect (thicker font weight)
            #cv2.putText(frame, f"Total Vehicles: {vehicle_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2+4)

            cv2.putText(frame, f"Total Pedestrians: {pedestrian_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
           # cv2.putText(frame, f"Total Pedestrians: {pedestrian_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2+4)

            if proximity_alert:
                cv2.putText(frame, "PROXIMITY ALERT", (int(width / 2) - 100, int(height / 2) - 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            resultsB = modelB.predict(frame_B, conf=0.3, verbose=False)
            for result in resultsB:
                boxes = result.boxes.xyxy
                for box, _, cls in zip(boxes, result.boxes.conf, result.boxes.cls):
                    b = box.cpu().numpy().astype(int)
                    x1, y1, x2, y2 = box.cpu().numpy().astype(int)
                    c = int(cls.cpu().numpy())
                    x_center = (x1 + x2) / 2  # Define x_center here
                    cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)

                    img_h, img_w, _ = frame.shape
                    y_center = y2
                    img_center_x = img_w / 2
                    img_center_y = img_h
                    squared_diff_x = (img_center_x - x_center) ** 2
                    squared_diff_y = (img_center_y - y_center) ** 2
                    distance = (squared_diff_x + squared_diff_y) ** 0.5
                    area = (x2 - x1) * (y2 - y1)
                    closeness_estimate = distance / area  # Relative measure of distance based on area and distance

                    cv2.putText(frame, f"{modelB.names[c]} Closeness Estimate: {closeness_estimate:.2f}", (b[0], b[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            out.write(frame)  # Write the processed frame to the output video
            
        flash('Video processed successfully! Click the link below to download.')
    except Exception as e:
        print(f"An error occurred: {e}")
        flash(f'An error occurred while processing the video: {e}')


    finally:
        # Release video capture and writer
        cap.release()
        out.release()
       
if __name__ == '__main__':
    app.run()
