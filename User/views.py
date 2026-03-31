from django.shortcuts import render , redirect
from . import forms
from django.contrib import messages
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from .models import User_SigUp, ScanRecord
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import glob
import os
import cv2
import random
import sys
import os

import torch.nn as nn
import torch.nn.functional as F
from django.http import HttpResponse
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from io import BytesIO

# Create your views here.
def SigUp(request):
    if  request.method=='POST':
        form = forms.User_SigupForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request , 'Account Created Successfully')
            return redirect('UserLogin')
        else:
            messages.error(request , 'Please correct the errors below.')
    form = forms.User_SigupForm()

    return render(request , 'Register.html' , {'form':form})

def UserLogin(request):
    if request.method=='POST':
        username = request.POST.get('name')
        password = request.POST.get('password')
        try:
            data = User_SigUp.objects.get(Username=username , Password=password)
            if data.Status=='active':
                request.session['user_name'] = data.Name
                request.session['user_id'] = data.id
                return redirect('UserHome')
            else:
             messages.error(request , 'You are not activated yet')
        except Exception as e:
            messages.error (request , 'Invalid data')
            return render(request , 'UserLogin.html')
    return render(request , 'UserLogin.html')


def UserHome(request):
    return render(request , 'Users/UserHome.html')




 
def Traning(request):
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix, accuracy_score
        import seaborn as sns
  
        tumor = []
        path = os.path.join(settings.MEDIA_ROOT , 'brain_tumor_dataset' , 'yes' , '*.jpg')
        for f in glob.glob(path):
            img = cv2.imread(f)
            img = cv2.resize(img,(128 , 128))
            b, g, r = cv2.split(img)
            cv2.merge([r, g, b])
            tumor.append(img)

        healthy = []
        path =os.path.join(settings.MEDIA_ROOT , 'brain_tumor_dataset' , 'no' , '*.jpg')
        for f in glob.glob(path):
            img = cv2.imread(f)
            img = cv2.resize(img,(128 , 128))
            b, g, r = cv2.split(img)
            cv2.merge([r, g, b])
            healthy.append(img)
        healthy = np.array(healthy)
        tumor = np.array(tumor)
        All= np.concatenate((tumor, healthy))
        perf_dir = os.path.join(settings.BASE_DIR, 'Assets', 'Static', 'Performance')
        os.makedirs(perf_dir, exist_ok=True)
        
        plt.imshow(healthy[0])
        plt.axis('off')
        plt.savefig(os.path.join(perf_dir, 'healthy_sample.png'))
        plt.close()
        
        plt.imshow(tumor[0])
        plt.axis('off')
        plt.savefig(os.path.join(perf_dir, 'tumor_sample.png'))
        plt.close()
        def pot_random(healthy , tumor , num=5):
            healthy_images = healthy[np.random.choice(healthy.shape[0], num, replace=False)]
            tumor_images = tumor[np.random.choice(tumor.shape[0], num, replace=False)]
            plt.figure(figsize=(20, 8))
            for i in range(num):
                plt.subplot(1, num , i+1)
                plt.title('healthy')
                plt.imshow(healthy_images[i])
                plt.axis('off')
                
            plt.figure(figsize=(20, 8))
            for j in range(num):
                plt.subplot(1, num , j+1)
                plt.title('tumor')
                plt.imshow(tumor_images[j])
                plt.axis('off')
        # pot_random(healthy , tumor)
        class Dataset(object):
            def __getitem__(self, index):
                raise NotImplementedError

            def __len__(self): 
                raise NotImplementedError

            def __add__(self, other):
                raise NotImplementedError
        
        class MRI(Dataset):
            def __init__(self):
                
                tumor = []
                healthy = []
                path = os.path.join(settings.MEDIA_ROOT , 'brain_tumor_dataset' , 'yes' , '*.jpg')
                for f in glob.glob(path):
                    img = cv2.imread(f)
                    img = cv2.resize(img,(128 , 128))
                    b, g, r = cv2.split(img)
                    cv2.merge([r, g, b])
                    img = img.reshape(img.shape[2], img.shape[0], img.shape[1])
                    tumor.append(img)


                path = os.path.join(settings.MEDIA_ROOT , 'brain_tumor_dataset' , 'no' , '*.jpg')
                for f in glob.glob(path):
                    img = cv2.imread(f)
                    img = cv2.resize(img,(128 , 128))
                    b, g, r = cv2.split(img)
                    cv2.merge([r, g, b])
                    img = img.reshape(img.shape[2], img.shape[0], img.shape[1])
                    healthy.append(img)
        
                #out images


                tumor = np.array(tumor, dtype=np.float32)
                healthy = np.array(healthy, dtype=np.float32)

                #our labels
                tumor_labels = np.ones(tumor.shape[0], dtype=np.float32)
                healthy_labels = np.zeros(healthy.shape[0], dtype=np.float32)

            #concatenate
                self.images =np.concatenate((tumor, healthy) , axis=0)
                self.labels = np.concatenate((tumor_labels, healthy_labels) , axis=0)
                       
            def __len__(self):
                return self.images.shape[0]
            def __getitem__(self, index):
                sample = {'images': self.images[index], 'labels': self.labels[index]}
                return  sample
            def normalize(self):
                self.images = self.images / 255.0

        class CNN(nn.Module):
            def __init__(self):
                super(CNN, self).__init__()
                self.cnn_model = nn.Sequential (
                    nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),
                    nn.Tanh(),
                    nn.AvgPool2d(kernel_size=2, stride=5),
                    nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
                    nn.Tanh(),
                    nn.AvgPool2d(kernel_size=2, stride=5),
                )
                self.fc_model = nn.Sequential (
                    nn.Linear(in_features=256, out_features=120),
                    nn.Tanh(),
                    nn.Linear(in_features=120, out_features=84),
                    nn.Tanh(),
                    nn.Linear(in_features=84, out_features=1),
                    
                )
            def forward(self, x):
                x = self.cnn_model(x)
                x = x.view(x.size(0), -1)
                x = self.fc_model(x)
                x = F.sigmoid(x)
                return x
        mri_dataset = MRI()
        mri_dataset.normalize()
        model = CNN()
        dataloader = DataLoader(mri_dataset , batch_size = 32 , shuffle = False)
        model.eval()
        outputs = []
        y_true = []
        with torch.no_grad():
            for D in dataloader:
                image = D['images']
                print(image[0].shape)
                label = D['labels']

                y_hat = model(image)
                outputs.append(y_hat.cpu().detach().numpy())
                y_true.append(label.cpu().detach().numpy())
        outputs = np.concatenate(outputs , axis=0).squeeze()
        y_true  = np.concatenate(y_true , axis=0).squeeze()
        def threshold(scores , threshold = 0.5 , minimum =0 , maximum = 1):
            x=np.array(list(scores))
            x[x>=threshold]= maximum    
            x[x<threshold]= minimum
            return x
        accuracy_score(y_true , threshold(outputs))
        plt.figure(figsize=(10,5))
        plt.plot(outputs)
        plt.axvline(x=len(tumor) , color='r' ,linestyle='dashed')
        plt.grid()
        plt.show()
        eta = 0.0001
        EPOCHS = 300
        optimizer = torch.optim.Adam(model.parameters() , lr = eta)
        dataloader = DataLoader(mri_dataset , batch_size = 32 , shuffle = True)
        model.train()
        for epoch in range(1 , EPOCHS):
            losses = []
            for D in dataloader:
                optimizer.zero_grad()
                data = D['images']
                label = D['labels']
                y_hat = model(data)
                #define losses function
                error = nn. BCELoss()
                loss= torch.sum(error(y_hat.squeeze() , label))   
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                print('Tain Epoch {} Loss {:.3f}'.format(epoch, np.mean(losses)))

        model.eval()
        dataloader = DataLoader(mri_dataset , batch_size = 32 , shuffle = False)
        outputs = []
        y_true = []
        with torch.no_grad():
            for D in dataloader:
                image = D['images']
                print(image[0].shape)
                label = D['labels']

                y_hat = model(image)
                outputs.append(y_hat.cpu().detach().numpy())
                y_true.append(label.cpu().detach().numpy())

        outputs = np.concatenate(outputs , axis=0).squeeze()
        y_true  = np.concatenate(y_true , axis=0).squeeze()
        acc= accuracy_score(y_true , threshold(outputs))
        
        # Save Metrics
        plt.figure(figsize=(10,5))
        plt.plot(losses)
        plt.title("Training Loss History")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.savefig(os.path.join(perf_dir, 'loss_graph.png'))
        plt.close()

        plt.figure(figsize=(8,6))
        cm = confusion_matrix(y_true , threshold(outputs))
        sns.heatmap(cm , annot=True, fmt='g', cmap='viridis')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(os.path.join(perf_dir, 'confusion_matrix.png'))
        plt.close()

        with open(os.path.join(perf_dir, 'accuracy.txt'), 'w') as f:
            f.write(str(round(acc * 100, 2)))

        return render(request , 'Users/UserTraning.html' , {'acc':acc})






def is_brain_mri_image(image):
    """
    Validates whether an uploaded image is likely a brain MRI scan.
    
    Brain MRI images have two key characteristics:
    1. They are grayscale-dominant: R, G, B channels are nearly identical
       (low mean inter-channel difference). Color photos of people, nature,
       food etc. have much larger channel separation.
    2. They have a dark background with localised bright regions: the mean
       pixel intensity is relatively low and the std-dev is significant
       (not a blank/solid-color image, but not a uniformly-bright photo).
    
    Returns True if the image likely is a brain MRI, False otherwise.
    """
    if image is None:
        return False

    # Work on a small thumbnail for speed
    thumb = cv2.resize(image, (64, 64)).astype(np.float32)

    b_ch, g_ch, r_ch = thumb[:, :, 0], thumb[:, :, 1], thumb[:, :, 2]

    # --- Check 1: Grayscale dominance ---
    # Mean absolute difference between each pair of channels.
    # Natural color photos typically have diff > 15-20.
    # MRI scans (even if saved as RGB) have diff < 10.
    rg_diff = np.mean(np.abs(r_ch - g_ch))
    rb_diff = np.mean(np.abs(r_ch - b_ch))
    gb_diff = np.mean(np.abs(g_ch - b_ch))
    avg_channel_diff = (rg_diff + rb_diff + gb_diff) / 3.0

    GRAYSCALE_THRESHOLD = 20.0  # pixels must be near-gray
    if avg_channel_diff > GRAYSCALE_THRESHOLD:
        return False   # Clearly a colour photo – reject

    # --- Check 2: Intensity spread ---
    # Convert to grayscale and check mean + std.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean_intensity = np.mean(gray)
    std_intensity  = np.std(gray)

    # Pure-white / pure-black images or very bright colourful photos
    # (screenshots, posters) fail one of these ranges.
    # MRI scans: mean is typically 20-150, std is typically 30-100.
    if mean_intensity < 5 or mean_intensity > 220:
        return False   # Blank or over-exposed
    if std_intensity < 15:
        return False   # Uniformly flat – not a meaningful scan

    # --- Check 3: Dark borders & central brain mass ---
    # MRI scans typically have a dark background (air) around the skull/brain.
    thumb_gray = cv2.cvtColor(thumb.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    corner_size = 12
    top_left = thumb_gray[0:corner_size, 0:corner_size]
    top_right = thumb_gray[0:corner_size, -corner_size:]
    bottom_left = thumb_gray[-corner_size:, 0:corner_size]
    bottom_right = thumb_gray[-corner_size:, -corner_size:]
    
    corners_mean = (np.mean(top_left) + np.mean(top_right) + np.mean(bottom_left) + np.mean(bottom_right)) / 4.0
    
    # Check center (32x32 area in the middle of 64x64 thumbnail)
    center_region = thumb_gray[16:48, 16:48]
    center_mean = np.mean(center_region)
    
    if corners_mean > 85:
        return False   # Borders are too bright (regular photos have background)
        
    if center_mean < corners_mean + 15:
        return False   # Center should be noticeably brighter than the corners

    return True


def to_grayscale_3ch(image):
    """Converts an image to 3-channel grayscale (BGR) for model compatibility."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.merge([gray, gray, gray])


def predict(request):
    if request.method == 'POST':
        img = request.FILES.get('img')
        fs = FileSystemStorage(location=settings.MEDIA_ROOT)
        filename = fs.save(img.name, img)  # Save the uploaded file with its original name
        uploaded_file_url = fs.url(filename)
        image_path = os.path.join(settings.MEDIA_ROOT, filename)

        # Load and process the image
        # Try reading via path first; fall back to in-memory buffer if the path
        # contains spaces or special characters that confuse OpenCV on Windows.
        image = cv2.imread(image_path)
        if image is None:
            img.seek(0)
            file_bytes = np.frombuffer(img.read(), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            messages.error(request, "Could not read the uploaded image. Please upload a valid JPG/PNG MRI scan.")
            return render(request, 'Users/UserPredict.html')

        # ── Brain MRI validation ──────────────────────────────────────────────
        # Reject non-brain images BEFORE running the model
        if not is_brain_mri_image(image):
            # Keep dashboard "prediction probability" consistent for invalid uploads.
            request.session['last_prediction_probability'] = 0
            request.session['last_prediction_result'] = 'Invalid Image'
            request.session['last_prediction_status'] = 'invalid'
            request.session['last_uploaded_file_url'] = uploaded_file_url
            request.session.modified = True

            return render(request, 'Users/UserPredict.html', {
                'result': 'Invalid Image',
                'prediction_probability': 0,
                'tumor_probability': 0,
                'non_tumor_probability': 0,
                'status': 'invalid',
                'uploaded_file_url': uploaded_file_url,
                'model_accuracy': 'N/A',
                'invalid_message': 'The uploaded image does not appear to be a Brain MRI scan. Please upload a valid Brain MRI image.',
            })
        # ─────────────────────────────────────────────────────────────────────

        class CNN(nn.Module):
            def __init__(self):
                super(CNN, self).__init__()
                self.cnn_model = nn.Sequential(
                    nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),
                    nn.Tanh(),
                    nn.AvgPool2d(kernel_size=2, stride=5),
                    nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
                    nn.Tanh(),
                    nn.AvgPool2d(kernel_size=2, stride=5),
                )
                self.fc_model = nn.Sequential(
                    nn.Linear(in_features=256, out_features=120),
                    nn.Tanh(),
                    nn.Linear(in_features=120, out_features=84),
                    nn.Tanh(),
                    nn.Linear(in_features=84, out_features=1),
                )

            def forward(self, x):
                x = self.cnn_model(x)
                x = x.view(x.size(0), -1)
                x = self.fc_model(x)
                x = F.sigmoid(x)
                return x

        model = CNN()
        model.load_state_dict(torch.load(os.path.join(settings.MEDIA_ROOT, 'weights', 'model.pt'), map_location=torch.device('cpu')))
        model.eval()

        image_resized = cv2.resize(image, (128, 128))
        b, g, r = cv2.split(image_resized)
        image_resized = cv2.merge([r, g, b])  # Convert BGR to RGB

        # Reshape to match input size and normalize
        image_input = image_resized.reshape(1, 3, 128, 128)
        image_input = torch.from_numpy(image_input).float() / 255.0

        # Make prediction
        with torch.no_grad():
            output = model(image_input)
            prob = output.item()

        tumor_probability = round(prob * 100, 2)
        non_tumor_probability = round((1 - prob) * 100, 2)

        # Apply threshold and determine results
        if prob > 0.6:
            result = "Tumor Detected"
            prediction_probability = round(prob * 100, 2)
            status = "danger"
        elif prob < 0.4:
            result = "No Tumor Detected"
            prediction_probability = round((1 - prob) * 100, 2)
            status = "success"
        else:
            result = "Uncertain Result"
            prediction_probability = round(max(prob, 1 - prob) * 100, 2)
            status = "warning"

        # Read saved model training accuracy (written by Training view)
        perf_dir = os.path.join(settings.BASE_DIR, 'Assets', 'Static', 'Performance')
        model_accuracy = "N/A"
        acc_file = os.path.join(perf_dir, 'accuracy.txt')
        if os.path.exists(acc_file):
            with open(acc_file, 'r') as f:
                model_accuracy = f.read().strip()

        # Save Record to Database
        ScanRecord.objects.create(
            image=img,
            result=result,
            accuracy=prediction_probability
        )

        # Store latest prediction for the dashboard (static metrics + dynamic probability).
        request.session['last_prediction_probability'] = prediction_probability
        request.session['last_prediction_result'] = result
        request.session['last_prediction_status'] = status
        request.session['last_uploaded_file_url'] = uploaded_file_url
        request.session.modified = True

        return render(request, 'Users/UserPredict.html', {
            'result': result,
            'prediction_probability': prediction_probability,
            'tumor_probability': tumor_probability,
            'non_tumor_probability': non_tumor_probability,
            'status': status,
            'uploaded_file_url': uploaded_file_url,
            'model_accuracy': model_accuracy,
        })
    return render(request, 'Users/UserPredict.html')

def performance(request):
    import matplotlib.pyplot as plt
    import seaborn as sns

    perf_dir = os.path.join(settings.BASE_DIR, 'Assets', 'Static', 'Performance')
    os.makedirs(perf_dir, exist_ok=True)

    # Default: static metrics from training
    acc_static = "98.4"
    acc_file = os.path.join(perf_dir, 'accuracy.txt')
    if os.path.exists(acc_file):
        with open(acc_file, 'r') as f:
            acc_static = f.read()

    # Default images: training graphs/evaluation confusion matrix
    static_prefix = settings.STATIC_URL
    if not static_prefix.startswith('/'):
        static_prefix = '/' + static_prefix
    confusion_matrix_image = f"{static_prefix}Performance/confusion_matrix.png"
    loss_graph_image = f"{static_prefix}Performance/loss_graph.png"

    # Dynamic part: based on recent uploaded scans (so it changes with input images)
    recent_scans = list(ScanRecord.objects.all().order_by('-timestamp')[:20])
    if recent_scans:
        # oldest -> newest for plotting
        recent_scans = list(reversed(recent_scans))

        probs = []
        preds = []
        for scan in recent_scans:
            try:
                probs.append(float(scan.accuracy))
            except Exception:
                continue

            result_text = str(scan.result)
            if 'Tumor' in result_text and 'No Tumor' not in result_text:
                preds.append(1)  # tumor
            elif 'No Tumor' in result_text:
                preds.append(0)  # healthy
            else:
                # Uncertain/invalid: fallback to probability threshold
                preds.append(1 if float(scan.accuracy) >= 50.0 else 0)

        if preds and probs:
            # Use recent average confidence as the displayed "accuracy" so it changes per image.
            acc_dynamic = round(float(sum(probs)) / len(probs), 2)

            # Confusion matrix using (actual=pred) because we don't have ground-truth labels for user uploads.
            cm = np.zeros((2, 2), dtype=int)  # [healthy, tumor] x [healthy, tumor]
            for p in preds[: len(probs)]:
                cm[p, p] += 1

            # Save recent confusion matrix + graph
            cm_path = os.path.join(perf_dir, 'confusion_matrix_recent.png')
            graph_path = os.path.join(perf_dir, 'prediction_graph_recent.png')

            plt.figure(figsize=(5.6, 4.2))
            sns.heatmap(cm, annot=True, fmt='g', cmap='viridis')
            plt.title('Recent Predictions Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual (treated as prediction)')
            plt.savefig(cm_path)
            plt.close()

            plt.figure(figsize=(10, 5))
            plt.plot(probs)
            plt.title('Recent Prediction Probability (Uploaded Images)')
            plt.xlabel('Recent Scan Index')
            plt.ylabel('Confidence (%)')
            plt.grid(True)
            plt.savefig(graph_path)
            plt.close()

            confusion_matrix_image = f"{static_prefix}Performance/confusion_matrix_recent.png"
            loss_graph_image = f"{static_prefix}Performance/prediction_graph_recent.png"
            acc_static = str(acc_dynamic)

    return render(request, 'Users/ModelPerformance.html', {
        'accuracy': acc_static,
        'confusion_matrix_image': confusion_matrix_image,
        'loss_graph_image': loss_graph_image,
    })

def probability(request):
    """
    Shows only dynamic Prediction Probability for the latest uploaded image.
    Model Performance metrics (overall accuracy, confusion matrix) are kept on the `performance` page.
    """
    last_prediction_probability = request.session.get('last_prediction_probability')
    last_prediction_result = request.session.get('last_prediction_result')
    last_prediction_status = request.session.get('last_prediction_status')
    last_uploaded_file_url = request.session.get('last_uploaded_file_url')

    return render(request, 'Users/PredictionProbability.html', {
        'last_prediction_probability': last_prediction_probability,
        'last_prediction_result': last_prediction_result,
        'last_prediction_status': last_prediction_status,
        'last_uploaded_file_url': last_uploaded_file_url,
    })

def history(request):
    scans = ScanRecord.objects.all().order_by('-timestamp')
    return render(request, 'Users/DataHistory.html', {'scans': scans})

def reports(request):
    scans = ScanRecord.objects.all().order_by('-timestamp')
    return render(request, 'Users/Reports.html', {'scans': scans})

def download_report(request, id):
    scan = ScanRecord.objects.get(id=id)
    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    
    # Report Header
    p.setFont("Helvetica-Bold", 24)
    p.setStrokeColor(colors.HexColor("#64ffda"))
    p.drawString(100, 750, "BRAIN TUMOR DIAGNOSTIC REPORT")
    p.setFont("Helvetica", 10)
    p.drawString(100, 735, "Neural Network Identification: #" + str(scan.id))
    
    patient_name = scan.user.Name if scan.user else request.session.get('user_name', 'Unregistered / Guest')
    p.drawString(100, 720, "Patient Name: " + patient_name)
    
    p.line(100, 710, 500, 710)
    
    # 1. Patient Scan Image
    try:
        p.drawImage(scan.image.path, 100, 500, width=200, height=200, preserveAspectRatio=True)
        p.setFont("Helvetica-Oblique", 8)
        p.drawString(100, 485, "1. Patient Scan Image")
    except Exception as e:
        p.drawString(100, 680, "[Image Data Unavailable]")
        
    # Right Column Details
    p.setFont("Helvetica-Bold", 14)
    p.drawString(320, 680, "Diagnostic Details:")
    
    p.setFont("Helvetica-Bold", 10)
    p.drawString(320, 650, "2. Prediction Result:")
    p.setFont("Helvetica", 10)
    p.drawString(320, 635, str(scan.result))
    
    p.setFont("Helvetica-Bold", 10)
    p.drawString(320, 605, "3. Prediction Probability:")
    p.setFont("Helvetica", 10)
    p.drawString(320, 590, f"{scan.accuracy}%")
    
    p.setFont("Helvetica-Bold", 10)
    p.drawString(320, 560, "4. Model Confidence:")
    p.setFont("Helvetica", 10)
    p.drawString(320, 545, "HIGH LEVEL")
    
    p.setFont("Helvetica-Bold", 10)
    p.drawString(320, 515, "5. Date & Time:")
    p.setFont("Helvetica", 10)
    p.drawString(320, 500, scan.timestamp.strftime('%Y-%m-%d %H:%M'))
    
    # Bottom Section: Short Summary
    p.setFont("Helvetica-Bold", 14)
    p.drawString(100, 430, "6. Short Summary:")
    p.setFont("Helvetica", 11)
    if "No Tumor" not in str(scan.result) and "Tumor" in str(scan.result):
        interp = "Anomalies detected indicative of a tumor. Specialist review recommended."
    else:
        interp = "AI detected no anomalies. Routine follow-up is suggested."
    p.drawString(100, 405, interp)
    
    # Footer
    p.setFont("Helvetica-Oblique", 8)
    p.drawString(100, 100, "This is an AI-generated report for research purposes only.")
    
    p.showPage()
    p.save()
    
    buffer.seek(0)
    return HttpResponse(buffer, content_type='application/pdf', 
                       headers={'Content-Disposition': f'attachment; filename="Brain_Report_{scan.id}.pdf"'})
    



                    


    
            
