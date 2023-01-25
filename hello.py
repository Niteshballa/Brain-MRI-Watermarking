from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
from PIL import Image 
import os
import hashlib
import random
from imwatermark import WatermarkDecoder,WatermarkEncoder
import cv2 as cv
from skimage import io, color
from skimage.io import imread, imshow
from skimage.filters import threshold_otsu
from skimage.metrics import mean_squared_error as MSE, peak_signal_noise_ratio as PSNR, structural_similarity as SSIM
import matplotlib.pyplot as plt
import pywt
import numpy as np
import math
from PIL import Image
from scipy.fftpack import dct
from scipy.fftpack import idct
from math import floor, ceil
import numpy as np
import random

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['OUTPUT_FOLDER'] = 'static/outputs'

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            img = cv.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            dim=(512,512)
            img = cv.resize(img, dim)

            # Process image and generate output images
            roi = fetchRoi(img)
            nroi = genNroi(img,roi)
            ROI_watermark_text = hashlib.sha256(roi).hexdigest()
            print(ROI_watermark_text)
            roi_img = genRoi(img,roi)
            hashed_roi = watermark (ROI_watermark_text, roi_img)
            encoded_img = watermark (ROI_watermark_text, img)
            NROI_gray = color.rgb2gray(nroi)
            img_gray = color.rgb2gray(img)
            encoded_img_gray = color.rgb2gray(encoded_img)
            model = 'haar'
            level = 1
            image_array, mainImg = convert_image(os.path.join(app.config['UPLOAD_FOLDER'], filename), 2048)
            s,d = iwt(hashed_roi.flatten())
            s = np.reshape(s,(512,256,3))
            cv.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], 'iwtRoi.jpg'),s)
            watermark_array, mark = convert_image(os.path.join(app.config['OUTPUT_FOLDER'], 'hashed_roi.jpg'), 128)
            coeffs_image = process_coefficients(image_array, model, level=level)
            dct_array = apply_dct(coeffs_image[0])
            dct_array = embed_watermark(watermark_array, dct_array)
            coeffs_image[0] = inverse_dct(dct_array)
            # reconstruction
            image_array_H=pywt.waverec2(coeffs_image, model)
            watermarked_img = print_image_from_array(image_array_H, 'image_with_watermark.jpg')
            # recover images
            recovered_img = recover_watermark(image_array = image_array_H, model=model, level = level)    
            recovered_img.save(os.path.join(app.config['OUTPUT_FOLDER'], 'recovered_ROI.jpg'))
            recovered_iwt = Image.open(os.path.join(app.config['OUTPUT_FOLDER'], 'recovered_ROI.jpg'))
            recovered_iwt = recovered_iwt.resize((256,512))
            iwt_recovered_array = np.asarray(recovered_iwt)
            res = iiwt(iwt_recovered_array.flatten(),d)
            res = np.reshape(res,(512,512))
            extracted_hash = decodeText(ROI_watermark_text,hashed_roi)
            # output_img_4 = process_image_4(img)
            # output_img_5 = process_image_5(img)
            # output_img_6 = process_image_6(img)

            # Save output images
            cv.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], 'uploaded.jpg'),img)
            cv.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], 'roi.jpg'), roi_img)
            cv.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], 'hashed_roi.jpg'), hashed_roi)
            cv.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], 'nroi.jpg'), nroi)
            cv.imwrite(os.path.join(app.config['OUTPUT_FOLDER'], 'watermarkHashed.jpg'), encoded_img)
            plt.imsave(os.path.join(app.config['OUTPUT_FOLDER'], 'iwtWatermarked.jpg'),watermarked_img, cmap = "gray")
            plt.imsave(os.path.join(app.config['OUTPUT_FOLDER'], 'iwtRecovered.jpg'),res, cmap="gray")
            
            
             # URLs for output images
            img = url_for('static',filename= 'outputs/uploaded.jpg')
            roi = url_for('static', filename='outputs/roi.jpg')
            hashed_roi = url_for('static', filename = 'outputs/hashed_roi.jpg')
            hashed_text= ROI_watermark_text
            nroi = url_for('static', filename='outputs/nroi.jpg')
            encoded_img = url_for('static', filename='outputs/watermarkHashed.jpg')
            iwtRoiUrl = url_for('static', filename='outputs/iwtRoi.jpg')
            iwtWatermarked = url_for('static', filename='outputs/iwtWatermarked.jpg')
            iwtRecovered = url_for('static', filename='outputs/iwtRecovered.jpg')

            return render_template('index.html', output_url_1=img, output_url_2=roi , output_url_3=nroi, output_text_hashed = hashed_text, output_url_4 = hashed_roi, output_url_5 = iwtWatermarked, output_url_6 = iwtRecovered, extracted_hash = extracted_hash)

    return render_template('index.html')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif'}

def fetchRoi(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY, 0.7)
    (T, thresh) = cv.threshold(gray, 155, 255, cv.THRESH_BINARY)
    (T, threshInv) = cv.threshold(gray, 155, 255,cv.THRESH_BINARY_INV)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (10, 5))
    closed = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
    ROI = cv.erode(closed, None, iterations = 14) 
    ROI = cv.dilate(ROI, None, iterations = 13)
    return ROI

def genNroi(img,roi):
    NROI = img.copy()
    NROI[roi.squeeze()!=0] = 0
    return NROI

def genRoi(img,roi):
    NROI = img.copy()
    NROI[roi.squeeze()==0] = 0
    return NROI

def watermark(ROI_watermark_text, img):
    encoder = WatermarkEncoder()
    encoder.set_watermark('bytes', ROI_watermark_text.encode('utf-8'))
    bgr_encoded = encoder.encode(img, 'dwtDct')
    return bgr_encoded

def decodeText(ROI_watermark_text,bgr_encoded):
    decoder = WatermarkDecoder('bytes',len(ROI_watermark_text)*8)
    watermark = decoder.decode(bgr_encoded, 'dwtDct')
    return watermark.decode('utf-8')

def convert_image(image_name, size):
    img = Image.open(image_name).resize((size, size), 1)
    img = img.convert('L')
    image_array = np.array(img.getdata(), dtype=float).reshape((size, size))
    return (image_array, img)

def process_coefficients(imArray, model, level):
    coeffs=pywt.wavedec2(data = imArray, wavelet = model, level = level)
    # print coeffs[0].__len__()
    coeffs_H=list(coeffs) 
   
    return coeffs_H

def embed_mod2(coeff_image, coeff_watermark, offset=0):
    for i in xrange(coeff_watermark.__len__()):
        for j in xrange(coeff_watermark[i].__len__()):
            coeff_image[i*2+offset][j*2+offset] = coeff_watermark[i][j]

    return coeff_image

def embed_mod4(coeff_image, coeff_watermark):
    for i in xrange(coeff_watermark.__len__()):
        for j in xrange(coeff_watermark[i].__len__()):
            coeff_image[i*4][j*4] = coeff_watermark[i][j]

    return coeff_image

def embed_watermark(watermark_array, orig_image):
    watermark_array_size = watermark_array[0].__len__()
    watermark_flat = watermark_array.ravel()
    ind = 0

    for x in range (0, orig_image.__len__(), 8):
        for y in range (0, orig_image.__len__(), 8):
            if ind < watermark_flat.__len__():
                subdct = orig_image[x:x+8, y:y+8]
                subdct[5][5] = watermark_flat[ind]
                orig_image[x:x+8, y:y+8] = subdct
                ind += 1 


    return orig_image

def apply_dct(image_array):
    size = image_array[0].__len__()
    all_subdct = np.empty((size, size))
    for i in range (0, size, 8):
        for j in range (0, size, 8):
            subpixels = image_array[i:i+8, j:j+8]
            subdct = dct(dct(subpixels.T, norm="ortho").T, norm="ortho")
            all_subdct[i:i+8, j:j+8] = subdct

    return all_subdct

def inverse_dct(all_subdct):
    size = all_subdct[0].__len__()
    all_subidct = np.empty((size, size))
    for i in range (0, size, 8):
        for j in range (0, size, 8):
            subidct = idct(idct(all_subdct[i:i+8, j:j+8].T, norm="ortho").T, norm="ortho")
            all_subidct[i:i+8, j:j+8] = subidct

    return all_subidct

def get_watermark(dct_watermarked_coeff, watermark_size):
    
    subwatermarks = []

    for x in range (0, dct_watermarked_coeff.__len__(), 8):
        for y in range (0, dct_watermarked_coeff.__len__(), 8):
            coeff_slice = dct_watermarked_coeff[x:x+8, y:y+8]
            subwatermarks.append(coeff_slice[5][5])

    watermark = np.array(subwatermarks).reshape(watermark_size, watermark_size)

    return watermark

def recover_watermark(image_array, model='haar', level = 1):


    coeffs_watermarked_image = process_coefficients(image_array, model, level=level)
    dct_watermarked_coeff = apply_dct(coeffs_watermarked_image[0])
    
    watermark_array = get_watermark(dct_watermarked_coeff, 128)

    watermark_array =  np.uint8(watermark_array)

#Save result
    img = Image.fromarray(watermark_array)
    img.save('recovered_watermark.jpg')
    return img

def print_image_from_array(image_array,name):
  
    image_array_copy = image_array.clip(0, 255)
    image_array_copy = image_array_copy.astype("uint8")
    img = Image.fromarray(image_array_copy)
    return img

def iwt(c):    
    s = c[0::2]
    d = c[1::2]
    l = len(s)

    a = d[0:l-1] - np.floor(0.5*(s[0:l-1]+s[1:l])) 
    b = d[l-1] - s[l-1] 
    d = np.concatenate((a, b), axis=None)   

    a = s[0] + np.floor(0.5*d[0] + 0.5)
    b = s[1:l] + np.floor(0.25*(d[1:l] + d[0:l-1]) + 0.5)
    s = np.concatenate((a, b), axis=None)
    
    return s, d

def iiwt(s,d):
    l = len(s)

    a = s[0] - np.floor(0.5*d[0] + 0.5)
    b = s[1:l] - np.floor(0.25*(d[1:l] + d[0:l-1]) + 0.5)
    s = np.concatenate((a, b), axis=None)

    a = d[0:l-1] + np.floor(0.5*(s[0:l-1]+s[1:l])) 
    b = d[l-1] + s[l-1] 
    d = np.concatenate((a, b), axis=None)   

    c2 = np.column_stack((s,d)).ravel()

    return c2

def imageOutput1():
    img = cv.imread(os.path.join(app.config['UPLOAD_FOLDER'], ))
    dim=(512,512)
    img = cv.resize(img, dim)