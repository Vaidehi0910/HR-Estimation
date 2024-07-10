import cv2
import numpy as np
import time
# from face_detection import FaceDetection
from scipy import signal
from face_utilities import Face_utilities
# from face_utilities import FaceUtilities
from signal_processing import Signal_processing
from imutils import face_utils
from keras.models import load_model
from keras.preprocessing import image
import itertools
import mediapipe as mp
# from sklearn.decomposition import FastICA

class Process(object):
    def __init__(self):
        self.frame_in = np.zeros((10, 10, 3), np.uint8)
        # self.frame_in = 
        self.frame_ROI = np.zeros((10, 10, 3), np.uint8)
        self.frame_out = np.zeros((10, 10, 3), np.uint8)
        self.ROI_forehead = np.zeros((10,10,3), np.uint8)
        self.ROI_face = np.zeros((10,10,3), np.uint8)
        self.samples = []
        self.buffer_size = 100
        self.times = [] 
        self.data_buffer = []
        self.fps = 0
        self.fft = []
        self.freqs = []
        self.t0 = time.time()
        self.bpm = 0
        # self.fd = FaceDetection()
        self.bpms = []
        self.peaks = []
        self.fu = Face_utilities()
        self.sp = Signal_processing()
        self.mymodel=load_model('mymodel.h5')
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mask = None
        

        #self.red = np.zeros((256,256,3),np.uint8)
        
    def extractColor(self, frame):
        
        #r = np.mean(frame[:,:,0])
        g = np.mean(frame[:,:,1])
        #b = np.mean(frame[:,:,2])
        #return r, g, b
        return g
        
    def run(self):
        # frame, face_frame, ROI1, ROI2, status, mask = self.fd.face_detect(self.frame_in)
        
        frame = self.frame_in

        ret_process = self.fu.no_age_gender_face_process(frame)
        if ret_process is None:
            print("ret_process is none")
            return False
        rects, face, shape, aligned_face, aligned_shape = ret_process
        original_aligned_face = aligned_face.copy()
        # print("aligned face",aligned_face.shape)
        # print("aligned shape", aligned_shape.shape)
        x, y, w, h = rects[0]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        
        if self.mask== None:
            face_img = frame[y:y+h, x:x+w]
            cv2.imwrite('temp.jpg',face_img)
            test_image=image.load_img('temp.jpg',target_size=(150,150,3))
            test_image=image.img_to_array(test_image)
            test_image=np.expand_dims(test_image,axis=0)
            pred=self.mymodel.predict(test_image)[0][0]
            if pred==1:
                self.mask= False
                print("NO MASK")
            else:
                self.mask = True
                print("MASK")
        if not self.mask:
            LEFT_EYE_INDEXES = list(set(itertools.chain(*self.mp_face_mesh.FACEMESH_LEFT_EYE)))
            RIGHT_EYE_INDEXES = list(set(itertools.chain(*self.mp_face_mesh.FACEMESH_RIGHT_EYE)))
            LEFT_EYEBROW_INDEXES=list(set(itertools.chain(*self.mp_face_mesh.FACEMESH_LEFT_EYEBROW)))
            RIGHT_EYEBROW_INDEXES=list(set(itertools.chain(*self.mp_face_mesh.FACEMESH_RIGHT_EYEBROW)))
            exclude_region = [6, 8, 9, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33, 34, 35, 56, 71, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 124, 127, 128, 130, 139, 143, 156, 162, 168, 188, 189, 190, 193, 196, 197, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 264, 265, 277,  286, 301, 339, 340, 341, 342, 343, 345, 346, 347, 348, 349, 350, 351, 353, 356, 357, 359, 368, 372, 383, 389, 399, 412, 413, 414, 417, 419, 441, 442, 443, 444, 445, 446, 448, 449, 450, 451, 452, 453, 463, 464, 465, 467]
            all_excluded_region = LEFT_EYE_INDEXES + RIGHT_EYE_INDEXES + LEFT_EYEBROW_INDEXES + RIGHT_EYEBROW_INDEXES + exclude_region

            excluded_points = [aligned_shape[i] for i in all_excluded_region]
            # print(excluded_points)
            # facepoints = aligned_shape - excluded_points
            for (x,y) in aligned_shape:
                if any((x_pt, y_pt) == (x,y) for (x_pt, y_pt) in excluded_points):
                    pass
                else:
                    cv2.circle(aligned_face, (x, y), 1, (0, 0, 255), -1)
            
            
            ROIs = self.fu.ROI_extraction_no_mask(original_aligned_face,aligned_shape)

            self.ROI_forehead = ROIs[0]
            self.ROI_face = ROIs[1]
            green_val = self.sp.extract_color_no_mask(ROIs)
        else:
            
            forehead = [54, 68, 103, 104, 67, 69, 108, 109, 10, 151, 338, 337, 297, 299, 332, 333, 284, 298]
            # print(shape[54], ":", aligned_shape[54])
            foreheadpts = [aligned_shape[i] for i in forehead]
            for (x, y) in aligned_shape: 
                if any((x_pt, y_pt) == (x, y) for (x_pt, y_pt) in foreheadpts):
                    cv2.circle(aligned_face, (x, y), 1, (0, 0, 255), -1)
            # print("forehead:", foreheadpts)
            ROIs = self.fu.ROI_extraction(original_aligned_face, aligned_shape)
            self.ROI_forehead = ROIs
            green_val = self.sp.extract_color(ROIs)

        # print(green_val)
        self.frame_out = frame
        self.frame_ROI = aligned_face
        # cv2.box(aligned_face
        # cv2.rectangle(ROIs)

        # Save the image
        
        # ROIs = np.array(ROIs)
        # print(ROIs.shape)
        L = len(self.data_buffer)

        g = green_val
        
        if(abs(g-np.mean(self.data_buffer))>10 and L>99): #remove sudden change, if the avg value change is over 10, use the mean of the data_buffer
            g = self.data_buffer[-1]
        
        self.times.append(time.time() - self.t0)
        self.data_buffer.append(g)

        #only process in a fixed-size buffer
        if L > self.buffer_size:
            self.data_buffer = self.data_buffer[-self.buffer_size:]
            self.times = self.times[-self.buffer_size:]
            self.bpms = self.bpms[-self.buffer_size//2:]
            L = self.buffer_size
            
        processed = np.array(self.data_buffer)
        processed = [x for x in processed if str(x) != 'nan']
        # print("processed: ",processed)
        # start calculating after the first 10 frames
        if L == self.buffer_size:
            
            self.fps = float(L) / (self.times[-1] - self.times[0])#calculate HR using a true fps of processor of the computer, not the fps the camera provide
            even_times = np.linspace(self.times[0], self.times[-1], L)
            
            processed = signal.detrend(processed)#detrend the signal to avoid interference of light change
            interpolated = np.interp(even_times, self.times, processed) #interpolation by 1
            interpolated = np.hamming(L) * interpolated#make the signal become more periodic (advoid spectral leakage)
            #norm = (interpolated - np.mean(interpolated))/np.std(interpolated)#normalization
            norm = interpolated/np.linalg.norm(interpolated)
            raw = np.fft.rfft(norm*30)#do real fft with the normalization multiplied by 10
            # print("norm",norm)
            # print("raw", raw)
            self.freqs = float(self.fps) / L * np.arange(L / 2 + 1)
            # print("freq", self.freqs)
            freqs = 60. * self.freqs
            
            # idx_remove = np.where((freqs < 50) & (freqs > 180))
            # raw[idx_remove] = 0
            
            self.fft = np.abs(raw)**2#get amplitude spectrum
            


            idx = np.where((freqs > 50) & (freqs < 180))#the range of frequency that HR is supposed to be within 
            pruned = self.fft[idx]
            pfreq = freqs[idx]
            
            self.freqs = pfreq 
            self.fft = pruned
            # max_index = np.argmax(self.fft)

            idx2 = np.argmax(pruned)#max in the range can be HR
            
            
            self.bpm = self.freqs[idx2]
            # print("BPM", self.bpm)
            self.bpms.append(self.bpm)
            
            processed = self.butter_bandpass_filter(processed,0.8,3,self.fps,order = 2)
            #ifft = np.fft.irfft(raw)
        self.samples = processed # multiply the signal with 5 for easier to see in the plot
        #TODO: find peaks to draw HR-like signal.
        
        # if(mask.shape[0]!=10): 
        #     out = np.zeros_like(aligned_face)
        #     mask = mask.astype(np.bool)
        #     out[mask] = aligned_face[mask]
        #     if(processed[-1]>np.mean(processed)):
        #         out[mask,2] = 180 + processed[-1]*10
        #     aligned_face[mask] = out[mask]
            
            
        #cv2.imshow("face", face_frame)
        #out = cv2.add(face_frame,out)
        # else:
            # cv2.imshow("face", face_frame)
        return True
    
    def reset(self):
        self.frame_in = np.zeros((10, 10, 3), np.uint8)
        self.frame_ROI = np.zeros((10, 10, 3), np.uint8)
        self.frame_out = np.zeros((10, 10, 3), np.uint8)
        self.ROI_forehead = np.zeros((10, 10, 3), np.uint8)
        self.ROI_face = np.zeros((10, 10, 3), np.uint8)
        self.samples = []
        self.times = [] 
        self.data_buffer = []
        self.fps = 0
        self.fft = []
        self.freqs = []
        self.t0 = time.time()
        self.bpm = 0
        self.bpms = []
        self.mask = None
        
    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        # print("fps",fs)
        # nyq = 1/(2 * fs)
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        return b, a


    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = signal.lfilter(b, a, data)
        return y 

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
