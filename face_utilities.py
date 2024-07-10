import cv2
import numpy as np
import dlib
from imutils import face_utils
import imutils
from collections import OrderedDict
import mediapipe as mp
import itertools
from statistics import mean 

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

class Face_utilities():
    '''
    This class contains all needed functions to work with faces in a frame
    '''
    
    def __init__(self, face_width = 200):
        self.detector = None
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
        self.drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.5)



        self.MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        self.age_list=['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']
        self.gender_list = ['Male', 'Female']
        
        self.desiredLeftEye = (0.35, 0.35)
        self.desiredFaceWidth = face_width
        self.desiredFaceHeight = None
        
        if self.desiredFaceHeight is None:
            self.desiredFaceHeight = self.desiredFaceWidth
    
        #last params
        self.last_age = None
        self.last_gender = None
        self.last_rects = None
        self.last_shape = None
        self.last_aligned_shape = None
        
        self.ROI_face = None
        self.ROI_forehead = None

    def face_alignment(self, frame, shape):
        # landmark=np.array(shape)
        # print(landmark.shape)
        ih, iw, _ = frame.shape

        # Extract left and right eye landmarks
        lStart, lEnd = 263, 374  # Indices for left eye in 468 landmarks
        rStart, rEnd = 374, 468  # Indices for right eye in 468 landmarks

        # LEFT_EYE_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_LEFT_EYE)))
        # RIGHT_EYE_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_RIGHT_EYE)))

        LEFT_EYE_INDEXES = [33, 133] 
        RIGHT_EYE_INDEXES = [263, 362]
        NOSE = [4]

        # LEFT_EYE_CENTER = [467]
        # RIGHT_EYE_CENTER = [473]

        # leftEyePts = shape[lStart:lEnd]
        # rightEyePts = shape[rStart:rEnd]
        leftEyePts = [shape[pt] for pt in LEFT_EYE_INDEXES]
        rightEyePts = [shape[pt] for pt in RIGHT_EYE_INDEXES]
        nosePt = [shape[pt] for pt in NOSE]
        # print(nosePt)

        # Converting to numpy
        leftEyePts = np.array(leftEyePts)
        rightEyePts = np.array(rightEyePts)
        shape = np.array(shape)

        # Calculate eye centers
        leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
        rightEyeCenter = rightEyePts.mean(axis=0).astype("int")
        # leftEyeCenter = [shape[LEFT_EYE_CENTER]]
        # rightEyeCenter = [shape[RIGHT_EYE_CENTER]]

        # Calculate angle between eye centroids
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX))

        # Compute the rotation matrix for face alignment
        eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) / 2,
                      (leftEyeCenter[1] + rightEyeCenter[1]) / 2)
        
        # nosecenter = (nosePt[0][0], nosePt[0][1])
        leftEye = (int(leftEyeCenter[0]),int(leftEyeCenter[1]))
        
        desiredRightEyeX = 1.0 - self.desiredLeftEye[0]
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
        desiredDist *= self.desiredFaceWidth
        scale = desiredDist / dist

        # print(type(nosecenter[0]))
        # print(type(leftEye[0]))
        # print(eyesCenter, angle, scale)
        M = cv2.getRotationMatrix2D(eyesCenter, angle= 0, scale= scale)

        # Define the translation amounts along x and y axes
        dx = -175  # shift left by 50 pixels
        dy = 0  # shift down by 30 pixels

        # Modify the translation part of the transformation matrix M
        M[0, 2] += dx
        M[1, 2] += dy

        # Apply the affine transformation to align the face
        aligned_face = cv2.warpAffine(frame, M, (iw, ih),
                                       flags=cv2.INTER_CUBIC)

        # Transform the landmarks using the rotation matrix
        aligned_shape = cv2.transform(shape.reshape(-1, 1, 2), M)
        aligned_shape = aligned_shape.squeeze()

        return aligned_face, aligned_shape
    
    # Using MEdiaPIPE
    def face_detection(self, frame):
        '''
        Detect faces in a frame
        
        Args:
            frame (cv2 image): a normal frame grab from camera or video
                
        Outputs:
            landmarks (array): facial landmarks' co-ords in format of of tuples (x,y)
        '''
        self.faces = []
        self.landmarks = []
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                self.faces.append(face_landmarks.landmark)
                for idx, landmark in enumerate(face_landmarks.landmark):
                    self.landmarks.append((landmark.x, landmark.y))
                
        return self.faces, self.landmarks

    
    def get_landmarks(self, frame):
        results = self.face_detection.process(frame)
        if results.detections:
            landmarks = []
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                
                keypoints = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if keypoints.multi_face_landmarks:
                    for face_landmarks in keypoints.multi_face_landmarks:
                        for i, landmark in enumerate(face_landmarks.landmark):
                            x = int(landmark.x * iw)
                            y = int(landmark.y * ih)
                            landmarks.append((x, y))
                
                return landmarks, [bbox]

        return None, None

    def ROI_extraction(self, face_image, landmarks):
        # Define landmark indices for the cheeks based on Mediapipe face mesh
        # These indices are based on the Mediapipe FaceMesh landmarks
        RIGHT_CHEEK_INDICES = [234, 132, 62, 172]
        LEFT_CHEEK_INDICES = [454, 234, 33, 263]
        FOREHEAD_INDICES = [54, 68, 103, 104, 67, 69, 108, 109, 10, 151, 338, 337, 297, 299, 332, 333, 284, 298]

        forehead_points= [landmarks[i] for i in FOREHEAD_INDICES]
        # print(forehead_points)
        # forehead_points = [(int(point[0] * face_image.shape[1]), int(point[1] * face_image.shape[0])) 
                            #   for point in forehead_points]
        # print(forehead_points) #min=45408 max: 278
        roi = face_image[min([point[1] for point in forehead_points]): max([point[1] for point in forehead_points]),
                         min([point[0] for point in forehead_points]): max(point[0] for point in forehead_points)]
        # print("face_image",face_image.shape)
        # print("landmarks ", landmarks.shape)
        # print("roi_extraction: ", roi.shape)
        # cv2.imwrite('image_with_rectangle.png', roi)
        return roi
    
    def ROI_extraction_no_mask(self, face_image, landmarks):
        # Define landmark indices for forehead based on Mediapipe face mesh
        roi1 = self.ROI_extraction(face_image, landmarks)
        # cv2.imwrite('Face_image.png', face_image)

        # Define landmark indices for the face based on Mediapipe face mesh
        LEFT_EYE_INDEXES = list(set(itertools.chain(*self.mp_face_mesh.FACEMESH_LEFT_EYE)))
        RIGHT_EYE_INDEXES = list(set(itertools.chain(*self.mp_face_mesh.FACEMESH_RIGHT_EYE)))
        LEFT_EYEBROW_INDEXES=list(set(itertools.chain(*self.mp_face_mesh.FACEMESH_LEFT_EYEBROW)))
        RIGHT_EYEBROW_INDEXES=list(set(itertools.chain(*self.mp_face_mesh.FACEMESH_RIGHT_EYEBROW)))
        forehead = [54, 68, 103, 104, 67, 69, 108, 109, 10, 151, 338, 337, 297, 299, 332, 333, 284, 298]
        exclude_region = [6, 8, 9, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33, 34, 35, 56, 71, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 124, 127, 128, 130, 139, 143, 156, 162, 168, 188, 189, 190, 193, 196, 197, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 264, 265, 277,  286, 301, 339, 340, 341, 342, 343, 345, 346, 347, 348, 349, 350, 351, 353, 356, 357, 359, 368, 372, 383, 389, 399, 412, 413, 414, 417, 419, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 463, 464, 465, 467]
        all_excluded_region = LEFT_EYE_INDEXES + RIGHT_EYE_INDEXES + LEFT_EYEBROW_INDEXES + RIGHT_EYEBROW_INDEXES + exclude_region + forehead

        facepts=[]
        for i in range(len(landmarks)):
            if i not in all_excluded_region:
                facepts.append(landmarks[i])
        # print(facepts)

        roi2 = face_image[min([point[1] for point in facepts]): max([point[1] for point in facepts]),
                         min([point[0] for point in facepts]): max(point[0] for point in facepts)]
        
        # print("roi forehead",roi1.shape)
        
        # cv2.imwrite('image_with_rectangle_forehead.png', roi1)
        # print("roi face: ", roi2.shape)
        # cv2.imwrite('image_with_rectangle_face.png', roi2)

        # print("face image", face_image.shape)
        # print("landmarks: ", landmarks.shape)

        return roi1, roi2


    def facial_landmarks_remap(self,shape):
        '''
        Need to re-arrange some facials landmarks to get correct params for cv2.fillConvexPoly
        
        Args: 
            shape (array): facial landmarks' co-ords in format of of tuples (x,y)
            
        Outputs:
            remapped_shape (array): facial landmarks after re-arranged
        '''
        
        remapped_shape = shape.copy()
        # left eye brow
        remapped_shape[17] = shape[26]
        remapped_shape[18] = shape[25]
        remapped_shape[19] = shape[24]
        remapped_shape[20] = shape[23]
        remapped_shape[21] = shape[22]
        # right eye brow
        remapped_shape[22] = shape[21]
        remapped_shape[23] = shape[20]
        remapped_shape[24] = shape[19]
        remapped_shape[25] = shape[18]
        remapped_shape[26] = shape[17]
        # neatening 
        remapped_shape[27] = shape[0]
        
        remapped_shape = cv2.convexHull(shape)        
        return remapped_shape       
    
    # Change according to mediapipe
    def no_age_gender_face_process(self, frame):
        '''
        full process to extract face, ROI but no age and gender detection
        
        Args:
            frame (cv2 image): input frame 
            type (str): 5 or 68 landmarks
            
        Outputs:
            rects (array): detected faces as rectangles
            face (cv2 image): face
            shape (array): facial landmarks' co-ords in format of tuples (x,y)
            aligned_face (cv2 image): face after alignment
            aligned_shape (array): facial landmarks' co-ords of the aligned face in format of tuples (x,y)
        
        '''
        shape, rects = self.get_landmarks(frame)
        if shape is None:
            return None
        
        # print("line 714",rects[0])
        # (x, y, w, h) = face_utils.rect_to_bb(rects[0])
        # x = rects[0][0]
        # y = rects[0][1]
        # w = rects[1][0] - rects[0][0]
        # h = rects[1][1] - rects[0][1]
        x, y, w, h = rects[0]

        face = frame[y:y+h,x:x+w]
        aligned_face,aligned_shape = self.face_alignment(frame, shape)
                
        return rects, face, shape, aligned_face, aligned_shape
        
    def face_full_process(self, frame, type, face_detect_on, age_gender_on):
        '''
        full process to extract face, ROI 
        face detection and facial landmark run every 3 frames
        age and gender detection runs every 6 frames
        last values of detections are used in other frames to reduce the time of the process
        ***NOTE: need 2 time facial landmarks, 1 for face alignment and 1 for facial landmarks in aligned face
        ***TODO: find facial landmarks after rotate (find co-ords after rotating) so don't need to do 2 facial landmarks
        Args:
            frame (cv2 image): input frame 
            type (str): 5 or 68 landmarks
            face_detect_on (bool): flag to run face detection and facial landmarks
            age_gender_on (bool): flag to run age gender detection
            
        Outputs:
            rects (array): detected faces as rectangles
            face (cv2 image): face
            (age, gender) (str,str): age and gender
            shape (array): facial landmarks' co-ords in format of tuples (x,y)
            aligned_face (cv2 image): face after alignment
            aligned_shape (array): facial landmarks' co-ords of the aligned face in format of tuples (x,y)
            #mask (cv2 image): mask of the face after fillConvexPoly
        '''
        
        #assign from last params
        age = self.last_age
        gender = self.last_gender
        rects = self.last_rects
        shape = self.last_shape
        aligned_shape = self.last_aligned_shape
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if face_detect_on:
            if(type=="5"):
                shape, rects = self.get_landmarks(frame, "5")
                #mask = None
                
                if shape is None:
                    return None
            else:    
                shape, rects = self.get_landmarks(frame, "68")
                if shape is None:
                    return None
        
        (x, y, w, h) = face_utils.rect_to_bb(rects[0])
        
        face = frame[y:y+h,x:x+w]        
        
        if age_gender_on:
            age, gender = self.age_gender_detection(face)
        
        aligned_face, aligned_face = self.face_alignment(frame, shape)
        
        #assign to last params
        self.last_age = age
        self.last_gender = gender
        self.last_rects = rects
        self.last_shape = shape
        self.last_aligned_shape = aligned_shape
        
        return rects, face, (age, gender), shape, aligned_face, aligned_shape
    
        
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    face_processor = Face_utilities()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform face detection and landmark estimation using MediaPipe here
        results = face_processor.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            landmarks = []
            for face_landmarks in results.multi_face_landmarks:
                for landmark in face_landmarks.landmark:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    landmarks.append((x, y))

            aligned_face, aligned_shape = face_processor.face_alignment(frame, np.array(landmarks))

            # Visualize the aligned face and landmarks (draw on 'aligned_face' if needed)
            cv2.imshow('Aligned Face', aligned_face)

            # Draw landmarks on aligned_face if needed
            for landmark in aligned_shape:
                cv2.circle(aligned_face, tuple(map(int, landmark)), 2, (0, 255, 0), -1)

            cv2.imshow('Aligned Face with Landmarks', aligned_face)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()