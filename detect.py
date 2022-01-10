import cv2
from PIL import Image
import torch
from torchvision import transforms

class Drowsiness:

    def __init__(self):

        # Loading cascades
        self.face_cascade = cv2.CascadeClassifier('xmls/haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier('xmls/haarcascade_eye.xml')

        # Load eye classifier
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.eyes_model = torch.load('models/eyes_resnet18_128x128.pt', map_location=self.device)
        self.eyes_model.eval()
        self.classes = {0: 'closed', 1: 'open'}

    def classify_eye(self, crop):

        ts = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
        
        # Converting to tensor
        crop = Image.fromarray(crop)
        crop_tensor = ts(crop).float().unsqueeze_(0).to(device=self.device)

        output = self.eyes_model(crop_tensor)
        index = self.classes[output.data.cpu().numpy().argmax()]
        
        return index

    
    def get_detections(self, gray, frame):
        
        face_dets = list()
        eye_dets = list()

        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        # 1.3 is kernel size
        # 5 is number of neighbors

        for (x, y, w, h) in faces:
            
            face_dets.append((x, y, w, h))

            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 3)

            for (ex, ey, ew, eh) in eyes:
                index = self.classify_eye(roi_gray[ey - 5:ey+eh + 5, ex - 5:ex+ew + 5])
                eye_dets.append((ex, ey, ew, eh, index))

        return face_dets, eye_dets
    
    def draw_detections(self, frame, face_dets, eye_dets):
        try:
            for (x, y, w, h) in face_dets:
                cv2.rectangle(frame, (x, y), (x+w, y+h),(255, 0, 0), 2)

            roi_color = frame[y:y+h, x:x+w]
            for (ex, ey, ew, eh, index) in eye_dets:
                color = None

                if index == 'open':
                    color = (0, 255, 0)
                elif index == 'closed':
                    color = (0, 0, 255)
                else:
                    color = (0, 255, 255)

                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), color, 2)
        except:
            pass

        return frame
    
    def run(self):
        
        # Initializing video capture
        video_capture = cv2.VideoCapture(0)

        while True:
            ret, frame = video_capture.read()
            
            if ret:
                # Now we convert read frame to gray coz cascading only works on gray

                gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                # calling detector

                face_dets, eye_dets = self.get_detections(gray, frame)
                canvas = self.draw_detections(frame, face_dets, eye_dets)

                cv2.imshow("Drowsiness Detection", canvas)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    drowsiness = Drowsiness()
    drowsiness.run()
    

