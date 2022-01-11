import cv2
from PIL import Image
import torch
from torchvision import transforms
from centerface import CenterFace

class Drowsiness:

    def __init__(self):

        # Load face detection model
        self.centerface_model = CenterFace()

        # Load eye classifier
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.eyes_model = torch.load('models/eyes_resnet18_128x128.pt', map_location=self.device)
        self.eyes_model.eval()
        self.classes = {0: 'closed', 1: 'open'}

        cv2.namedWindow("Drowsiness Detection", cv2.WINDOW_NORMAL)

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
    
    def get_detections_centerface(self, frame):
        
        h, w = frame.shape[:2]
        dets, points = self.centerface_model(frame, h, w, threshold=0.30)
        eyes = list()

        for det, fts in zip(dets, points):
            x1, y1, x2, y2, prob = det
            
            left_eye_x = int(fts[0])
            left_eye_y = int(fts[1])
            right_eye_x = int(fts[2])
            right_eye_y = int(fts[3])

            left_x_factor = abs(left_eye_x - x1) * 0.55
            left_y_factor = abs(left_eye_y - y1) * 0.35
            right_x_factor = abs(right_eye_x - x2) * 0.55
            right_y_factor = abs(right_eye_y - y1) * 0.35

            left_eye_x1 = int(left_eye_x - left_x_factor)
            left_eye_y1 = int(left_eye_y - left_y_factor)
            left_eye_x2 = int(left_eye_x + left_x_factor)
            left_eye_y2 = int(left_eye_y + left_y_factor)

            right_eye_x1 = int(right_eye_x - right_x_factor)
            right_eye_y1 = int(right_eye_y - right_y_factor)
            right_eye_x2 = int(right_eye_x + right_x_factor)
            right_eye_y2 = int(right_eye_y + right_y_factor)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            index_left = self.classify_eye(gray[left_eye_y1:left_eye_y2, left_eye_x1:left_eye_x2])
            index_right = self.classify_eye(gray[right_eye_y1:right_eye_y2, right_eye_x1:right_eye_x2])

            eyes.append([
                left_eye_x1, left_eye_y1, left_eye_x2, left_eye_y2, index_left,
                right_eye_x1, right_eye_y1, right_eye_x2, right_eye_y2, index_right
            ])

        return dets, points, eyes
    
    def get_color(self, index):
        color = (0, 0, 0)
        if index == 'open':
            color = (0, 255, 0)
        elif index == 'closed':
            color = (0, 0, 255)
        else:
            color = (0, 255, 255)
        
        return color

    def draw_detections_centerface(self, frame, dets, points, eyes):
        
        try:
            for det, fts, eye in zip(dets, points, eyes):
                x1, y1, x2, y2, prob = det
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),(255, 0, 0), 2)

                for i in range(0, len(fts), 2):
                    cv2.circle(frame, (int(fts[i]), int(fts[i+1])), 2, (0, 0, 255), -1)
                
                cv2.rectangle(frame, (eye[0], eye[1]), (eye[2], eye[3]), self.get_color(eye[4]), 2)
                cv2.rectangle(frame, (eye[5], eye[6]), (eye[7], eye[8]), self.get_color(eye[9]), 2)
                
        except:
            pass

        return frame
    
    def run(self):
        
        # Initializing video capture
        video_capture = cv2.VideoCapture(2)

        while True:
            ret, frame = video_capture.read()
            
            if ret:
                # Now we convert read frame to gray coz cascading only works on gray

                gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

                dets, points, eyes = self.get_detections_centerface(frame)
                canvas = self.draw_detections_centerface(frame, dets, points, eyes)

                cv2.imshow("Drowsiness Detection", canvas)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    drowsiness = Drowsiness()
    drowsiness.run()