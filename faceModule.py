import numpy as np
import cv2
import torch
# from model import EmoModel

# model = EmoModel()

model = torch.load('filename.pt')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

face_cascade = cv2.CascadeClassifier('harr_cass.xml')
cap = cv2.VideoCapture(0)
emo = ['NE','HA','SA','SU','AN','DI', 'FE']
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

while(True):
    # Capture frame-by-frame
    ret, fr = cap.read()

    # Our operations on the frame come here
    Base_frame = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
    faces = face_cascade.detectMultiScale(Base_frame, 1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(Base_frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_color = Base_frame[y:y+h,x:x+w]
        roi_color = cv2.resize(roi_color,  dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
        roi_color = np.rollaxis(roi_color, 2, 0).astype('float32')
        img_data = np.reshape(roi_color, (1,3, 224, 224))
        image = torch.from_numpy(img_data)
        image = image.to(device)
        pred = model(image)
        pred = pred.cpu()
        pred = pred.detach().numpy()
        templist = list()
        for i in pred:
            templist.append(sigmoid(i)*100)
        # cv2.imshow("color", roi_color)
        print ('face detected', templist)
    # Display the resulting frame
    cv2.imshow('frame',Base_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()