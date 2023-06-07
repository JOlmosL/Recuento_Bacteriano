#Modulos
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

#Lectura                                    
red = cv2.dnn.readNetFromDarknet("./yolov3.cfg", r"./backup/yolov3_2000.weights")

#Clases
classes = ['bacteria']

#Input
Path = "./Images/"
img_name = input("Nombre de la muestra: ")
if (img_name in os.listdir(Path)):
    print("La muestra existe, procediendo a realizar el conteo bacteriano...")
    img_path = Path + img_name
    img = cv2.imread(img_path)
    img = cv2.resize(img,(512,512))
    hight,width,_ = img.shape
    cv2.imshow('Original ' + img_name, img)
    blob = cv2.dnn.blobFromImage(img, 1/255,(416,416),(0,0,0),swapRB = True,crop= False)
    
    red.setInput(blob)
    
    output_layers_name = red.getUnconnectedOutLayersNames()
    
    layerOutputs = red.forward(output_layers_name)
    
    boxes =[]
    confidences = []
    class_ids = []
    
    for output in layerOutputs:
        for detection in output:
            score = detection[5:]
            class_id = np.argmax(score)
            confidence = score[class_id]
            if confidence > 0.1:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * hight)
                w = int(detection[2] * width)
                h = int(detection[3] * hight)
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                boxes.append([x,y,w,h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)            
                
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, .5, .4)
    
    boxes =[]
    confidences = []
    class_ids = []
    
    for output in layerOutputs:
        for detection in output:
            score = detection[5:]
            class_id = np.argmax(score)
            confidence = score[class_id]
            if confidence > 0:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * hight)
                w = int(detection[2] * width)
                h = int(detection[3]* hight)
    
                x = int(center_x - w/2)
                y = int(center_y - h/2)
    
                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)
                
    indexes = cv2.dnn.NMSBoxes(boxes, confidences,.5,.4)
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size = (len(boxes), 3))
    if  len(indexes) > 0:
        count = len(indexes)
        for i in indexes.flatten():
            x,y,w,h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h),(0, 255, 0),1)
            cv2.putText(img, "Total: " + str(count), (0, 30), font, 2, (0, 255, 0), 2)
    
    cv2.imshow('Muestra ' + img_name, img)
    cv2.waitKey(0)  
    cv2.destroyAllWindows()    

else:
    print("No existe la muestra mencionada.\nFavor de verificar que la muestra existe.")
    exit()