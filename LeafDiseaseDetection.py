from PyQt5 import QtCore, QtGui, QtWidgets
import tensorflow as tf
import keras
from keras.models import model_from_json
import numpy as np
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import BatchNormalization

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

global filename
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(836, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(200, 0, 461, 81))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(310, 60, 181, 31))
        self.label_2.setObjectName("label_2")
        self.Imagelabel = QtWidgets.QLabel(self.centralwidget)
        self.Imagelabel.setGeometry(QtCore.QRect(250, 120, 281, 241))
        self.Imagelabel.setText("")
        self.Imagelabel.setObjectName("Imagelabel")
        self.loadbtn = QtWidgets.QPushButton(self.centralwidget)
        self.loadbtn.setGeometry(QtCore.QRect(50, 370, 131, 41))
        self.loadbtn.setObjectName("loadbtn")
        self.Classifybtn = QtWidgets.QPushButton(self.centralwidget)
        self.Classifybtn.setGeometry(QtCore.QRect(50, 430, 131, 41))
        self.Classifybtn.setObjectName("Classifybtn")
        self.output = QtWidgets.QTextEdit(self.centralwidget)
        self.output.setGeometry(QtCore.QRect(240, 420, 421, 51))
        self.output.setObjectName("output")
        self.Trainbtn = QtWidgets.QPushButton(self.centralwidget)
        self.Trainbtn.setGeometry(QtCore.QRect(50, 490, 131, 41))
        self.Trainbtn.setObjectName("Trainbtn")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(240, 390, 71, 31))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(350, 560, 211, 16))
        self.label_4.setObjectName("label_4")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.loadbtn.clicked.connect(self.Loadimg)
        self.Classifybtn.clicked.connect(self.Classify)
        self.Trainbtn.clicked.connect(self.Train)

    
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:26pt; font-weight:600;\">Leaf Disease detection</span></p></body></html>"))
        self.label_2.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:14pt; color:#aa0000;\">Using Deep Learning</span></p></body></html>"))
        self.loadbtn.setText(_translate("MainWindow", "Select Image"))
        self.Classifybtn.setText(_translate("MainWindow", "Classify"))
        self.output.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>"))
        self.Trainbtn.setText(_translate("MainWindow", "Train Model"))
        self.label_3.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:10pt; color:#aa0000;\">Output Data</span></p></body></html>"))
        self.label_4.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:9pt; font-weight:600; color:#aa0000;\">@Sureshkrish2005</span></p></body></html>"))


    def Loadimg(self):
        print('Loading Image...')
        
        filename,_ = QtWidgets.QFileDialog.getOpenFileName(None,"Select Data to Test","","All Files (*)")
        if filename:                
            print(filename)
            self.file=filename
            Pixmap=QtGui.QPixmap(filename)
            Pixmap=Pixmap.scaled(self.Imagelabel.width(),self.Imagelabel.height(),QtCore.Qt.KeepAspectRatio)

            self.Imagelabel.setPixmap(Pixmap)
            self.Imagelabel.setAlignment(QtCore.Qt.AlignCenter)

        
        
    def Classify(self):
            print('Classifying Data...')
            #modelfile,_=QtWidgets.QFileDialog.getOpenFileName(None,"Select modle to Test","","Json file (*.json)")
            jsonfile=open('model.json','r')
            loadedmodel=jsonfile.read()
            jsonfile.close()
            #weights,_=QtWidgets.QFileDialog.getOpenFileName(None,"Select weights to Test","","h5 file (*.h5)")
            model=model_from_json(loadedmodel)
            model.load_weights('model.h5')
            print('Model Loaded successfully.... ')

            diseases=["Apple___Apple_scab","Apple___Black_rot","Apple___Cedar_apple_rust","Apple___Healthy",
                   "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot","Corn_(maize)___Common_rust_",
                   "Corn_(maize)___Healthy","Corn_(maize)___Northern_Leaf_Blight","Grape___Black_rot",
                   "Grape___Esca_(Black_Measles)","Grape___Healthy","Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
                   "Potato___Early_blight","Potato___Healthy","Potato___Late_blight","Tomato___Bacterial_spot",
                   "Tomato___Early_blight","Tomato___Healthy","Tomato___Late_blight","Tomato___Leaf_Mold",
                   "Tomato___Septoria_leaf_spot","Tomato___Spider_mites Two-spotted_spider_mite","Tomato___Target_Spot",
                   "Tomato___Tomato_Yellow_Leaf_Curl_Virus","Tomato___Tomato_mosaic_virus"]

            
            path=self.file
            
            Testimg=keras.utils.load_img(path,target_size=(128,128))
            Testimg=tf.keras.preprocessing.image.img_to_array(Testimg)
            Testimg=np.expand_dims(Testimg,axis=0)

            result=model.predict(Testimg)
               
            DetectedDisease=diseases[result.argmax()]
            print("Detected Diseases : ",DetectedDisease)
            print(result.argmax())
            self.output.setText(DetectedDisease)
           
    def Train(self):
        print('Training Model....')
        model = Sequential()
        model.add(Conv2D(32, kernel_size = (3, 3), activation='relu', input_shape=(128,128, 3)))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())

        model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())
    
        model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())

        model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())

        model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())
    
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(25, activation = 'softmax'))

        model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])



        train_datagen = ImageDataGenerator(rescale = None,
                                       shear_range = 0.2,
                                       zoom_range = 0.2,
                                       horizontal_flip = True)
        
        test_datagen = ImageDataGenerator(rescale = 1./255)

        training_set = train_datagen.flow_from_directory('dataset/train',
                                                     target_size = (128, 128),
                                                     batch_size = 32,
                                                     class_mode = 'categorical')
        #print(test_datagen)
        labels = (training_set.class_indices)
        print(labels)

        test_set = test_datagen.flow_from_directory('dataset/val',
                                                    target_size = (128, 128),
                                                batch_size = 32,
                                                class_mode = 'categorical')

        labels2 = (test_set.class_indices)
        print(labels2)

        model.fit(training_set,steps_per_epoch = 375,
                             epochs = 10,
                             validation_data = test_set,
                             validation_steps = 125)




        model_json=model.to_json()
        with open("model/model.json", "w") as json_file:
            json_file.write(model_json)
            model.save_weights("model/model.h5")
            print("Saved model to disk")
        self.output.setText("Model Trained and Saved Successfully...")

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
