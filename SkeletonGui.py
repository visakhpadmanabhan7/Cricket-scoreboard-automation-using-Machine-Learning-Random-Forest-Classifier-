import cv2
import math
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,precision_score,recall_score
from sklearn.ensemble import RandomForestClassifier

#specify the path for 2 files 

protoFile = "pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile = "pose_iter_160000.caffemodel"
#Read the network into memory

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

def angle(p1, p2):  
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1]) #anticlockwise angle
    return np.rad2deg((ang1 - ang2) % (2 * np.pi)) #clockwise and degree conversion

class Ui_SkeletonWindow(object):
    def setupUi(self, SkeletonWindow):
        SkeletonWindow.setObjectName("SkeletonWindow")
        SkeletonWindow.resize(1200, 633)
        SkeletonWindow.setStyleSheet("QWidget{\n"
"background-image:url(cs1.jpg);\n"
"}\n"
"QLabel{\n"
"background-color: white;\n"
"selection-background-color: white;\n"

"color: white;\n"
"font-size:15px;\n"
"font-family:Times new Roman;\n"
"font:bold;\n"
"}\n"
"QFrame{\n"
    "selection-background-color: transparent;\n"
    "background: transparent;\n"
"}\n"
"\n"
"QLineEdit{\n"
"selection-background-color: white;\n"
"background: white;\n"
"color: black;\n"
"}\n"
"QPushButton{\n"
"background-color: rgb(55, 155, 105);\n"
"color: rgb(255,255,255);\n"
"}\n"
"#label{\n"
 "color: rgb(255,255,255);\n"
  "font-size:20px;\n"
  "font-family:Times new roman;\n"
  "font:bold;\n"
"}\n"


    )
        self.centralwidget = QtWidgets.QWidget(SkeletonWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(1000, 30, 175,35))
        self.label.setObjectName("label")
        self.progress = QtWidgets.QProgressDialog("Please Wait!", "Cancel", 0, 100, SkeletonWindow)
        self.progress.setWindowModality(QtCore.Qt.WindowModal)
        
        self.txtImage = QtWidgets.QLineEdit(self.centralwidget)
        self.txtImage.setGeometry(QtCore.QRect(200, 60, 471, 31))
        self.txtImage.setObjectName("txtImage")
        self.btnBrowse = QtWidgets.QPushButton(self.centralwidget)
        self.btnBrowse.setGeometry(QtCore.QRect(700, 60, 171, 31))
        self.btnBrowse.setObjectName("btnBrowse")
        self.lblImage = QtWidgets.QLabel(self.centralwidget)
        self.lblImage.setGeometry(QtCore.QRect(100, 70, 71, 16))
        self.lblImage.setObjectName("lblImage")
        self.mainFrame = QtWidgets.QFrame(self.centralwidget)
        self.mainFrame.setGeometry(QtCore.QRect(120, 100, 761, 481))
        self.mainFrame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.mainFrame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.mainFrame.setObjectName("mainFrame")
        self.label_2 = QtWidgets.QLabel(self.mainFrame)
        self.label_2.setGeometry(QtCore.QRect(320, 10, 81, 16))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.mainFrame)
        self.label_3.setGeometry(QtCore.QRect(60, 70, 51, 16))
        self.label_3.setObjectName("label_3")
        self.txtHeadDis = QtWidgets.QLineEdit(self.mainFrame)
        self.txtHeadDis.setGeometry(QtCore.QRect(130, 70, 81, 20))
        self.txtHeadDis.setObjectName("txtHeadDis")
        self.txtHeadAng = QtWidgets.QLineEdit(self.mainFrame)
        self.txtHeadAng.setGeometry(QtCore.QRect(250, 70, 81, 20))
        self.txtHeadAng.setObjectName("txtHeadAng")
        self.txtLSAng = QtWidgets.QLineEdit(self.mainFrame)
        self.txtLSAng.setGeometry(QtCore.QRect(250, 110, 81, 20))
        self.txtLSAng.setObjectName("txtLSAng")
        self.label_4 = QtWidgets.QLabel(self.mainFrame)
        self.label_4.setGeometry(QtCore.QRect(60, 110, 51, 16))
        self.label_4.setObjectName("label_4")
        self.txtLSDis = QtWidgets.QLineEdit(self.mainFrame)
        self.txtLSDis.setGeometry(QtCore.QRect(130, 110, 81, 20))
        self.txtLSDis.setObjectName("txtLSDis")
        self.txtLElbowAng = QtWidgets.QLineEdit(self.mainFrame)
        self.txtLElbowAng.setGeometry(QtCore.QRect(250, 150, 81, 20))
        self.txtLElbowAng.setObjectName("txtLElbowAng")
        self.label_5 = QtWidgets.QLabel(self.mainFrame)
        self.label_5.setGeometry(QtCore.QRect(60, 150, 51, 16))
        self.label_5.setObjectName("label_5")
        self.txtLElbowDis = QtWidgets.QLineEdit(self.mainFrame)
        self.txtLElbowDis.setGeometry(QtCore.QRect(130, 150, 81, 20))
        self.txtLElbowDis.setObjectName("txtLElbowDis")
        self.txtLWristAng = QtWidgets.QLineEdit(self.mainFrame)
        self.txtLWristAng.setGeometry(QtCore.QRect(250, 190, 81, 20))
        self.txtLWristAng.setObjectName("txtLWristAng")
        self.label_6 = QtWidgets.QLabel(self.mainFrame)
        self.label_6.setGeometry(QtCore.QRect(60, 190, 51, 16))
        self.label_6.setObjectName("label_6")
        self.txtLWristDis = QtWidgets.QLineEdit(self.mainFrame)
        self.txtLWristDis.setGeometry(QtCore.QRect(130, 190, 81, 20))
        self.txtLWristDis.setObjectName("txtLWristDis")
        self.txtLHibAng = QtWidgets.QLineEdit(self.mainFrame)
        self.txtLHibAng.setGeometry(QtCore.QRect(250, 230, 81, 20))
        self.txtLHibAng.setObjectName("txtLHibAng")
        self.label_7 = QtWidgets.QLabel(self.mainFrame)
        self.label_7.setGeometry(QtCore.QRect(60, 230, 51, 16))
        self.label_7.setObjectName("label_7")
        self.txtLHibDis = QtWidgets.QLineEdit(self.mainFrame)
        self.txtLHibDis.setGeometry(QtCore.QRect(130, 230, 81, 20))
        self.txtLHibDis.setObjectName("txtLHibDis")
        self.txtLKneeAng = QtWidgets.QLineEdit(self.mainFrame)
        self.txtLKneeAng.setGeometry(QtCore.QRect(250, 270, 81, 20))
        self.txtLKneeAng.setObjectName("txtLKneeAng")
        self.label_8 = QtWidgets.QLabel(self.mainFrame)
        self.label_8.setGeometry(QtCore.QRect(60, 270, 51, 16))
        self.label_8.setObjectName("label_8")
        self.txtLKneeDis = QtWidgets.QLineEdit(self.mainFrame)
        self.txtLKneeDis.setGeometry(QtCore.QRect(130, 270, 81, 20))
        self.txtLKneeDis.setObjectName("txtLKneeDis")
        self.txtLFootDis = QtWidgets.QLineEdit(self.mainFrame)
        self.txtLFootDis.setGeometry(QtCore.QRect(130, 310, 81, 20))
        self.txtLFootDis.setObjectName("txtLFootDis")
        self.txtLFootAng = QtWidgets.QLineEdit(self.mainFrame)
        self.txtLFootAng.setGeometry(QtCore.QRect(250, 310, 81, 20))
        self.txtLFootAng.setObjectName("txtLFootAng")
        self.label_9 = QtWidgets.QLabel(self.mainFrame)
        self.label_9.setGeometry(QtCore.QRect(60, 310, 51, 16))
        self.label_9.setObjectName("label_9")
        self.txtRElbowAng = QtWidgets.QLineEdit(self.mainFrame)
        self.txtRElbowAng.setGeometry(QtCore.QRect(620, 110, 81, 20))
        self.txtRElbowAng.setObjectName("txtRElbowAng")
        self.txtRSAng = QtWidgets.QLineEdit(self.mainFrame)
        self.txtRSAng.setGeometry(QtCore.QRect(620, 70, 81, 20))
        self.txtRSAng.setObjectName("txtRSAng")
        self.label_10 = QtWidgets.QLabel(self.mainFrame)
        self.label_10.setGeometry(QtCore.QRect(430, 190, 51, 16))
        self.label_10.setObjectName("label_10")
        self.label_11 = QtWidgets.QLabel(self.mainFrame)
        self.label_11.setGeometry(QtCore.QRect(430, 270, 51, 16))
        self.label_11.setObjectName("label_11")
        
        self.txtRSDis = QtWidgets.QLineEdit(self.mainFrame)
        self.txtRSDis.setGeometry(QtCore.QRect(500, 70, 81, 20))
        self.txtRSDis.setObjectName("txtRSDis")
        self.txtWaisteDis = QtWidgets.QLineEdit(self.mainFrame)
        self.txtWaisteDis.setGeometry(QtCore.QRect(500, 190, 81, 20))
        self.txtWaisteDis.setObjectName("txtWaisteDis")
        self.txtRFootDis = QtWidgets.QLineEdit(self.mainFrame)
        self.txtRFootDis.setGeometry(QtCore.QRect(500, 310, 81, 20))
        self.txtRFootDis.setObjectName("txtRFootDis")
        self.txtRHibDis = QtWidgets.QLineEdit(self.mainFrame)
        self.txtRHibDis.setGeometry(QtCore.QRect(500, 230, 81, 20))
        self.txtRHibDis.setObjectName("txtRHibDis")
        self.label_12 = QtWidgets.QLabel(self.mainFrame)
        self.label_12.setGeometry(QtCore.QRect(430, 150, 51, 16))
        self.label_12.setObjectName("label_12")
        self.label_13 = QtWidgets.QLabel(self.mainFrame)
        self.label_13.setGeometry(QtCore.QRect(430, 110, 51, 16))
        self.label_13.setObjectName("label_13")
        self.txtRHibAng = QtWidgets.QLineEdit(self.mainFrame)
        self.txtRHibAng.setGeometry(QtCore.QRect(620, 230, 81, 20))
        self.txtRHibAng.setObjectName("txtRHibAng")
        self.txtRKneeDis = QtWidgets.QLineEdit(self.mainFrame)
        self.txtRKneeDis.setGeometry(QtCore.QRect(500, 270, 81, 20))
        self.txtRKneeDis.setObjectName("txtRKneeDis")
        self.label_14 = QtWidgets.QLabel(self.mainFrame)
        self.label_14.setGeometry(QtCore.QRect(430, 310, 51, 16))
        self.label_14.setObjectName("label_14")
        self.label_15 = QtWidgets.QLabel(self.mainFrame)
        self.label_15.setGeometry(QtCore.QRect(430, 230, 51, 16))
        self.label_15.setObjectName("label_15")
        self.txtWaisteAng = QtWidgets.QLineEdit(self.mainFrame)
        self.txtWaisteAng.setGeometry(QtCore.QRect(620, 190, 81, 20))
        self.txtWaisteAng.setObjectName("txtWaisteAng")
        self.txtRWristAng = QtWidgets.QLineEdit(self.mainFrame)
        self.txtRWristAng.setGeometry(QtCore.QRect(620, 150, 81, 20))
        self.txtRWristAng.setObjectName("txtRWristAng")
        self.txtRWristDis = QtWidgets.QLineEdit(self.mainFrame)
        self.txtRWristDis.setGeometry(QtCore.QRect(500, 150, 81, 20))
        self.txtRWristDis.setObjectName("txtRWristDis")
        self.label_16 = QtWidgets.QLabel(self.mainFrame)
        self.label_16.setGeometry(QtCore.QRect(430, 70, 51, 16))
        self.label_16.setObjectName("label_16")
        self.txtRKneeAng = QtWidgets.QLineEdit(self.mainFrame)
        self.txtRKneeAng.setGeometry(QtCore.QRect(620, 270, 81, 20))
        self.txtRKneeAng.setObjectName("txtRKneeAng")
        self.txtRElbowDis = QtWidgets.QLineEdit(self.mainFrame)
        self.txtRElbowDis.setGeometry(QtCore.QRect(500, 110, 81, 20))
        self.txtRElbowDis.setObjectName("txtRElbowDis")
        self.txtRFootAng = QtWidgets.QLineEdit(self.mainFrame)
        self.txtRFootAng.setGeometry(QtCore.QRect(620, 310, 81, 20))
        self.txtRFootAng.setObjectName("txtRFootAng")
        self.label_17 = QtWidgets.QLabel(self.mainFrame)
        self.label_17.setGeometry(QtCore.QRect(140, 40, 60, 16))
        self.label_17.setObjectName("label_17")
        self.label_18 = QtWidgets.QLabel(self.mainFrame)
        self.label_18.setGeometry(QtCore.QRect(260, 40, 51, 16))
        self.label_18.setObjectName("label_18")
        self.label_19 = QtWidgets.QLabel(self.mainFrame)
        self.label_19.setGeometry(QtCore.QRect(520, 40, 60, 16))
        self.label_19.setObjectName("label_19")
        self.label_20 = QtWidgets.QLabel(self.mainFrame)
        self.label_20.setGeometry(QtCore.QRect(640, 40, 51, 16))
        self.label_20.setObjectName("label_20")
        self.txtLeftHandDis = QtWidgets.QLineEdit(self.mainFrame)
        self.txtLeftHandDis.setGeometry(QtCore.QRect(130, 350, 81, 20))
        self.txtLeftHandDis.setObjectName("txtLeftHandDis")
        self.txtLeftLegDis = QtWidgets.QLineEdit(self.mainFrame)
        self.txtLeftLegDis.setGeometry(QtCore.QRect(130, 390, 81, 20))
        self.txtLeftLegDis.setObjectName("txtLeftLegDis")
        self.label_21 = QtWidgets.QLabel(self.mainFrame)
        self.label_21.setGeometry(QtCore.QRect(60, 390, 51, 16))
        self.label_21.setObjectName("label_21")
        self.txtRightLegAng = QtWidgets.QLineEdit(self.mainFrame)
        self.txtRightLegAng.setGeometry(QtCore.QRect(620, 390, 81, 20))
        self.txtRightLegAng.setObjectName("txtRightLegAng")
        self.txtRightLegDis = QtWidgets.QLineEdit(self.mainFrame)
        self.txtRightLegDis.setGeometry(QtCore.QRect(500, 390, 81, 20))
        self.txtRightLegDis.setObjectName("txtRightLegDis")
        self.txtRightHandDis = QtWidgets.QLineEdit(self.mainFrame)
        self.txtRightHandDis.setGeometry(QtCore.QRect(500, 350, 81, 20))
        self.txtRightHandDis.setObjectName("txtRightHandDis")
        self.txtLeftLegAng = QtWidgets.QLineEdit(self.mainFrame)
        self.txtLeftLegAng.setGeometry(QtCore.QRect(250, 390, 81, 20))
        self.txtLeftLegAng.setObjectName("txtLeftLegAng")
        self.label_22 = QtWidgets.QLabel(self.mainFrame)
        self.label_22.setGeometry(QtCore.QRect(60, 350, 51, 16))
        self.label_22.setObjectName("label_22")
        self.txtRightHandAng = QtWidgets.QLineEdit(self.mainFrame)
        self.txtRightHandAng.setGeometry(QtCore.QRect(620, 350, 81, 20))
        self.txtRightHandAng.setObjectName("txtRightHandAng")
        self.txtLeftHandAng = QtWidgets.QLineEdit(self.mainFrame)
        self.txtLeftHandAng.setGeometry(QtCore.QRect(250, 350, 81, 20))
        self.txtLeftHandAng.setObjectName("txtLeftHandAng")
        self.label_23 = QtWidgets.QLabel(self.mainFrame)
        self.label_23.setGeometry(QtCore.QRect(430, 350, 51, 16))
        self.label_23.setObjectName("label_23")
        self.label_24 = QtWidgets.QLabel(self.mainFrame)
        self.label_24.setGeometry(QtCore.QRect(430, 390, 51, 16))
        self.label_24.setObjectName("label_24")

        self.label_25 = QtWidgets.QLabel(self.mainFrame)
        self.label_25.setGeometry(QtCore.QRect(270, 430, 51, 16))
        self.label_25.setObjectName("label_25")
        self.comboAction = QtWidgets.QComboBox(self.mainFrame)
        self.comboAction.setGeometry(QtCore.QRect(320, 430, 150, 25))
        self.comboAction.setObjectName("comboAction")
        self.comboAction.addItems(["Select","Out","One","Two","Three","Four","Five","Six","Wide","No Ball"])
 
        self.btnProcess = QtWidgets.QPushButton(self.centralwidget)
        self.btnProcess.setGeometry(QtCore.QRect(200, 590, 171, 31))
        self.btnProcess.setObjectName("btnProcess")
        self.btnTrain = QtWidgets.QPushButton(self.centralwidget)
        self.btnTrain.setGeometry(QtCore.QRect(400, 590, 171, 31))
        self.btnTrain.setObjectName("btnTrain")
        self.btnPredict = QtWidgets.QPushButton(self.centralwidget)
        self.btnPredict.setGeometry(QtCore.QRect(600, 590, 171, 31))
        self.btnPredict.setObjectName("btnPredict")


        self.scoreFrame = QtWidgets.QFrame(self.centralwidget)
        self.scoreFrame.setGeometry(QtCore.QRect(920, 100, 261, 201))
        self.scoreFrame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.scoreFrame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.scoreFrame.setObjectName("scoreFrame")

        self.label_26 = QtWidgets.QLabel(self.scoreFrame)
        self.label_26.setGeometry(QtCore.QRect(80, 40, 51, 16))
        self.label_26.setObjectName("label_26")
        self.label_26.setText("Runs:")

        self.txtRuns = QtWidgets.QLabel(self.scoreFrame)
        self.txtRuns.setGeometry(QtCore.QRect(180, 40, 51, 16))
        self.txtRuns.setObjectName("txtRuns")
        self.txtRuns.setText("0")

        self.label_27 = QtWidgets.QLabel(self.scoreFrame)
        self.label_27.setGeometry(QtCore.QRect(80, 80, 51, 16))
        self.label_27.setObjectName("label_27")
        self.label_27.setText("Wickets:")

        self.txtWicket = QtWidgets.QLabel(self.scoreFrame)
        self.txtWicket.setGeometry(QtCore.QRect(180, 80, 51, 16))
        self.txtWicket.setObjectName("txtWicket")
        self.txtWicket.setText("0")

        self.label_28 = QtWidgets.QLabel(self.scoreFrame)
        self.label_28.setGeometry(QtCore.QRect(80, 120, 51, 16))
        self.label_28.setObjectName("label_28")
        self.label_28.setText("Extras:")

        self.txtExtra = QtWidgets.QLabel(self.scoreFrame)
        self.txtExtra.setGeometry(QtCore.QRect(180, 120, 51, 16))
        self.txtExtra.setObjectName("txtExtra")
        self.txtExtra.setText("0")

        self.label_29 = QtWidgets.QLabel(self.scoreFrame)
        self.label_29.setGeometry(QtCore.QRect(80, 160, 70, 16))
        self.label_29.setObjectName("label_29")
        self.label_29.setText("Total Runs:")

        self.txtTotal = QtWidgets.QLabel(self.scoreFrame)
        self.txtTotal.setGeometry(QtCore.QRect(180, 160, 51, 16))
        self.txtTotal.setObjectName("txtTotal")
        self.txtTotal.setText("0")

        self.label_100 = QtWidgets.QLabel(self.centralwidget)
        self.label_100.setGeometry(QtCore.QRect(1000, 55, 140, 80))
        self.label_100.setText("")

        SkeletonWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(SkeletonWindow)
        QtCore.QMetaObject.connectSlotsByName(SkeletonWindow)
        self.btnBrowse.clicked.connect(self.clickBrowse)
        self.btnProcess.clicked.connect(self.clickProcess)
        self.btnTrain.clicked.connect(self.clickTrain)
        self.btnPredict.clicked.connect(self.clickPredict)
    
    def clickPredict(self):
        balance_data = pd.read_csv('score.csv',sep= ',', header= None)
        X = balance_data.values[1:, 0:36]
        Y = balance_data.values[1:,36]
        print(X)
        X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.2, random_state = 100)
        clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)

        y_pred = clf.fit(X_train, y_train).predict(X_test)
        
        cm= confusion_matrix(y_test,y_pred)
        print("Confusion matrix")
        print(cm)
        print("Precision score")
        print(precision_score(y_test, y_pred,average='macro',labels=np.unique(y_pred)))
        print("Recall score")
        print(recall_score(y_test, y_pred,average='micro'))
        print("F1 score")
        print(f1_score(y_test, y_pred,average='weighted', labels=np.unique(y_pred)))
        print("Accuracy score")
        print(accuracy_score(y_test, y_pred, normalize=False))
        clf.fit(X,Y)
        #print(clf.predict([[152.0691,339.5988,101.5529,348.692,94.59387,1.244541,153.0621,345.5263,155.467,345.5263,134.2013,348.0254,0,0,174.0402,343.081,174.0402,343.081,160.7047,7.379833,217,350.0357,217,349.4468,216.7487,354.8121,216,353.1157,281.6416,333.5517,0,0,433.374,344.8478,433,342.5625],[0,0,68.11755,354.1301,90.42677,359.1215,115.382,348.9946,128.9496,348.9946,0,0,131.746,353.7728,121.0496,351.8152,0,0,0,0,0,0,0,0,154.9839,354.1987,154.9839,349.8391,0,0,153.8343,347.5075,0,0,0,0],[38.32754,346.9118,24.35159,353.0681,26.62705,359.3819,37.20215,346.2912,38.01316,346.2912,0,0,40.22437,350.0591,53.33854,345.4124,0,0,0,0,0,0,0,0,0,0,0,0,0,0,55.10898,346.1496,0,0,0,0]]))
        score=clf.predict([[float(self.txtHeadDis.text()),float(self.txtHeadAng.text()),float(self.txtLSDis.text()),float(self.txtLSAng.text()),float(self.txtRSDis.text()),float(self.txtRSAng.text()), float(self.txtLElbowDis.text()), float(self.txtLElbowAng.text()), float(self.txtRElbowDis.text()), float(self.txtLElbowAng.text()),float(self.txtLWristDis.text()), float(self.txtLWristAng.text()),float(self.txtRWristDis.text()), float(self.txtRWristAng.text()), float(self.txtWaisteDis.text()),float(self.txtWaisteAng.text()), float(self.txtLHibDis.text()), float(self.txtLHibAng.text()),float(self.txtRHibDis.text()), float(self.txtRHibAng.text()),float(self.txtLKneeDis.text()), float(self.txtLKneeAng.text()),float(self.txtRKneeDis.text()),float(self.txtRKneeAng.text()),float(self.txtLFootDis.text()),float(self.txtLFootAng.text()), float(self.txtRFootDis.text()), float(self.txtRFootAng.text()),float(self.txtLeftHandDis.text()), float(self.txtLeftHandAng.text()), float(self.txtRightHandDis.text()), float(self.txtRightHandAng.text()),float(self.txtLeftLegDis.text()), float(self.txtLeftLegAng.text()), float(self.txtRightLegDis.text()), float(self.txtRightLegAng.text())]])
        print(score[0])
        
        if float(score[0])==0.0:
            self.txtWicket.setText(str(int(self.txtWicket.text())+1))
        action=""
        if float(score[0])==1.0:
            self.txtRuns.setText(str(int(self.txtRuns.text())+1))
        if float(score[0])==2.0:
            self.txtRuns.setText(str(int(self.txtRuns.text())+2))
        if float(score[0])==3.0:
            self.txtRuns.setText(str(int(self.txtRuns.text())+3))
        if float(score[0])==4.0:
            self.txtRuns.setText(str(int(self.txtRuns.text())+4))
        if float(score[0])==5.0:
            self.txtRuns.setText(str(int(self.txtRuns.text())+5))
        if float(score[0])==6.0:
            self.txtRuns.setText(str(int(self.txtRuns.text())+6))
        if float(score[0])==7.0 or float(score[0])==8.0:
            self.txtExtra.setText(str(int(self.txtExtra.text())+1))
        
        if float(score[0])==0.0:
            action="         OUT!"
        
        if float(score[0])==1.0:
            action="           ONE"
        
        if float(score[0])==2.0:
            action="          TWO"
        
        if float(score[0])==3.0:
            action="         THREE"
        
        if float(score[0])==4.0:
            action="         Four"

        if float(score[0])==6.0:
            action="         SIX !"

        if float(score[0])==7.0:
            action="         WIDE"
        if float(score[0])==8.0:
            action="         NO BALL"
        self.label_100.setText(action)
        self.txtTotal.setText(str(int(self.txtExtra.text())+int(self.txtRuns.text())))
    def clickTrain(self):
        if self.comboAction.currentText()!="Select":
            self.action=0.0
            if self.comboAction.currentText()=="Out":
                self.action=0.0
            if self.comboAction.currentText()=="One":
                self.action=1.0
            if self.comboAction.currentText()=="Two":
                self.action=2.0
            if self.comboAction.currentText()=="Three":
                self.action=3.0
            if self.comboAction.currentText()=="Four":
                self.action=4.0
            if self.comboAction.currentText()=="Five":
                self.action=5.0
            if self.comboAction.currentText()=="Six":
                self.action=6.0 
            if self.comboAction.currentText()=="Wide":
                self.action=7.0
            if self.comboAction.currentText()=="No Ball":
                self.action=8.0     
            print(self.comboAction.currentText())
          
            fields = ['head_dis', 'head_angle', 'ls_dis', 'ls_ang', 'rs_dis', 'rs_ang', 'lelbow_dis', 'lelbow_ang', 'relbow_dis', 'relbow_ang', 'lwrist_dis', 'lwrist_ang', 'rwrist_dis', 'rwrist_ang', 'waiste_dis', 'waiste_ang', 'lhip_dis', 'lhip_ang', 'rhip_dis', 'rhip_ang', 'lknee_dis', 'lknee_ang', 'rknee_dis', 'rknee_ang', 'lfoot_dis', 'lfoot_ang', 'rfoot_dis', 'rfoot_ang','lhand_dis', 'lhand_ang', 'rhand_dis', 'rhand_ang','lleg_dis', 'lleg_ang', 'rleg_dis', 'rleg_ang', 'action']
            with open('score.csv', 'a') as csvfile:
                action_res=[{'head_dis':self.txtHeadDis.text(), 'head_angle':self.txtHeadAng.text(),
                 'ls_dis':self.txtLSDis.text(), 'ls_ang':self.txtLSAng.text(), 'rs_dis':self.txtRSDis.text(), 'rs_ang':self.txtRSAng.text(), 'lelbow_dis':self.txtLElbowDis.text(), 'lelbow_ang':self.txtLElbowAng.text(), 'relbow_dis':self.txtRElbowDis.text(), 'relbow_ang':self.txtLElbowAng.text(),'lwrist_dis':self.txtLWristDis.text(), 'lwrist_ang':self.txtLWristAng.text(), 'rwrist_dis':self.txtRWristDis.text(), 'rwrist_ang':self.txtRWristAng.text(), 'waiste_dis':self.txtWaisteDis.text(), 'waiste_ang':self.txtWaisteAng.text(), 'lhip_dis':self.txtLHibDis.text(), 'lhip_ang':self.txtLHibAng.text(), 'rhip_dis':self.txtRHibDis.text(), 'rhip_ang':self.txtRHibAng.text(), 'lknee_dis':self.txtLKneeDis.text(), 'lknee_ang':self.txtLKneeAng.text(),
                  'rknee_dis':self.txtRKneeDis.text(), 'rknee_ang':self.txtRKneeAng.text(), 'lfoot_dis':self.txtLFootDis.text(), 'lfoot_ang':self.txtLFootAng.text(), 'rfoot_dis':self.txtRFootDis.text(), 'rfoot_ang':self.txtRFootAng.text(),'lhand_dis':self.txtLeftHandDis.text(), 'lhand_ang':self.txtLeftHandAng.text(), 'rhand_dis':self.txtRightHandDis.text(), 'rhand_ang':self.txtRightHandAng.text(),'lleg_dis':self.txtLeftLegDis.text(), 'lleg_ang':self.txtLeftLegAng.text(), 'rleg_dis':self.txtRightLegDis.text(), 'rleg_ang':self.txtRightLegAng.text(), 'action':self.action}]
                writer = csv.DictWriter(csvfile, fieldnames = fields) 
                # writing headers (field names) 
                #writer.writeheader() 
      
                # writing data rows 
                writer.writerows(action_res)

        else:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Warning)
            msg.setText("Field can't be Empty")
            msg.setWindowTitle("Error")
            msg.exec_()

    def clickProcess(self):
        if (self.txtImage.text()!=""):
            self.progress.setAutoReset(False)
            self.progress.setAutoClose(False)
            self.progress.setMinimum(0)
            self.progress.setMaximum(100)
            self.progress.resize(400,100)
            self.progress.setWindowTitle("Processing, Please Wait!")
            self.progress.show()
            self.progress.setValue(5)
            frame = cv2.imread(self.txtImage.text())
            frameHeight = np.size(frame, 0)
            frameWidth = np.size(frame, 1)
             #specify the input dimensions

            inWidth = 368
            inHeight = 368
            #prepare the frame to be fed to the network

            inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
            
            self.progress.setValue(25)
            #set the prepared object to the input blob of the network

            net.setInput(inpBlob)
             #For prediction

            output = net.forward()
            print(output)
            img = np.zeros((frameHeight,frameWidth,3), np.uint8)
            H = output.shape[2]
            W = output.shape[3]
            points = [] # empty list to store the detected keypoints
            self.progress.setValue(30)
            for i in range(0,15):
                probMap = output[0, i, :, :] # confidence map of corresponding body's part
                minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)#Find global maxima of the probMap
                '''frameWidth=866
                frameHeight=1392'''
                 # scale the point to fit the original image

                x = (frameWidth * point[0]) / W
                y = (frameHeight * point[1]) / H
 
                if prob > 0.5 : 
                    cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                    frameCopy=frame.copy()
                    cv2.putText(frame, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, lineType=cv2.LINE_AA)
                    points.append((int(x), int(y)))#add the point to the list if the probability is greater than threshold


                else :
                    points.append(None)
            print(points)
            self.progress.setValue(50)
            cv2.imshow("OutputPoints",frame)
            if points[0]!=None and points[1]!=None:
                cv2.line(img, points[0], points[1], (0,255,0), 2)
                dis= math.sqrt( (points[0][0] - points[1][0])**2 + (points[0][1] - points[1][1])**2 )
                ang=angle(points[0],points[1])
                print("Angle {} Distance:{}".format(ang,dis))
                self.txtHeadDis.setText('{0:.7g}'.format(dis))
                self.txtHeadAng.setText('{0:.7g}'.format(ang))
            else:
                self.txtHeadDis.setText("0")
                self.txtHeadAng.setText("0")
            
            #Slder
            if points[1]!=None and points[2]!=None:
                cv2.line(img, points[1], points[2], (255,0,0), 2)
                dis= math.sqrt( (points[1][0] - points[2][0])**2 + (points[1][1] - points[2][1])**2 )
                ang=angle(points[1],points[2])
                print("Angle {} Distance:{}".format(ang,dis))
                self.txtLSDis.setText('{0:.7g}'.format(dis))
                self.txtLSAng.setText('{0:.7g}'.format(ang))
            else:
                self.txtLSDis.setText("0")
                self.txtLSAng.setText("0")


            if points[1]!=None and points[5]!=None:
                cv2.line(img, points[1], points[5], (255,0,0), 2)
                dis= math.sqrt( (points[1][0] - points[5][0])**2 + (points[1][1] - points[5][1])**2 )
                ang=angle(points[1],points[5])
                print("Angle {} Distance:{}".format(ang,dis))
                self.txtRSDis.setText('{0:.7g}'.format(dis))
                self.txtRSAng.setText('{0:.7g}'.format(ang))
            else:
                self.txtRSDis.setText("0")
                self.txtRSAng.setText("0")
            #left Hand
            if points[3]!=None and points[2]!=None:
                cv2.line(img, points[2], points[3], (156,105,100), 2)
                dis= math.sqrt( (points[2][0] - points[3][0])**2 + (points[2][1] - points[3][1])**2 )
                ang=angle(points[2],points[3])
                print("Angle {} Distance:{}".format(ang,dis))
                self.txtLElbowDis.setText('{0:.7g}'.format(dis))
                self.txtLElbowAng.setText('{0:.7g}'.format(ang))
            else:
                self.txtLElbowDis.setText("0")
                self.txtLElbowAng.setText("0")
            if points[3]!=None and points[4]!=None:
                cv2.line(img, points[3], points[4], (156,105,100), 2)
                dis= math.sqrt( (points[3][0] - points[4][0])**2 + (points[3][1] - points[4][1])**2 )
                ang=angle(points[3],points[4])
                print("Angle {} Distance:{}".format(ang,dis))
                self.txtLWristDis.setText('{0:.7g}'.format(dis))
                self.txtLWristAng.setText('{0:.7g}'.format(ang))
            else:
                self.txtLWristDis.setText("0")
                self.txtLWristAng.setText("0")
            #right Hand
            if points[6]!=None and points[5]!=None:
                cv2.line(img, points[5], points[6], (156,105,100), 2)
                dis= math.sqrt( (points[5][0] - points[6][0])**2 + (points[5][1] - points[6][1])**2 )
                ang=angle(points[5],points[6])
                print("Angle {} Distance:{}".format(ang,dis))
                self.txtRElbowDis.setText('{0:.7g}'.format(dis))
                self.txtRElbowAng.setText('{0:.7g}'.format(ang))
            else:
                self.txtRElbowDis.setText("0")
                self.txtRElbowAng.setText("0")


            if points[6]!=None and points[7]!=None:
                cv2.line(img, points[6], points[7], (156,105,100), 2)
                dis= math.sqrt( (points[6][0] - points[7][0])**2 + (points[6][1] - points[7][1])**2 )
                ang=angle(points[6],points[7])
                print("Angle {} Distance:{}".format(ang,dis))
                self.txtRWristDis.setText('{0:.7g}'.format(dis))
                self.txtRWristAng.setText('{0:.7g}'.format(ang))
            else:
                self.txtRWristDis.setText("0")
                self.txtRWristAng.setText("0")

            #Waiste
            if points[14]!=None and points[1]!=None:
                cv2.line(img, points[1], points[14], (0,255,0), 2)
                dis= math.sqrt( (points[1][0] - points[14][0])**2 + (points[1][1] - points[14][1])**2 )
                ang=angle(points[1],points[14])
                print("Angle {} Distance:{}".format(ang,dis))
                self.txtWaisteDis.setText('{0:.7g}'.format(dis))
                self.txtWaisteAng.setText('{0:.7g}'.format(ang))
            else:
                self.txtWaisteDis.setText("0")
                self.txtWaisteAng.setText("0")

            #hip
            if points[8]!=None and points[14]!=None:
                cv2.line(img, points[14], points[8], (190,55,70), 2)
                dis= math.sqrt( (points[1][0] - points[14][0])**2 + (points[1][1] - points[14][1])**2 )
                ang=angle(points[1],points[14])
                print("Angle {} Distance:{}".format(ang,dis))
                self.txtLHibDis.setText('{0:.7g}'.format(dis))
                self.txtLHibAng.setText('{0:.7g}'.format(ang))
            else:
                self.txtLHibDis.setText("0")
                self.txtLHibAng.setText("0")

            if points[11]!=None and points[14]!=None:
                cv2.line(img, points[14], points[11], (190,55,70), 2)
                dis= math.sqrt( (points[11][0] - points[14][0])**2 + (points[11][1] - points[14][1])**2 )
                ang=angle(points[11],points[14])
                print("Angle {} Distance:{}".format(ang,dis))
                self.txtRHibDis.setText('{0:.7g}'.format(dis))
                self.txtRHibAng.setText('{0:.7g}'.format(ang))
            else:
                self.txtRHibDis.setText("0")
                self.txtRHibAng.setText("0")

            #leftleg
            if points[8]!=None and points[9]!=None:
                cv2.line(img, points[8], points[9], (90,155,70), 2)
                dis= math.sqrt( (points[8][0] - points[9][0])**2 + (points[8][1] - points[9][1])**2 )
                ang=angle(points[8],points[9])
                print("Angle {} Distance:{}".format(ang,dis))
                self.txtLKneeDis.setText('{0:.7g}'.format(dis))
                self.txtLKneeAng.setText('{0:.7g}'.format(ang))
            else:
                self.txtLKneeDis.setText("0")
                self.txtLKneeAng.setText("0")
            if points[9]!=None and points[10]!=None:
                cv2.line(img, points[9], points[10], (90,155,70), 2)
                dis= math.sqrt( (points[9][0] - points[10][0])**2 + (points[9][1] - points[10][1])**2 )
                ang=angle(points[9],points[10])
                print("Angle {} Distance:{}".format(ang,dis))
                self.txtLFootDis.setText('{0:.7g}'.format(dis))
                self.txtLFootAng.setText('{0:.7g}'.format(ang))
            else:
                self.txtLFootDis.setText("0")
                self.txtLFootAng.setText("0")
            #rightleg
            if points[11]!=None and points[12]!=None:
                cv2.line(img, points[11], points[12], (90,155,70), 2)
                dis= math.sqrt( (points[11][0] - points[12][0])**2 + (points[11][1] - points[12][1])**2 )
                ang=angle(points[11],points[12])
                print("Angle {} Distance:{}".format(ang,dis))
                self.txtRKneeDis.setText('{0:.7g}'.format(dis))
                self.txtRKneeAng.setText('{0:.7g}'.format(ang))
            else:
                self.txtRKneeDis.setText("0")
                self.txtRKneeAng.setText("0")

            if points[12]!=None and points[13]!=None:
                cv2.line(img, points[12], points[13], (90,155,70), 2)
                dis= math.sqrt( (points[12][0] - points[13][0])**2 + (points[12][1] - points[13][1])**2 )
                ang=angle(points[12],points[13])
                print("Angle {} Distance:{}".format(ang,dis))
                self.txtRFootDis.setText('{0:.7g}'.format(dis))
                self.txtRFootAng.setText('{0:.7g}'.format(ang))
            else:
                self.txtRFootDis.setText("0")
                self.txtRFootAng.setText("0")

            #righthand
            if points[5]!=None and points[7]!=None:
                
                dis= math.sqrt( (points[5][0] - points[7][0])**2 + (points[5][1] - points[7][1])**2 )
                ang=angle(points[5],points[7])
                print("Angle {} Distance:{}".format(ang,dis))
                self.txtRightHandDis.setText('{0:.7g}'.format(dis))
                self.txtRightHandAng.setText('{0:.7g}'.format(ang))
            else:
                self.txtRightHandDis.setText("0")
                self.txtRightHandAng.setText("0")
            
            #lefthand
            if points[2]!=None and points[4]!=None:
                
                dis= math.sqrt( (points[2][0] - points[4][0])**2 + (points[2][1] - points[4][1])**2 )
                ang=angle(points[2],points[4])
                print("Angle {} Distance:{}".format(ang,dis))
                self.txtLeftHandDis.setText('{0:.7g}'.format(dis))
                self.txtLeftHandAng.setText('{0:.7g}'.format(ang))
            else:
                self.txtLeftHandDis.setText("0")
                self.txtLeftHandAng.setText("0")


            #rightleg
            if points[11]!=None and points[13]!=None:
                
                dis= math.sqrt( (points[11][0] - points[13][0])**2 + (points[11][1] - points[13][1])**2 )
                ang=angle(points[11],points[13])
                print("Angle {} Distance:{}".format(ang,dis))
                self.txtRightLegDis.setText('{0:.7g}'.format(dis))
                self.txtRightLegAng.setText('{0:.7g}'.format(ang))
            else:
                self.txtRightLegDis.setText("0")
                self.txtRightLegAng.setText("0")
            
            #leftleg
            if points[8]!=None and points[10]!=None:
                
                dis= math.sqrt( (points[8][0] - points[10][0])**2 + (points[8][1] - points[10][1])**2 )
                ang=angle(points[8],points[10])
                print("Angle {} Distance:{}".format(ang,dis))
                self.txtLeftLegDis.setText('{0:.7g}'.format(dis))
                self.txtLeftLegAng.setText('{0:.7g}'.format(ang))
            else:
                self.txtLeftLegDis.setText("0")
                self.txtLeftLegAng.setText("0")

            self.progress.setValue(75)
            cv2.imshow("Skeleton",img)
            cv2.waitKey(0)
            self.progress.setValue(100)
            self.progress.hide()
        else:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Warning)
            msg.setText("Browse Image")
            msg.setWindowTitle("Image")
            msg.exec_()
    def clickBrowse(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None,"Browse Image", "","Image files (*.jpg *.png)")
        self.txtImage.setText(fileName)

    def retranslateUi(self, SkeletonWindow):
        _translate = QtCore.QCoreApplication.translate
        SkeletonWindow.setWindowTitle(_translate("SkeletonWindow", "MainWindow"))
        self.label.setText(_translate("SkeletonWindow", " SCORECARD "))
        self.btnBrowse.setText(_translate("SkeletonWindow", "Browse"))
        self.lblImage.setText(_translate("SkeletonWindow", "Image"))
        self.label_2.setText(_translate("SkeletonWindow", "Points"))
        self.label_3.setText(_translate("SkeletonWindow", "P(0-1)"))
        self.label_4.setText(_translate("SkeletonWindow", "P(1-2)"))
        self.label_5.setText(_translate("SkeletonWindow", "P(1-3)"))
        self.label_6.setText(_translate("SkeletonWindow", "P(1-4)"))
        self.label_7.setText(_translate("SkeletonWindow", "P(14-8)"))
        self.label_8.setText(_translate("SkeletonWindow", "P(8-9)"))
        self.label_9.setText(_translate("SkeletonWindow", "P(9-10)"))
        self.label_10.setText(_translate("SkeletonWindow", "P(1-14)"))
        self.label_11.setText(_translate("SkeletonWindow", "P(11-12)"))
        self.label_12.setText(_translate("SkeletonWindow", "P(6-7)"))
        self.label_13.setText(_translate("SkeletonWindow", "P(5-6)"))
        self.label_14.setText(_translate("SkeletonWindow", "P(12-13)"))
        self.label_15.setText(_translate("SkeletonWindow", "P(14-11)"))
        self.label_16.setText(_translate("SkeletonWindow", "P(1-5)"))
        self.label_17.setText(_translate("SkeletonWindow", "Distance"))
        self.label_18.setText(_translate("SkeletonWindow", "Angle"))
        self.label_19.setText(_translate("SkeletonWindow", "Distance"))
        self.label_20.setText(_translate("SkeletonWindow", "Angle"))
        self.label_21.setText(_translate("SkeletonWindow", "P(8-10)"))
        self.label_22.setText(_translate("SkeletonWindow", "P(2-4)"))
        self.label_23.setText(_translate("SkeletonWindow", "P(5-7)"))
        self.label_24.setText(_translate("SkeletonWindow", "P(11-13)"))
        self.label_25.setText(_translate("SkeletonWindow", "Action:"))
        self.btnProcess.setText(_translate("SkeletonWindow", "Process"))
        self.btnPredict.setText(_translate("SkeletonWindow", "Predict"))
        self.btnTrain.setText(_translate("SkeletonWindow", "Train Image"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    SkeletonWindow = QtWidgets.QMainWindow()
    ui = Ui_SkeletonWindow()
    ui.setupUi(SkeletonWindow)
    SkeletonWindow.show()
    sys.exit(app.exec_())

