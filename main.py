from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog, QListWidgetItem, QDialog, QMessageBox
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtGui import QPalette
from design import Ui_Form
from media import CMultiMedia
import sys
import datetime
from os.path import basename
import os
import subprocess
from deid_dialog import DeidDialog
 
QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)

class VideoItem(QListWidgetItem):
    def __init__(self, name, path):
        super().__init__(name)
        self.path = path
        self.name = name
 
class CWidget(QWidget):
    def __init__(self):
        super().__init__()
        
        # UI setup using the design.py
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        # Multimedia Object
        self.mp = CMultiMedia(self, self.ui.view)
 
        # Video background color
        pal = QPalette()        
        pal.setColor(QPalette.Background, Qt.black)
        self.ui.view.setAutoFillBackground(True)
        self.ui.view.setPalette(pal)
         
        # Volume, slider
        self.ui.vol.setRange(0,100)
        self.ui.vol.setValue(50)
 
        # Play time
        self.duration = ''
 
        # Signal
        self.ui.btn_add.clicked.connect(self.clickAdd)
        self.ui.btn_del.clicked.connect(self.clickDel)
        self.ui.btn_play.clicked.connect(self.clickPlay)
        self.ui.btn_stop.clicked.connect(self.clickStop)
        self.ui.btn_pause.clicked.connect(self.clickPause)
        self.ui.btn_forward.clicked.connect(self.clickForward)
        self.ui.btn_prev.clicked.connect(self.clickPrev)
        self.ui.btn_deidentification.clicked.connect(self.run_deidentification_script)

        self.ui.list.itemDoubleClicked.connect(self.dbClickList)
        self.ui.vol.valueChanged.connect(self.volumeChanged)
        self.ui.bar.sliderMoved.connect(self.barChanged)       

    def clickAdd(self):
        files, ext = QFileDialog.getOpenFileNames(self
                                             , 'Select one or more files to open'
                                             , ''
                                             , 'Video (*.mp4 *.mpg *.mpeg *.avi *.wma)') 
         
        if files:
            cnt = len(files)       
            for file_path in files:
                filename = os.path.splitext(os.path.basename(file_path))[0]
                
                # VideoItem 클래스를 사용하여 아이템 추가
                video_item = VideoItem(filename, file_path)
                self.ui.list.addItem(video_item)
            self.ui.list.setCurrentRow(0)
            self.mp.addMedia(files)
 
    def run_deidentification_script(self):
        selected_items = self.ui.list.selectedItems()
        if not selected_items:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please select a video from the playlist!")
            return

        selected_video = selected_items[0]
        selected_video_path = selected_video.path
        video_name = os.path.splitext(os.path.basename(selected_video_path))[0]  # 파일 이름만 추출

        
        # 클러스터링 스크립트 실행
        result = subprocess.run(["python", "clustering.py", selected_video_path])
        

        
        
        if result.returncode == 0:            
            image_folder = os.path.join("./", "output_folders", video_name, "faces", "suggestion_faces")
            image_files = [os.path.join(image_folder, filename) for filename in os.listdir(image_folder) if filename.endswith('.jpg')]
            dialog = DeidDialog(video_name, image_files, selected_video_path, self)
            if dialog.exec_() == QDialog.Accepted:
                selected_images = dialog.accepted_images
                print("Selected images:", selected_images)


    def clickDel(self):
        row = self.ui.list.currentRow()
        self.ui.list.takeItem(row)
        self.mp.delMedia(row)
 
    def clickPlay(self):
        index = self.ui.list.currentRow()        
        self.mp.playMedia(index)
 
    def clickStop(self):
        self.mp.stopMedia()
 
    def clickPause(self):
        self.mp.pauseMedia()
 
    def clickForward(self):
        cnt = self.ui.list.count()
        curr = self.ui.list.currentRow()
        if curr<cnt-1:
            self.ui.list.setCurrentRow(curr+1)
            self.mp.forwardMedia()
        else:
            self.ui.list.setCurrentRow(0)
            self.mp.forwardMedia(end=True)
 
    def clickPrev(self):
        cnt = self.ui.list.count()
        curr = self.ui.list.currentRow()
        if curr==0:
            self.ui.list.setCurrentRow(cnt-1)    
            self.mp.prevMedia(begin=True)
        else:
            self.ui.list.setCurrentRow(curr-1)    
            self.mp.prevMedia()
 
    def dbClickList(self, item):
        row = self.ui.list.row(item)
        self.mp.playMedia(row)
 
    def volumeChanged(self, vol):
        self.mp.volumeMedia(vol)
 
    def barChanged(self, pos):   
        print(pos)
        self.mp.posMoveMedia(pos)    
 
    def updateState(self, msg):
        self.ui.state.setText(msg)
 
    def updateBar(self, duration):
        self.ui.bar.setRange(0,duration)    
        self.ui.bar.setSingleStep(int(duration/10))
        self.ui.bar.setPageStep(int(duration/10))
        self.ui.bar.setTickInterval(int(duration/10))
        td = datetime.timedelta(milliseconds=duration)        
        stime = str(td)
        idx = stime.rfind('.')
        self.duration = stime[:idx]
 
    def updatePos(self, pos):
        self.ui.bar.setValue(pos)
        td = datetime.timedelta(milliseconds=pos)
        stime = str(td)
        idx = stime.rfind('.')
        stime = f'{stime[:idx]} / {self.duration}'
        self.ui.playtime.setText(stime)
 
 
if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = CWidget()
    w.show()
    sys.exit(app.exec_())