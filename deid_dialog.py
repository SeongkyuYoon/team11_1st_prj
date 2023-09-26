# deid_dialog.py
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QGridLayout, 
                             QLabel, QCheckBox, QPushButton, QHBoxLayout, QScrollArea, QWidget, QMessageBox)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QProcess
import pandas as pd
import os
import sys


class DeidDialog(QDialog):
    def __init__(self, video_name, image_files, selected_video_path, parent=None):        
        super(DeidDialog, self).__init__(parent)

        self.process = QProcess()   # QProcess인스턴스 생성
        
        self.accepted_images = []
        self.video_name = video_name # video_name 을 인스턴스 변수로 저장
        self.selected_video_path = selected_video_path
        self.image_files = image_files
        self.layout = QVBoxLayout(self)
        self.setWindowTitle("Select Images for De-Identification")
        
        self.resize(500, 500)

        # Scroll Area for images
        self.scroll = QScrollArea(self)
        self.scroll_widget = QWidget()
        self.grid_layout = QGridLayout(self.scroll_widget)

        self.image_widgets = []

        for idx, img_path in enumerate(image_files):
            pixmap = QPixmap(img_path)
            pixmap = pixmap.scaled(500, 500, Qt.KeepAspectRatio)
            image_label = QLabel()
            image_label.setPixmap(pixmap)
            checkbox = QCheckBox()
            
            self.grid_layout.addWidget(checkbox, idx, 0)
            self.grid_layout.addWidget(image_label, idx, 1)

            self.image_widgets.append((checkbox, img_path))
        
        self.scroll_widget.setLayout(self.grid_layout)
        self.scroll.setWidget(self.scroll_widget)
        self.layout.addWidget(self.scroll)
        
        # Confirm and Cancel buttons
        self.button_layout = QHBoxLayout()

        self.confirm_button = QPushButton("Confirm", self)
        self.confirm_button.clicked.connect(self.on_confirm)
        self.button_layout.addWidget(self.confirm_button)

        self.mosaic_button = QPushButton("De-id", self)
        self.mosaic_button.clicked.connect(self.start_mosaic)
        self.button_layout.addWidget(self.mosaic_button)
        
        self.cancel_button = QPushButton("Cancel", self)
        self.cancel_button.clicked.connect(self.reject)
        self.button_layout.addWidget(self.cancel_button)
        
        self.layout.addLayout(self.button_layout)
        
        self.setLayout(self.layout)

    def on_confirm(self):
        selected_images = []
        for checkbox, img_path in self.image_widgets:
            if checkbox.isChecked():
                selected_images.append(img_path)
        
        self.accepted_images = selected_images

        # 선택한 이미지 파일들의 이름을 텍스트 파일에 저장
        text_file_path = os.path.join("./","output_folders", self.video_name, "faces", "suggestion_faces","selected_images.txt")
        with open(text_file_path, "w") as f:
            for img_path in selected_images:
                img_name = os.path.basename(img_path)
                f.write(img_name + "\n")

        QMessageBox.information(self, "info", "인물이 선택 되었습니다",
                                QMessageBox.Ok)
        

    def start_mosaic(self):
        # 비식별화가 진행 중임을 알리는 알림 대화 상자 표시
        self.progress_dialog = QMessageBox(self)
        self.progress_dialog.setText("비식별화가 진행 중입니다.")
        self.progress_dialog.show()

       

        # exporting.py 파일 실행
        script_path = os.path.join(".", "exporting.py")
        self.process.start("python", [script_path, self.selected_video_path])

        # exporting.py 프로세스의 상태를 모니터링하기 위해 시그널 연결
        self.process.finished.connect(self.on_process_finished)
        self.process.errorOccurred.connect(self.on_process_error)

    @pyqtSlot(int, QProcess.ExitStatus)
    def on_process_finished(self, exitCode, exitStatus):
        if exitStatus == QProcess.NormalExit:
            # 비식별화가 완료됨을 알리는 알림 대화 상자 표시
            self.progress_dialog.accept()
            QMessageBox.information(self, "info", "비식별화가 완료되었습니다!", QMessageBox.Ok)
        else:
            QMessageBox.critical(self, "Error", "비식별화 중 오류가 발생했습니다.", QMessageBox.Ok)

    @pyqtSlot(QProcess.ProcessError)
    def on_process_error(self, error):
        QMessageBox.critical(self, "Error", f"비식별화 프로세스에서 오류 발생: {error}", QMessageBox.Ok)
        
        

