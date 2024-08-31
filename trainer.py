import sounddevice
import json
import time
import PyQt5.QtWidgets as qw
from PyQt5.QtCore import pyqtSlot
import PyQt5.QtGui as qg
from spectrum_analyzer import SpectrumAnalyser
from settings import settings
import pathlib
from ai import AI


class TrainerWindow(qw.QMainWindow):
    def __init__(self):
        super(TrainerWindow, self).__init__()

        self.title = 'AI Trainer'
        self.setWindowTitle(self.title)

        self.sa = SpectrumAnalyser(block_duration=50, channel=0, plot_interval=0, record_path="default")
        self.ai = AI()

        self.add_widgets_input()
        self.add_widgets_playback()
        self.add_widgets_recording()
        self.add_widgets_ai()
    
        # Construct the layout
        self.main_layout = qw.QGridLayout()
        self.main_layout.addWidget(self.input_group_box, 0, 0)
        self.main_layout.addWidget(self.playback_group_box, 0, 1)
        self.main_layout.addWidget(self.recording_group_box, 1, 0)
        self.main_layout.addWidget(self.ai_group_box, 2, 0)
        self.main_layout.setRowStretch(1, 1)
        self.main_layout.setRowStretch(2, 1)
        self.main_layout.setColumnStretch(0, 1)
        self.main_layout.setColumnStretch(1, 1)

        self.central_widget = qw.QWidget()
        self.central_widget.setLayout(self.main_layout)
        self.setCentralWidget(self.central_widget)


    def add_widgets_input(self):
         # Add input APIs
        self.input_layout = qw.QVBoxLayout()
        self.input_layout.addWidget(qw.QLabel("API"))
        self.input_devices = qw.QComboBox()
        self.input_apis = qw.QComboBox()
        self.input_apis.setCurrentIndex(settings['input-api'])
        self.set_input_api(settings['input-api'])
        self.input_apis.activated.connect(self.set_input_api)
        for api in sounddevice.query_hostapis():
            self.input_apis.addItem(api.get('name'))
        self.input_layout.addWidget(self.input_apis)
        self.set_input_api(self.input_apis.currentIndex())

        self.set_input_device(settings['input-device'])
        self.input_devices.setCurrentText(self.sa.input_device)
        self.input_devices.textActivated.connect(self.set_input_device)

        self.input_layout.addWidget(qw.QLabel("Device"))
        self.input_layout.addWidget(self.input_devices)
        
        self.input_group_box = qw.QGroupBox("Input")
        self.input_group_box.setLayout(self.input_layout)
    
    def add_widgets_playback(self):
        # Add playback APIs
        self.playback_layout = qw.QVBoxLayout()
        self.playback_layout.addWidget(qw.QLabel("API"))
        self.playback_devices = qw.QComboBox()
        self.playback_apis = qw.QComboBox()
        self.playback_apis.setCurrentIndex(settings['playback-api'])
        self.set_playback_api(settings['playback-api'])
        self.playback_apis.activated.connect(self.set_playback_api)
        for api in sounddevice.query_hostapis():
            self.playback_apis.addItem(api.get('name'))
        self.playback_layout.addWidget(self.playback_apis)
        self.set_playback_api(self.playback_apis.currentIndex())

        self.set_playback_device(settings['playback-device'])
        self.playback_devices.setCurrentText(self.sa.playback_device)
        self.playback_devices.textActivated.connect(self.set_playback_device)

        self.playback_layout.addWidget(qw.QLabel("Device"))
        self.playback_layout.addWidget(self.playback_devices)
        
        self.playback_group_box = qw.QGroupBox("Playback")
        self.playback_group_box.setLayout(self.playback_layout)

    def add_widgets_recording(self):
        self.recording_layout = qw.QVBoxLayout()
        
        self.recording_label_1 = qw.QLabel(self)
        self.recording_label_1.setText("<i>Press record then read the following text, pressing Finish when done:</i><br>That quick beige fox jumped in the air over each thin dog.<br>Look out, I shout, for he's foiled you again, creating chaos.")

        self.recording_button_1 = qw.QPushButton('button_recording_1', self)
        self.recording_button_1.setObjectName("button_recording_1")
        self.recording_button_1.setText("Record")
        # self.recording_button_1.setIcon(qg.QIcon('record_img.png'))
        self.recording_button_1.clicked.connect(lambda: self.toggle_recording("recording_1"))

        self.recording_layout.addWidget(self.recording_label_1)
        self.recording_layout.addWidget(self.recording_button_1)

        self.recording_group_box = qw.QGroupBox("Recording")
        self.recording_group_box.setLayout(self.recording_layout)

    def add_widgets_ai(self):
        self.ai_layout = qw.QVBoxLayout()
        
        self.train_button = qw.QPushButton('button_train', self)
        self.train_button.setObjectName("button_train")
        self.train_button.setText("Train")
        # self.train_button.setIcon(qg.QIcon('train_img.png'))
        self.train_button.clicked.connect(self.train)

        self.ai_layout.addWidget(self.train_button)

        self.ai_group_box = qw.QGroupBox("AI")
        self.ai_group_box.setLayout(self.ai_layout)

    @pyqtSlot()
    def toggle_recording(self, ctx):
        button = qw.qApp.focusWidget()
        text = button.text()
        if text == "Record":
            button.setText("Finish")
            self.sa.set_record_path(ctx)
            self.sa.start_recording()
            print("Starting Recording")
        else:
            button.setText("Record")
            self.sa.stop_recording()
            print("Done Recording")

        # # button.setText("Recording")
        # # print("AAAAAAAAAAAAAAAA", button.objectName())
        # # # button.setIcon(qg.QIcon('record_img.png'))
        # # self.sa.set_record_path(ctx)
        # self.sa.start_recording()
        # time.sleep(5)
        # self.sa.stop_recording()
        # # button.setText("Start Recording")
        # # button.setIcon(qg.QIcon())
        # print("Done Recording")

    @pyqtSlot()
    def train(self):
        # df_noise = pd.read_csv("sample_typing.csv")
        # df_talking = pd.read_csv("talking_4.csv")
        
        self.ai.load_data_talking("sample_talking")
        self.ai.load_data_noise("sample_typing")
        self.ai.merge_datasets()
        self.ai.generate_features()

        self.ai.generate_voting()


    def change_style(self, styleName):
        qw.QApplication.setStyle(qw.QStyleFactory.create(styleName))
    
    def update_devices(self, kind, api):
        if kind == 'input':
            self.input_devices.clear()
        elif kind == 'playback':
            self.playback_devices.clear()

        devices = sounddevice.query_devices()
        for device_id in range(len(devices)):
            device = devices[device_id]
            if device.get("hostapi") == api:
                try:
                    if kind == 'input':
                        sounddevice.check_input_settings(device_id)
                        self.input_devices.addItem(device['name'])
                    elif kind == 'playback':
                        sounddevice.check_output_settings(device_id)
                        self.playback_devices.addItem(device['name'])
                except:
                    pass

    def set_input_api(self, index):
        self.update_devices('input', index)
        settings['input-api'] = index
        
    def set_playback_api(self, index):
        self.update_devices('playback', index)
        settings['playback-api'] = index

    def set_input_device(self, input_device):
        # print(self.input_apis.currentIndex())
        # print(self.input_apis.currentText())
        self.sa.set_input(input_device, self.input_apis.currentIndex())
        settings['input-device'] = self.sa.input_device
    
    def set_playback_device(self, playback_device):
        self.sa.set_playback(playback_device, self.playback_apis.currentIndex())
        settings['playback-device'] = self.sa.playback_device
    
if __name__ == "__main__":
    app = qw.QApplication([])
    print(qw.QStyleFactory.keys())
    if 'Fusion' in qw.QStyleFactory.keys():
        app.setStyle('Fusion')
    fw = TrainerWindow()
    fw.show()
    app.exec_()