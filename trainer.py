import sounddevice
import json
import PyQt5.QtWidgets as qw
from PyQt5.QtCore import pyqtSlot
from spectrum_analyzer import SpectrumAnalyser
from settings import settings
import pathlib


class TrainerWindow(qw.QMainWindow):
    def __init__(self):
        super(TrainerWindow, self).__init__()

        self.title = 'AI Trainer'
        self.setWindowTitle(self.title)

        self.sa = SpectrumAnalyser(block_duration=50, channel=0, plot_interval=0, record_path="Text1")

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

        # self.sa = SpectrumAnalyser(block_duration=50, channel=0, plot_interval=0, record_path="Text1")

        # self.start_recording()
        # import time
        # time.sleep(2)
        # self.stop_recording()

        # Add input and playback devices
        self.set_input_device(settings['input-device'])
        print("SDASD")
        self.input_devices.setCurrentText(self.sa.input_device)
        self.input_devices.textActivated.connect(self.set_input_device)

        self.set_playback_device(settings['playback-device'])
        self.playback_devices.setCurrentText(self.sa.playback_device)
        self.playback_devices.textActivated.connect(self.set_playback_device)

        self.input_layout.addWidget(qw.QLabel("Device"))
        self.input_layout.addWidget(self.input_devices)
        self.playback_layout.addWidget(qw.QLabel("Device"))
        self.playback_layout.addWidget(self.playback_devices)
        
        self.input_group_box = qw.QGroupBox("Input")
        self.input_group_box.setLayout(self.input_layout)
        self.playback_group_box = qw.QGroupBox("Playback")
        self.playback_group_box.setLayout(self.playback_layout)


        # Add everything together
        self.main_layout = qw.QGridLayout()
        self.main_layout.addWidget(self.input_group_box, 0, 0)
        self.main_layout.addWidget(self.playback_group_box, 0, 1)
        self.main_layout.setRowStretch(1, 1)
        self.main_layout.setRowStretch(2, 1)
        self.main_layout.setColumnStretch(0, 1)
        self.main_layout.setColumnStretch(1, 1)

        self.central_widget = qw.QWidget()
        self.central_widget.setLayout(self.main_layout)
        self.setCentralWidget(self.central_widget)


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
        print(self.input_apis.currentIndex())
        print(self.input_apis.currentText())
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