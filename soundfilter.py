#!/usr/bin/env python3
"""

"""
from numpy import abs, mean
from numpy.fft import rfft
import json
import pickle
import sounddevice as sd

class SoundFilter():
    def __init__(self, input_device='microphone', output_device='CABLE Input', block_duration = 50):
        self.model_name = None
        self._input_id, self._input_device = self.get_device(input_device, 'input')
        self._output_id, self._output_device = self.get_device(output_device, 'output')

        # If input device cannot be found, try other names
        input_names = ['cam', 'web', 'phone']
        if not self._input_device:
            for name in input_names:
                self._input_id, self._input_device = self.get_device(name, 'input')
                if input_device:
                    break

        if not input_device:
            input_device = sd.query_devices(kind='input')
        
        if not input_device:
            raise ValueError("Unable to find any input device for sound filter to function.")

        print("Input device info:")
        print(json.dumps(sd.query_devices(self._input_id), indent = 4))

        print("Output device info:")
        print(json.dumps(sd.query_devices(self._output_id), indent = 4))

        self.samplerate = self._input_device['default_samplerate']
        self.block_size = int(self.samplerate * block_duration / 1000)
        self.stream = None

        import atexit
        atexit.register(self.stop)

    def _new_stream(self):
        if self.stream:
            self.stream.stop()
            self.stream = None
        self.stream = sd.Stream(device=(self._input_id, self._output_id), channels=2, samplerate=self.samplerate, blocksize=self.block_size, callback=self.callback)

    @property
    def input(self):
        return self._input_device['name']

    @property
    def output(self):
        return self._output_device['name']

    def set_input(self, input_device, api = 'MME', block_duration = 50):
        self._input_id, self._input_device = self.get_device(input_device, 'input', api)
        self.samplerate = self._input_device['default_samplerate']
        self.block_size = int(self.samplerate * block_duration / 1000)
        if self.stream:
            self.start()
    
    def set_output(self, output_device, api = 'MME'):
        self._output_id, self._output_device = self.get_device(output_device, 'output', api)
        if self.stream:
            self.start()

    def start(self):
        with open(self.model_name+".pickle", 'rb') as f:
            self.clf = pickle.load(f)
        self._new_stream()
        self.stream.start()
        return self
    
    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream = None
        return self

    def activation_function(self, data):
        data = data.reshape(1, -1)
        if self.clf.predict(data)[0]:
            return True
        else:
            return False

    def callback(self, indata, outdata, frames, time, status):
        data = rfft(indata[:, 0])
        data = abs(data)
        if self.activation_function(data):
            outdata[:] = indata
        else:
            outdata[:] = 0

    @staticmethod
    def get_device(name = 'CABLE Input', kind = 'output', api = 0):    
        # print(name)   
        if isinstance(name, int):
            return name, sd.query_devices(name)

        devices = sd.query_devices()
        matching_devices = []
        for device_id in range(len(devices)):
            if name.lower() in devices[device_id].get('name').lower():
                try:
                    if kind == 'input':
                        sd.check_input_settings(device_id)
                    elif kind == 'output':
                        sd.check_output_settings(device_id)
                    elif kind == 'playback':
                        sd.check_output_settings(device_id)
                    else:
                        print('Invalid kind')
                        return None
                    matching_devices.append((device_id, devices[device_id]))
                    # print(matching_devices)
                except:
                    pass
        
        if not matching_devices:
            print("Unable to find device matching name", name, 'of kind', kind)
            return None

        found = False

        if isinstance(api, int):
            api = sd.query_hostapis(api).get('name')
        for device_id, device in matching_devices:
            if api in sd.query_hostapis(int(device.get('hostapi'))).get('name'):
                found = True
                break
        
        if not found:
            print("Unable to find device matching host api", api, 'using first available...')
            return matching_devices[0]
        else:
            return device_id, device


if __name__ == "__main__":
    import argparse
    from settings import settings

    def parse_arguments():
        def int_or_str(text):
            """Helper function for argument parsing."""
            try:
                return int(text)
            except ValueError:
                return text

        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('-i', '--input-device', type=int_or_str, default = settings.get("input-device", 'microphone'), help='input device ID or substring')
        parser.add_argument('-o', '--output-device', type=int_or_str, default = settings.get("output-device", 'CABLE Input'), help='output device ID or substring')
        parser.add_argument('-b', '--block-duration', type=float, metavar='DURATION', default = settings.get("block-duration", 50), help='block size (default %(default)s milliseconds)')
        return parser
        
    parser = parse_arguments()
    args = parser.parse_args()

    sf = SoundFilter(args.input_device, args.output_device, args.block_duration).start()
    print('#' * 80)
    print('press Return to quit')
    print('#' * 80)
    input()