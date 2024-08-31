#!/usr/bin/env python3
"""Plot the live microphone signal(s) RFFT with matplotlib.

Matplotlib and NumPy have to be installed.

"""
import argparse
import pathlib
import queue
import sys
import math
import json
import sounddevice as sd
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
from soundfilter import SoundFilter
from settings import settings

from threading import Timer

class RepeatedTimer(object):
    def __init__(self, interval, function, *args, **kwargs):
        self._timer     = None
        self.interval   = interval
        self.function   = function
        self.args       = args
        self.kwargs     = kwargs
        self.is_running = False
        self.start()

    def _run(self):
        self.is_running = False
        self.start()
        self.function(*self.args, **self.kwargs)

    def start(self):
        if not self.is_running:
            self._timer = Timer(self.interval, self._run)
            self._timer.start()
            self.is_running = True

    def stop(self):
        self._timer.cancel()
        self.is_running = False

def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text

class SpectrumAnalyser(object):

    def __init__(self, block_duration, channel, plot_interval, record_path):
        
        self.block_duration = block_duration
        self.channel = channel
        self.plot_interval = plot_interval
        if record_path != None:
            self.record_path = pathlib.Path(record_path)
        else:
            self.record_path = record_path
        self.q = queue.Queue()
        self.datafile = None
        self.plot_date = None
        self.alpha = 0.25
        self.rt = None

    def set_input(self, input_device, block_duration = 250, api = 'MME'):
        self._input_id, self._input_device = SoundFilter.get_device(input_device, 'input', api)

        # If input device cannot be found, try other names
        input_names = ['cam', 'web', 'phone']
        if not self._input_device:
            for name in input_names:
                self._input_id, self._input_device = SoundFilter.get_device(name, 'input')
                if input_device:
                    break
        if not input_device:
            input_device = sd.query_devices(kind='input')
        if not input_device:
            raise ValueError("Unable to find any input device for sound filter to function.")

        # Do information calculation and output information for user
        self.input_device_info = sd.query_devices(self.input_device, 'input')
        print('Using input device:', self.input_device, self.input_device_info)

        self.samplerate = self.input_device_info['default_samplerate']
        self.block_size = math.ceil(self.samplerate * self.block_duration / 1000)
        self.freq = np.fft.rfftfreq(self.block_size, d=1./self.samplerate)
        self.fftlen = len(self.freq)
        self.plot_data = np.zeros(self.fftlen)
        print('FFT spectrum with', self.fftlen, 'points of data corresponding to a range of', str(self.freq[1])+'Hz per point of data.')
    
    def set_playback(self, playback_device, api = 'MME'):
        self._playback_id, self._playback_device = SoundFilter.get_device(playback_device, 'playback', api)

    def set_record_path(self, record_path):
        if record_path != None:
            self.record_path = pathlib.Path(record_path)
        else:
            self.record_path = record_path
        self.set_input(self.input_device)

    def start_recording(self, max_duration: float=120):
        if self.record_path:
            try:
                self.datafile = open(self.record_path.with_suffix('.csv'),'wb')
                np.savetxt(self.datafile, self.freq[None,:], delimiter=',', fmt='%.4f')
                ## Need a thread that continuously services the queue
                ## First argument is time in seconds between calls to update_datafile
                self.rt = RepeatedTimer(0.01, self.update_datafile) # it auto-starts, no need of rt.start()
            except Exception as e:
                print(e)
                self.datafile = None
        self.stream = sd.InputStream(device=self.input_device, blocksize=int(self.samplerate * self.block_duration / 1000),
            samplerate=self.samplerate, callback=self.audio_callback)
        self.recording = sd.rec(int(max_duration * self.samplerate), samplerate=self.samplerate, channels=2)
        self.stream.start()

    def stop_recording(self):
        self.stream.stop()
        self.stream.close()
        if self.rt != None:
            self.rt.stop()
        sd.stop()
        if self.datafile:
            self.update_datafile()
            self.datafile.close()
        print("Recording Stopped")

    def playback_recording(self):
        # TODO: Blocking=True doesn't work properly...
        try:
            # import time
            print(self._playback_id, self._playback_device)
            sd.play(self.recording, self.samplerate, device=self._playback_id, blocking=False)
            # time.sleep(30)
            # sd.stop()
        except Exception as e:
            print(e)

    def live_plot_and_capture(self):
        try:
            self.start_recording()

            fig, ax = plt.subplots()
            self.lines = ax.plot(self.plot_data)
            ax.axis((0, len(self.plot_data), 0, 1))
            # ax.set_yticks([0])
            ax.yaxis.grid(True)
            # ax.tick_params(bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
            fig.tight_layout(pad=0)

            ani = FuncAnimation(fig, self.update_plot, interval=self.plot_interval, blit=True)
            plt.show()
    
        except Exception as e:
            print(e)
        finally:
            self.stop_recording()
            if self.datafile:
                self.datafile.close()
            print("Capture Finished")
    
    def audio_callback(self, indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(status)
        if not self.q.full():
            self.q.put(indata[:, self.channel])

    def update_datafile(self):
        try:
            while not self.q.empty():
                data = self.q.get()
                data = np.fft.rfft(data)
                data = np.abs(data)

                np.savetxt(self.datafile, data[None,:], delimiter=',', fmt='%.4f')

        except queue.Empty:
            print("nothing to save")

    def update_plot(self, frame):
        """This is called by matplotlib for each plot update.s

        Typically, audio callbacks happen more frequently than plot updates,
        therefore the queue tends to contain multiple blocks of audio data.

        """
        try:
            data = self.q.get()
            data = np.fft.rfft(data)
            data = np.abs(data)
            self.plot_data = self.alpha*data+(1-self.alpha)*self.plot_data

            for line in self.lines:
                line.set_ydata(self.plot_data/np.ptp(self.plot_data))

            if self.datafile:
                np.savetxt(self.datafile, data[None,:], delimiter=',', fmt='%.4f')

            return self.lines

        except queue.Empty:
            return self.lines

    @property
    def input_device(self):
        return self._input_device['name']

    @property
    def playback_device(self):
        return self._playback_device['name']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-l', '--list-devices', action='store_true', help='show list of audio devices and exit')
    parser.add_argument('-i', '--input-device', type=int_or_str, default = settings['input-api'], help='input device (numeric ID or substring)')
    parser.add_argument('-o', '--playback-device', type=int_or_str, default = settings['playback-api'], help='playback (output) device (numeric ID or substring)')
    parser.add_argument('-b', '--block-duration', type=float, metavar='DURATION', default=50, help='block size (default %(default)s milliseconds)')
    parser.add_argument('-c', '--channel', type=int, default=0, help='channel used for spectrum analysis (default 0)')
    parser.add_argument('-p', '--plot-interval', type=float, default=30, help='minimum time (in ms) between plot updates (default: %(default)s ms)')
    parser.add_argument('-r', '--record-path', default=None, help='Save fft of recording to <logging path>.csv. If not provided, recording is not saved')
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        parser.exit(0)

    if args.channel < 0:
        parser.error('Channel must be greater than 0')

    # if args.record_path:
    #     record_path = pathlib.Path(args.record_path)
    # else:
    #     record_path = None

    sa = SpectrumAnalyser(args.block_duration, args.channel, args.plot_interval, args.record_path)
    sa.set_input(args.input_device)
    sa.set_playback(args.playback_device) # Not currently used
    if args.plot_interval != 0:
        sa.live_plot_and_capture()
    else:
        sa.start_recording()
        input()
        sa.stop_recording()
