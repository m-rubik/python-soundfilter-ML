### python-soundfilter
Quick and dirty spectrum analyzer and audio filter for Windows using the sounddevice library. Currently is able to be used to filter out noise, keyboard typing, and provide a sound level based activation filter for VoIP use.
### Pre-requisites
Download and install https://www.vb-audio.com/Cable/ in order to create a virtual microphone to be used by VoIP applications to receive your filtered audio.
### Quick Start Guide
1) Download a copy of the filter from releases: https://github.com/harryhecanada/python-soundfilter/releases
2) Run the application
3) Change target application's input device to CABLE Output (Requires the Virtual Audio Cable!)


Use spectrum_analyser with log option to record audio of:
1) Talking
2) Talking + Typing
3) Talking + Mouse Clicks
4) Talking + Mouse Clicks + Typing
5) No talking (background baseline)
6) Typing
7) Mouse clicks
8) Mouse clicks + Typing