List of dependencies:

	- Numpy 
	- Mediapipe
	- Scikit-Learn
	- Scipy
	- PySide2
	- Opencv-python-headless
	- Scikit-Learn
	- Onnxruntime
	- Onnx
	- telegram-send (cofiguration: https://medium.com/@robertbracco1/how-to-write-a-telegram-bot-to-send-messages-with-python-bcdf45d0a580)
	- depthai

It´s recomended to install them in the specified order, mediapipe will install opencv, aidTalk needs openCV-headless. Before pip install opencv-python-headless, is needed pip uninstall opencv-python to remove the opencv installed with mediapipe.

It´s is recomended that PyQt5 is not in the running environment, because it may cause somo conflict with PySide2.

Run main_aidTalk.py

Help:

	-v: Voice gender.  0 for female, 1 for male
	-m: Monitor selection. 0 for main monitor, 1 for second monitor. Second monitor should be place at the left.
	-w: Webcam source. 0, 1, 2, etc.
	

Example for Macbook Pro:

python3 main_aidTalk.py -c 1280 720 -F 0.63 -f 1085.54 -w 0 -m 1 -v 1

Example for Surface Pro:

python main_aidTalk.py -c 640 480 -F 0.3 -f 520.37






