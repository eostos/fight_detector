sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install software-properties-common
sudo apt-get install -y python3.6 python3.6-venv python3.6-dev libgl1-mesa-glx  zip protobuf-compiler git
sudo python3.6 -m pip install --upgrade pip
python3.6 -m venv venv_fight_3.6
python3.6 -m pip install  testresources
##still tensor flow is not necesary cause it is not running on gpu 
python3.6  -m pip install tensorflow-gpu==2.0.0
python3.6 -m pip install redis
python3.6 -m pip install --upgrade pillow
python3.6 -m pip install keras==2.3.1
python3.6 -m pip install  'h5py==2.10.0' --force-reinstall
python3.6 -m pip install scikit-image
python3.6  -m pip install tensorflow==2.0.0
python3.6  -m pip install opencv-contrib-python-headless==4.5.3.56

