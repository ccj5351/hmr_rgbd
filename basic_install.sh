#------------------------
# after come into the docker env, do the following inside your docker:
apt-get update && apt-get install vim && apt-get install python-tk
dpkg-reconfigure tzdata
pip install --user transforms3d
pip install --upgrade pip

# inside your docker:
echo $PYTHONPATH
# if opendr is not found
# or some error like: camfrom opendr.camera import ProjectPoints
# module camera is not found
# then do the following:
export PYTHONPATH=/usr/local/lib/python2.7/dist-packages/:${PYTHONPATH}
# i.e., add 
#'/usr/local/lib/python2.7/dist-packages/' to 
# PYTHONPATH


# ----------------------------------
# this part is not always necessary
# inside the docker env:
# for opendr and smpl
apt-get install libgl1-mesa-dev
# install OSMesa
apt-get install libosmesa6-dev

# install GLU
apt-get install freeglut3-dev
