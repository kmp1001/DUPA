export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/compat
pip3 install -r requirements.txt
git branch --set-upstream-to=origin/master master
git pull