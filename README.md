# CMPE 255 Project

Project proposal: https://docs.google.com/document/u/1/d/1Eae2wwwXqEORiP8mPCPLyD8CwjsVQ8aXj1zd8F0BXD0/edit?usp=drive_web&ouid=109387012218868591245  
RandomForestAlgorithm implementation is based on Charanpal Dhanjal work https://gist.github.com/charanpald/c216800e25480ee838e8  

Group members:  Yang Chen, Fulbert Jong, David Tang

## Environment setup

1.  Install docker-ce
2.  Install docker-compose via `pip install docker-compose`
3.  The dataset used is the KDD Cup 99 dataset http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html

To download dataset:  
   Download KDD Cup 99 dataset with:

   wget http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data.gz

   or

   wget http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz

   and uncompress it. The first one occupies about 740MBytes, the second one
   around 71MBytes. Should you use the smaller dataset, please adjust filename
   in code:
   raw_data_filename = data_dir + "kddcup.data"
   change by
   raw_data_filename = data_dir + "kddcup.data_10_percent"


## Running the project

`docker-compose up --build`  
`python detectAttack.py`


## Errors

1. ERROR: Couldn't connect to Docker daemon - you might need to run `docker-machine start default`.  
`$newgrp docker`
