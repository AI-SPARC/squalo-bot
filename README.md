# squalo-bot
Framework for a waste collector unmanned surface vehicle running as a differential boat

# Instructions

First, the user needs to create and activate an env to run the project with the command: 

```
python3 -m venv env
source env/bin/activate
```

Then, install requirements with:

```
pip3 install -r requirements.txt
```

In the end, to run Casadi correctly, the ambient need ipopt module, so, the command will help:

```
sudo apt-get install gcc g++ gfortran git patch wget pkg-config liblapack-dev libmetis-dev
```

## Running

Now, open your coppelia software, and load the simulation.ttt file there.

Once it loaded, just run:

```
python3 boat_control.py
```