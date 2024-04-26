# About SUMO-Ray-VFogsim++

This repo contains the Python code for training vehicular fog nodes using multi-agent reinforcement learning (MARL), and the environment is built in Ray RLlib.
Check more details from: https://www.techrxiv.org/doi/full/10.36227/techrxiv.24184047.v2

To get it running, you need to download the VFogSim Platform (OMNeT) from the following link:
https://mobilecloud.aalto.fi/?page_id=1441

The demonstration video is available at:
https://www.youtube.com/watch?v=KjmwOsR4WEc

![5G_V2X](https://github.com/JiamingYIN/vfogsim/assets/61701502/738a510c-8f18-465b-848c-516d041c3bf1)


How to install:
-------------------------------------------------------------------------  

VFogSim (Omnetpp Version)

1. Download and install sumo-1.8.0, omnetpp-5.6.2, inet-4.2.2, veins-5.2, and simu5g-1.1.0. If there is no /usr/bin/python, set a symbolic link to an existing python (2.7 tested)
   'ln -s /usr/bin/python2.7 /usr/bin/python'

2. use file LtePhyUe.h instead of /simu5G/src/stack/phy/layer/LtePhyUe.h (to add the get_SINR function for cars and VFNs)
   Copy LteNetworkConfigurator*(three files) into /simu5G/src/common/

3. Install Gurobi-8.0.1 (for mac) or Gurobi-9.5.2 (for linux, tested with ubuntu 20.04.1)
   Copy lib/libgurobi_g++4.2.a and libgurobi80.so (for mac) or lib/libgurobi_g++5.2.a and libgurobi95.so (for linux) into veins-inet/src
   Click Project/Properties/OMNeT++/Makemake/src:makemake.../ click Options in Build.
   Click Link and add "gurobi80" and "gurobi_g++4.2" (for max) or "gurobi95" and "gurobi_g++5.2" (for linux) in Additional libraries to link with:(-l option)

   Click Project/Properties/ C/C++ General/Paths and Symbols/Includes
   Add the include path of gurobi in the Assembly,GNU C,GNU C++

4. move directory vfogsim to /simu5G/src/apps
   move directory cars to /Simu5G/simulations/NR
   
-------------------------------------------------------------------------   

   The directory vfogsim contains the functions of cars, VFNs, server and the scheduler algorithm.

   /vfogsim/vfogsim_car.cc
   /vfogsim/vfogsim_car.h
   /vfogsim/vfogsim_car.ned

   These three files define the operation of the cars.
   Basically, the car receives three kinds of messages
     - selfInfo message means it is time to send the information of cars to the server;
     - Decision packet means the car receives the task decision of the server;
     - Result packet means the car receives the task result of the VFN
   The car sends two kinds of messages
     - selfInfo message will be sent to itself each TTI as a timer
     - Task_car packet will be sent to the allocated VFN


   /vfogsim/vfogsim_vfn.cc
   /vfogsim/vfogsim_vfn.h
   /vfogsim/vfogsim_vfn.ned

   These three files define the operation of the VFNs.
   Basically, the VFN receives two kinds of messages
     - selfInfo message means it is time to send the information of VFNs to the server;
     - Task_car packet means the VFN receives the task of a car
   The VFN sends two kinds of messages
     - selfInfo message will be sent to itself each TTI as a timer
     - Result packet will be sent to the car which sends the Task_car packet


   /vfogsim/vfogsim_server.cc
   /vfogsim/vfogsim_server.h
   /vfogsim/vfogsim_server.ned
   /vfogsim/scheduler.cc
   /vfogsim/scheduler.h
   /vfogsim/info.h

   The first three files define the operation of the server.
   Basically, the server receives three kinds of messages
     - selfDealInfo message means it is time to begin the scheduler decision making;
     - Info_car packet means the server receives the information of a car;
     - Info_VFN packet means the server receives the information of a VFN;
   The server sends two kinds of messages
     - selfDealInfo message will be sent to itself each TTI as a timer
     - Decision packet will be sent to the car to tell whether the task of a car is blocked or not.

   The last three files define the network and computation scheduler which use gurobi.


   /vfogsim/vfogsim_info.msg
   /vfogsim/vfogsim_vfn_info.msg
   /vfogsim/vfogsim_decison.msg
   /vfogsim/vfogsim_task.msg

   These four files define the packet parameters.


   /cars/heterogenous*
     These files are related to vehicle route in sumo
   /cars/Highway.ned
     This file determines the network topology
   /cars/omnetpp.ini
     This file is related to the configuration of the simulation
     "*.veinsManager.updateInterval" control the update interval between sumo and omnetpp
     "*.gNodeB_[*].mobility.initial*" control the location of the base station
     "*.car[*].cellularNic.nrPhy.handoverLatency" control the migration latency of cars
     "*.car[*].app[0].typename" control whether a car is a user or a VFN



5. Start a terminal, run 'omnetpp' to start the OMNeT++ IDE, run '/home/vfogsim/Documents/veins-5.2/veins-veins-5.2/sumo-launchd.py -vv -c sumo' to connect OMNeT++ with SUMO via TraCI

6. run the omnetpp.ini file under /Simu5G/simulations/NR/cars and choose the config VFogsim
   By far, the operation log of cars,VFNs and server will be output into the log file defined in the /vfogsim/vfogsim_car/vfn/server.h

User Manual of Ray-SUMO-VFogSim++    
-------------------------------------------------------------------------

-----------------------In terminal------------------------------------

<How to connect Veins with SUMO?>
/home/vfogsim/Documents/veins-5.2/veins-veins-5.2/sumo-launchd.py -vv -c sumo
(There should be something like "Listening on port 9999")

<How to open OMNET++?>
omnetpp
(Then the OMNeT GUI should pop up)

-----------------------In OMNeT IDE---------------------------------------

<What shall I do if the project explorer is empty? (This happens when restart the VM unproperly)>
Click "file/import/Existing Projects into Workspace", choose inet, veins, veins_inet(sub-directory of veins), simu5G in "Home/Documents"
Build them properly according to the guid in veins and simu5G webpages, as well as "About VFogSim.txt"

<How can I run the VFogSim++ simulation in OMNeT IDE?>
Go to "simu5G/simulations/NR/cars/omnetpp.ini", click the run button (a green circle with a while triangle)
Note: The line in "omnetpp.ini" should be commented: ned-path = ../..;../../../src;../../../../inet-4.2.2-src/inet4/src; ......

<How can I check the VFogSim source file?>
Go to "simu5G/src/apps/vfogsim"

-----------------------In pycharm-----------------------------------------

<How can I go the reinforcement learning directory?>
cd /home/vfogsim/Documents/rllib_v2v

<How to train the reinforcement learning model?>
python3 train_centralized_critic_omnet_5agent.py
Note: The python file can be replaced here
1. When the route file changes, the new route file should be added to "test.lauchd.xml"; the "test.sumocfg" should be changed (both in simu5G/simulations/NR/cars)
2. When the number of agents changes, the config file should be changed; the input size in "model.py" should be 30*N, where N is the number of agents. The "omnetpp.ini" file should also be changed. For example, for three agents:

 assignment of VFN
*.car[0..2].numApps = 1
*.car[0..2].app[0].typename = "vfogsim_vfn"
*.car[0..2].app[0].destAddress = "server"
*.car[0..2].app[0].destPort = 3000
*.car[0..2].app[*].localPort = 3000

assignment of user vehicles
*.car[3..].numApps = 1
*.car[3..].app[0].typename = "vfogsim_car"
*.car[3..].app[0].destAddress = "server"
*.car[3..].app[0].destPort = 3000
*.car[3..].app[*].localPort = 3000

<How to run OMNeT at the same time without OMNeT IDE?>
bash run_omnet.sh
Note: 1. This command should be entered after the python file runs (to reset timestep)
2. The line in "omnetpp.ini" should be added: ned-path = ../..;../../../src;../../../../inet-4.2.2-src/inet4/src; ......

<When the training is completed, how can we see the results?>
tensorboard --logdir="/home/vfogsim/ray_results/<name of the folder>"



