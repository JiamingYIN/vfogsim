for i in {1..100}
do
  cd /home/vfogsim/Documents/Simu5G-1.1.0/simulations/NR/cars
  opp_run omnetpp.ini -l /home/vfogsim/Documents/Simu5G-1.1.0/src/libsimu5g.so -u Cmdenv -c VFogsim --sim-time-limit=300s
done