/*
 * vfogsim_cfn.h
 *
 *  Created on: 2022年10月27日
 *      Author: lys
 */

#ifndef _VFOGSIM_CFN_H_
#define _VFOGSIM_CFN_H_

#include <string.h>
#include <omnetpp.h>
#include <fstream>

#include "stack/phy/layer/LtePhyUe.h"
#include <inet/networklayer/common/L3AddressResolver.h>
#include <inet/transportlayer/contract/udp/UdpSocket.h>
//#include "apps/vfogsim/vfogsim_cfn_info_m.h"
#include "veins_inet/veins_inet.h"
#include "veins_inet/VeinsInetMobility.h"

using namespace std;

class vfogsim_cfn : public omnetpp::cSimpleModule
{
    inet::UdpSocket socket;
    veins::VeinsInetMobility* mobility;
    veins::TraCICommandInterface* traci;
    veins::TraCICommandInterface::Vehicle* traciVehicle;


    int calculate_resource = 2500;

    ofstream fout;
    string path = "/Users/lys/Desktop/aalto/output/cfn/"; //output log path
    string vfn_ids;
    vector<string> existed_vfns;

    double period_time =1;
    omnetpp::cMessage* selfInfo_;

    unsigned int totalSentBytes_;
    omnetpp::simtime_t warmUpPer_;


    omnetpp::cMessage *initTraffic_;

    omnetpp::simtime_t timestamp_;
    int localPort_;
    int destPort_;
    inet::L3Address destAddress_;

    void initTraffic();
    bool Is_In_the_Same_Period(simtime_t time1,simtime_t time2);
    omnetpp::simtime_t next_period_time(double period,double bias,omnetpp::simtime_t now_time);
  public:
    ~vfogsim_cfn();
    vfogsim_cfn();

  protected:

    virtual int numInitStages() const  override { return inet::NUM_INIT_STAGES; }
    void initialize(int stage) override;
    void handleMessage(omnetpp::cMessage *msg) override;
    void add_vehicle(Packet* pPacket);
    void detector();

};

#endif







