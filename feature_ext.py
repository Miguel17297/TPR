import sys
import argparse
import datetime
import numpy as np
from netaddr import IPNetwork, IPAddress, IPSet
import pyshark
import os

PKT_UPLOAD = 0  # Upload Packet
PKT_DOWNLOAD = 1  # Download Packet


def captureProcess(capture, lengthObsWindow, slidingValue,file_name):

    features_path = os.path.join(os.path.join(os.getcwd()), "data")

    if not os.path.exists(features_path):
        os.makedirs(features_path)

    with open(os.path.join(features_path, file_name), "w") as f:

        n_pkts = 0  # number of packets processed
        

        for pkt in capture: 
            # reset
            if n_pkts == lengthObsWindow:
                # update counter
                n_pkts = lengthObsWindow - slidingValue
                
                # compute feature
                features = computeFeatures(data)
                # save result
                np.savetxt(f, features.reshape(1, -1))
                # update data
                data = data[slidingValue:, ]
            if ((pkt_info := pktValidation(pkt)) is not None):

                if 'data' not in locals():
                    data = [*pkt_info]
                else:
                    data = np.vstack((data, [*pkt_info]))
                n_pkts = n_pkts + 1
                


def computeFeatures(data):
    # compute metrics
    diff_pkts = np.diff(data[:, 1])  # diference between packets

    diff_up_up = np.diff(data[data[:, 0] == PKT_UPLOAD][:, 1])  # differente between upload upload
    diff_down_down = np.diff(data[data[:, 0] == PKT_DOWNLOAD][:, 1])  # differente between upload upload
    diff_up_down = np.array(
        [row2[1] - row1[1] for row1, row2 in zip(data, data[1:, ])  # differente between upload downloads
         if (row1[0] == PKT_UPLOAD and row2[0] == PKT_DOWNLOAD) or (
                 row2[0] == PKT_UPLOAD and row1[0] == PKT_DOWNLOAD)])
    

    p = [75, 90, 95, 98]
    for metric in [diff_pkts, diff_up_up, diff_down_down, diff_up_down]:
        
        if len(metric) == 0 :
            m1 = np.zeros(1)
            md1 = np.zeros(1)
            std1 = np.zeros(1)

            pr1 = np.zeros(len(p))
            
        else:
            m1 = np.mean(metric, axis=0)
            md1 = np.median(metric, axis=0)
            std1 = np.std(metric, axis=0)
            pr1 = np.array(np.percentile(metric, p, axis=0))
            

        features = np.hstack((m1, md1, std1,pr1))

        if 'obs_features' not in locals():
            obs_features = features.copy()
        else:
            obs_features = np.hstack((obs_features, features))

        
    return obs_features


def pktValidation(pkt):
    global scnets
    global ssnets
    

    timestamp, srcIP, dstIP = pkt.sniff_timestamp, pkt.ip.src, pkt.ip.dst

    if (IPAddress(srcIP) in scnets and IPAddress(dstIP) in ssnets) or (
            IPAddress(srcIP) in ssnets and IPAddress(dstIP) in scnets):


        if IPAddress(srcIP) in ssnets:  # download

            return PKT_DOWNLOAD, float(timestamp)

        if IPAddress(srcIP) in scnets:  # upload
            return PKT_UPLOAD, float(timestamp)


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', nargs='?', required=True, help='input file')
    parser.add_argument('-o', '--output', nargs='?', required=False, help='output file')
    parser.add_argument('-c', '--cnet', nargs='+', required=True, help='client network(s)')
    parser.add_argument('-s', '--snet', nargs='+', required=True, help='service network(s)')
    parser.add_argument('-w', '--width', nargs='?', required=False, help='observation windows width', default=20)
    parser.add_argument('-sd', '--slide', nargs='?', required=False, help='observation windows slide value', default=5)
    parser.add_argument('-f', '--file', nargs='?', required=True, help='file name')
    args = parser.parse_args()


    # sliding observation window
    lengthObsWindow = int(args.width)
    slidingValue = int(args.slide)
    file_name = args.file
    # client networks
    cnets = []
    for n in args.cnet:
        try:
            nn = IPNetwork(n)
            cnets.append(nn)
        except:
            print('{} is not a network prefix'.format(n))

    if len(cnets) == 0:
        print("No valid client network prefixes.")
        sys.exit()
    global scnets
    scnets = IPSet(cnets)
    # server networks
    snets = []

    for n in args.snet:
        try:
            nn = IPNetwork(n)
            snets.append(nn)
        except:
            print('{} is not a network prefix'.format(n))

    if len(snets) == 0:
        print("No valid service network prefixes.")
        sys.exit()

    global ssnets
    ssnets = IPSet(snets)
    file_input = args.input

    capture = pyshark.FileCapture(file_input, display_filter='tls', keep_packets=False)

    captureProcess(capture, lengthObsWindow, slidingValue,file_name)


if __name__ == '__main__':
    main()
