import sys
import argparse
from netaddr import IPNetwork, IPAddress, IPSet
import pyshark


def pktHandler(timestamp, srcIP, dstIP, outfile):
    pass

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('-i', '--input', nargs='?',required=True, help='input file')
    parser.add_argument('-o', '--output', nargs='?',required=False, help='output file')
    parser.add_argument('-c', '--cnet', nars='?',required=True, help='client network(s)')
    parser.add_argument('-s', '--snet', nargs='+',required=True, help='service network(s)')

    args = parser.parse_args()

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
    global scnets  # set of client networks
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

    global ssnets  # set of server networks
    ssnets = IPSet(snets)

    fileInput = args.input

    if args.output is None:
        fileOutput = fileInput + ".dat"
    else:
        fileOutput = args.output

    global npkts
    global T0
    global outc
    global last_ks

    npkts=0
    outc=[0,0,0,0]

    outfile = open(fileOutput, 'w')

    # Filtering

    capture = pyshark.FileCapture(fileInput, display_filter='ip tls', keep_packets=False)  # por cada pacote na captura
    for pkt in capture:
        timestamp, srcIP, dstIP = pkt.sniff_timestamp, pkt.ip.src, pkt.ip.dst

        if (IPAddress(srcIP) in scnets and IPAddress(dstIP) in ssnets) or (
            IPAddress(srcIP) in ssnets and IPAddress(dstIP) in scnets):
            # só nos interesa o timestamp, srcIP, dstIP
            pktHandler(timestamp, srcIP, dstIP, outfile)

    outfile.close()

if __name__ == '__main__':
    main()