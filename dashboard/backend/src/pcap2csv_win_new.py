# pcap2csv_win.py [Convert PCAP to CSV]
# Minimal CIC-style flow features from a PCAP (Windows-friendly, no tcpdump)
# Requires: scapy (you already have it)

import argparse, csv, math, statistics, time, threading, signal, sys, os
from scapy.all import PcapReader, IP, IPv6, TCP, UDP, sniff
import numpy as np


BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data"))  # <-- Add this

flows = {}
running = True
written_ids = set()



#Statistical Helper Functions(To prevent mathematical errors)
def safe_mean(x):
    return statistics.fmean(x) if x else 0.0

def safe_std(x):
    return statistics.pstdev(x) if len(x) > 1 else 0.0


def safe_div(a, b): 
    return a / b if b != 0 else 0.0 #To check if safe division


#To calculate percentiles for various metrics to provide distribution insights beyond mean/std
def pctile(lst, q):
    return float(np.percentile(lst, q)) if lst else 0.0


#It takes a list of packet timestamps as input and returns the metrics below.This helps in revealing burst patters, detects retransmission/Out of order etc.
#Small Gaps->Bursty Traffic(HTTP requests or video streaming, Large Gaps->Idle periods,delayed responses, Variable Gaps->Congested Network or mixed traffic types)
def iat_stats(times):
    if len(times) < 2: return (0,0,0,0)
    gaps = [t2 - t1 for t1,t2 in zip(times[:-1], times[1:])]
    return (safe_mean(gaps), safe_std(gaps), min(gaps), max(gaps))


#Provides complete statistical distribution of iat, including percentiles for detailed analysis
def iat_all(times):
    if len(times) < 2: return (0,0,0,0,0,0,0,0)
    gaps = [t2 - t1 for t1,t2 in zip(times[:-1], times[1:])]
    return (
        safe_mean(gaps), safe_std(gaps), min(gaps), max(gaps),
        pctile(gaps,25), pctile(gaps,50), pctile(gaps,75), pctile(gaps,90)
    )

#Total sum of iat [total time span covered by the gaps between packets (excluding the actual packet processing time).]
def total_iat(times):
    if len(times) < 2: return 0.0
    gaps = [t2 - t1 for t1,t2 in zip(times[:-1], times[1:])]
    return sum(gaps)



#Separates network activity into bursts (active) and pauses (idle) using a threshold (1.0 second default).
def active_idle_stats(times, threshold=1.0):
    #Sort timestamps 
    if len(times) < 2:
        return (0,0,0,0,0,0,0,0)
    times_sorted = sorted(times)
    gaps = [t2 - t1 for t1, t2 in zip(times_sorted[:-1], times_sorted[1:])]

    #Identify gaps > threshold as idle periods , Periods between idle gaps are active bursts
    actives, idles = [], []
    cur_start = times_sorted[0]
    for g in gaps:
        if g <= threshold:
            continue
        else:
            actives.append(times_sorted[gaps.index(g)] - cur_start)
            idles.append(g)
            cur_start = times_sorted[gaps.index(g)+1]
    # last active
    if cur_start != times_sorted[-1]:
        actives.append(times_sorted[-1] - cur_start)

    #Calculate statistics for both active and idle periods
    def stats(lst):
        return (
            min(lst) if lst else 0,
            safe_mean(lst) if lst else 0,
            max(lst) if lst else 0,
            safe_std(lst) if lst else 0
        )

    min_act, mean_act, max_act, std_act = stats(actives)
    min_idle, mean_idle, max_idle, std_idle = stats(idles)

    return (min_act, mean_act, max_act, std_act,
            min_idle, mean_idle, max_idle, std_idle)


#Packet Length Extraction
def pkt_len(pkt):
    try:
        return len(bytes(pkt))
    except Exception:
        return 0

#IP Protocol Detection
def get_ip_layer(pkt):
    if IP in pkt:
        return pkt[IP], 'IPv4'
    if IPv6 in pkt:
        return pkt[IPv6], 'IPv6'
    return None, None

#Transport Layer Analysis->Identifies TCP/UDP Protocols,src/dst ports
def get_l4_info(pkt):
    if TCP in pkt: l4=pkt[TCP]; return 'TCP', int(l4.sport), int(l4.dport), l4.flags
    if UDP in pkt: l4=pkt[UDP]; return 'UDP', int(l4.sport), int(l4.dport), None
    return None,None,None,None


#Bidirectional Flow Key Generation
def make_bi_key(proto, a_ip, a_port, b_ip, b_port):
    # direction-agnostic 5-tuple key (so fwd/bwd live in one flow)
    a = (a_ip, a_port)
    b = (b_ip, b_port)
    if a <= b:
        return (proto, a, b)
    else:
        return (proto, b, a)
    

# ---------------- Packet Processing ----------------
def process_packet(src, dst, sport, dport, proto, ts, length, flags):
    key = make_bi_key(proto, src, sport, dst, dport)
    f = flows.get(key)
    if f is None:
        f = {"src":src,"dst":dst,"sport":sport,"dport":dport,"proto":proto,
             "start":ts,"end":ts,
             "fwd_times":[ts],"bwd_times":[],
             "fwd_lens":[length],"bwd_lens":[],
             "fwd_flags":[],"bwd_flags":[]}
        if flags is not None: f["fwd_flags"].append(flags)
        flows[key] = f
    else:
        f["end"] = max(f["end"], ts)
        if src == f["src"] and dst == f["dst"] and sport == f["sport"] and dport == f["dport"]:
            f["fwd_times"].append(ts); f["fwd_lens"].append(length)
            if flags is not None: f["fwd_flags"].append(flags)
        else:
            f["bwd_times"].append(ts); f["bwd_lens"].append(length)
            if flags is not None: f["bwd_flags"].append(flags)


# ---------------- Live Capture Handlers ----------------
def handle_packet(pkt):
    ip, _ = get_ip_layer(pkt)
    if ip is None: return
    proto, sport, dport, flags = get_l4_info(pkt)
    if proto is None: return
    process_packet(ip.src, ip.dst, sport, dport, proto, float(pkt.time), pkt_len(pkt), flags)

def dump_flows_to_csv(filename):
    # Ensure output is in DATA_DIR
    if not os.path.isabs(filename):
        filename = os.path.join(DATA_DIR, filename)

    headers=[  # same headers as pcap2csv_win.py
        "FlowID","SrcIP","DstIP","SrcPort","DstPort","Protocol",
        "FlowDuration",
        "TotFwdPkts","TotBwdPkts","TotLenFwd","TotLenBwd",
        "FwdPktLenMean","FwdPktLenStd","FwdPktLenMin","FwdPktLenMax",
        "BwdPktLenMean","BwdPktLenStd","BwdPktLenMin","BwdPktLenMax",
        "FlowIATMean","FlowIATStd","FlowIATMin","FlowIATMax",
        "FwdIATMean","FwdIATStd","FwdIATMin","FwdIATMax",
        "BwdIATMean","BwdIATStd","BwdIATMin","BwdIATMax",
        "TotalFwdIAT","TotalBwdIAT",
        "TotalBytes","TotalPackets",
        "BytesPerSec","PktsPerSec","FwdBwdPktRatio","FwdBwdByteRatio",
        "FwdPktLenPct25","FwdPktLenPct50","FwdPktLenPct75","FwdPktLenPct90",
        "BwdPktLenPct25","BwdPktLenPct50","BwdPktLenPct75","BwdPktLenPct90",
        "FlowIAT25","FlowIAT50","FlowIAT75","FlowIAT90",
        "FwdIAT25","FwdIAT50","FwdIAT75","FwdIAT90",
        "BwdIAT25","BwdIAT50","BwdIAT75","BwdIAT90",
        "Fwd_SYN","Fwd_FIN","Fwd_RST","Fwd_PSH","Fwd_ACK","Fwd_URG",
        "Bwd_SYN","Bwd_FIN","Bwd_RST","Bwd_PSH","Bwd_ACK","Bwd_URG",
        "MinActive","MeanActive","MaxActive","StdActive",
        "MinIdle","MeanIdle","MaxIdle","StdIdle",
        "SrcPortCat","DstPortCat"
    ]
    with open(filename,"w",newline="",encoding="utf-8") as fcsv:
        w=csv.DictWriter(fcsv,fieldnames=headers); w.writeheader()
        for idx,(key,fl) in enumerate(flows.items(),start=1):
            dur=max(0.0,fl["end"]-fl["start"])
            all_times=sorted(fl["fwd_times"]+fl["bwd_times"])
            flow_iat=iat_stats(all_times)
            fwd_iat=iat_stats(fl["fwd_times"])
            bwd_iat=iat_stats(fl["bwd_times"])
            fwd_len_p=[pctile(fl["fwd_lens"],q) for q in (25,50,75,90)]
            bwd_len_p=[pctile(fl["bwd_lens"],q) for q in (25,50,75,90)]
            flow_iat_p=iat_all(all_times)
            fwd_iat_p=iat_all(fl["fwd_times"])
            bwd_iat_p=iat_all(fl["bwd_times"])
            total_bytes=sum(fl["fwd_lens"])+sum(fl["bwd_lens"])
            total_pkts=len(fl["fwd_lens"])+len(fl["bwd_lens"])
            bytes_per_sec=safe_div(total_bytes,dur)
            pkts_per_sec=safe_div(total_pkts,dur)
            pkt_ratio=safe_div(len(fl["fwd_lens"]),len(fl["bwd_lens"]))
            byte_ratio=safe_div(sum(fl["fwd_lens"]),sum(fl["bwd_lens"]))
            total_fiat=total_iat(fl["fwd_times"])
            total_biat=total_iat(fl["bwd_times"])
            act_idle=active_idle_stats(all_times)
            def count_flags(flags_list,mask): return sum(1 for f in flags_list if f & mask)
            fwd_syn=count_flags(fl["fwd_flags"],0x02)
            fwd_fin=count_flags(fl["fwd_flags"],0x01)
            fwd_rst=count_flags(fl["fwd_flags"],0x04)
            fwd_psh=count_flags(fl["fwd_flags"],0x08)
            fwd_ack=count_flags(fl["fwd_flags"],0x10)
            fwd_urg=count_flags(fl["fwd_flags"],0x20)
            bwd_syn=count_flags(fl["bwd_flags"],0x02)
            bwd_fin=count_flags(fl["bwd_flags"],0x01)
            bwd_rst=count_flags(fl["bwd_flags"],0x04)
            bwd_psh=count_flags(fl["bwd_flags"],0x08)
            bwd_ack=count_flags(fl["bwd_flags"],0x10)
            bwd_urg=count_flags(fl["bwd_flags"],0x20)
            def port_cat(p):
                if p in (80,443): return "Web"
                if p in (1935,554,8554): return "Multimedia"
                if p in (5222,5228,443): return "Social"
                if p<1024: return "System"
                return "Other"
            row={
                "FlowID":idx,
                "SrcIP":fl["src"],"DstIP":fl["dst"],
                "SrcPort":fl["sport"],"DstPort":fl["dport"],"Protocol":fl["proto"],
                "FlowDuration":dur,
                "TotFwdPkts":len(fl["fwd_lens"]), "TotBwdPkts":len(fl["bwd_lens"]),
                "TotLenFwd":sum(fl["fwd_lens"]), "TotLenBwd":sum(fl["bwd_lens"]),
                "FwdPktLenMean":safe_mean(fl["fwd_lens"]), "FwdPktLenStd":safe_std(fl["fwd_lens"]),
                "FwdPktLenMin":min(fl["fwd_lens"],default=0), "FwdPktLenMax":max(fl["fwd_lens"],default=0),
                "BwdPktLenMean":safe_mean(fl["bwd_lens"]), "BwdPktLenStd":safe_std(fl["bwd_lens"]),
                "BwdPktLenMin":min(fl["bwd_lens"],default=0), "BwdPktLenMax":max(fl["bwd_lens"],default=0),
                "FlowIATMean":flow_iat[0],"FlowIATStd":flow_iat[1],
                "FlowIATMin":flow_iat[2],"FlowIATMax":flow_iat[3],
                "FwdIATMean":fwd_iat[0],"FwdIATStd":fwd_iat[1],
                "FwdIATMin":fwd_iat[2],"FwdIATMax":fwd_iat[3],
                "BwdIATMean":bwd_iat[0],"BwdIATStd":bwd_iat[1],
                "BwdIATMin":bwd_iat[2],"BwdIATMax":bwd_iat[3],
                "TotalFwdIAT":total_fiat,"TotalBwdIAT":total_biat,
                "TotalBytes":total_bytes,"TotalPackets":total_pkts,
                "BytesPerSec":bytes_per_sec,"PktsPerSec":pkts_per_sec,
                "FwdBwdPktRatio":pkt_ratio,"FwdBwdByteRatio":byte_ratio,
                "FwdPktLenPct25":fwd_len_p[0],"FwdPktLenPct50":fwd_len_p[1],
                "FwdPktLenPct75":fwd_len_p[2],"FwdPktLenPct90":fwd_len_p[3],
                "BwdPktLenPct25":bwd_len_p[0],"BwdPktLenPct50":bwd_len_p[1],
                "BwdPktLenPct75":bwd_len_p[2],"BwdPktLenPct90":bwd_len_p[3],
                "FlowIAT25":flow_iat_p[4],"FlowIAT50":flow_iat_p[5],
                "FlowIAT75":flow_iat_p[6],"FlowIAT90":flow_iat_p[7],
                "FwdIAT25":fwd_iat_p[4],"FwdIAT50":fwd_iat_p[5],
                "FwdIAT75":fwd_iat_p[6],"FwdIAT90":fwd_iat_p[7],
                "BwdIAT25":bwd_iat_p[4],"BwdIAT50":bwd_iat_p[5],
                "BwdIAT75":bwd_iat_p[6],"BwdIAT90":bwd_iat_p[7],
                "Fwd_SYN":fwd_syn,"Fwd_FIN":fwd_fin,"Fwd_RST":fwd_rst,
                "Fwd_PSH":fwd_psh,"Fwd_ACK":fwd_ack,"Fwd_URG":fwd_urg,
                "Bwd_SYN":bwd_syn,"Bwd_FIN":bwd_fin,"Bwd_RST":bwd_rst,
                "Bwd_PSH":bwd_psh,"Bwd_ACK":bwd_ack,"Bwd_URG":bwd_urg,
                "MinActive":act_idle[0],"MeanActive":act_idle[1],
                "MaxActive":act_idle[2],"StdActive":act_idle[3],
                "MinIdle":act_idle[4],"MeanIdle":act_idle[5],
                "MaxIdle":act_idle[6],"StdIdle":act_idle[7],
                "SrcPortCat":port_cat(fl["sport"]), "DstPortCat":port_cat(fl["dport"])
            }
            w.writerow(row)
    print(f"[+] Updated {filename} with {len(flows)} flows")

def periodic_dump(filename, interval=30):
    while running:
        time.sleep(interval)
        dump_flows_to_csv(filename)

def signal_handler(sig, frame):
    global running
    running = False
    print("\n[!] Stopping capture...")
    dump_flows_to_csv(os.path.join(DATA_DIR, "final_liveflows.csv"))  # <-- Use DATA_DIR
    sys.exit(0)


def main():
    ap = argparse.ArgumentParser(description="PCAP/Live -> CSV (CIC-like flow features, Windows-friendly)")
    ap.add_argument("-i","--input", help="Input PCAP file")
    ap.add_argument("-o","--output", required=True, help="Output CSV file")
    ap.add_argument("--live", action="store_true", help="Enable live capture mode")
    ap.add_argument("--iface", default="Wi-Fi", help="Network interface for live capture")
    args = ap.parse_args()

    if args.live:
        signal.signal(signal.SIGINT, signal_handler)
        print(f"[*] Sniffing on {args.iface}... Press Ctrl+C to stop.")
        threading.Thread(target=periodic_dump, args=(args.output,), daemon=True).start()
        sniff(iface=args.iface, prn=handle_packet, store=False)
    else:
        # Ensure input/output are in DATA_DIR
        input_pcap = args.input
        if not os.path.isabs(input_pcap):
            input_pcap = os.path.join(DATA_DIR, input_pcap)
        output_csv = args.output
        if not os.path.isabs(output_csv):
            output_csv = os.path.join(DATA_DIR, output_csv)
        with PcapReader(input_pcap) as pr:
            for pkt in pr:
                ip, _ = get_ip_layer(pkt)
                if ip is None: continue
                proto, sport, dport, flags = get_l4_info(pkt)
                if proto is None: continue
                process_packet(ip.src, ip.dst, sport, dport, proto, float(pkt.time), pkt_len(pkt), flags)
        dump_flows_to_csv(output_csv)

if __name__=="__main__":
    main()





