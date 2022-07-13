#!/usr/bin/env bash

#clear
#
#echo "Creating datasets for n clients:"
#
#gnome-terminal -e "python3 data/federated_data_extractor.py"
#
#sleep 3

echo "Start federated learning on n clients:"
#gnome-terminal -e "python3 miner.py -g 1 -l 2"
gnome-terminal -x bash -c "python3 SBS1.py; read"

sleep 3

for i in `seq 1 2`;
        do
                echo "Start client $i"
#                gnome-terminal -e "python3 client.py -d \"data/federated_data_$i.d\" -e 1"
                gnome-terminal -x bash -c "python3 clients.py -i $i; read"
#                gnome-terminal -x bash -c "python3 client.py -d \"data/federated_data_$i.d\" -e 1; read"
        done

#sleep 3

#gnome-terminal -e "python3 create_csv.py"
