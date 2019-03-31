import logging
import numpy as np
import networkx as nx
import os
import json
import random
from config import *
from utils import bps_to_human_string,url_quote,json_post_req,criteria_type_key_to_self_key, criteria_type_key_to_value_key, json_get_req, pretty, softmax


def matrix_to_onos_v(matrix):
    # 将矩阵扁平化
    return matrix.flatten()


def vector_to_file(vector, file_name, action):
    string = ','.join(pretty(_) for _ in vector)
    with open(file_name, action) as file:
        return file.write(string + '\n')


DEVICES = 'Devices.txt'
PORTS = 'Ports.txt'
LOADS = 'Loads.txt'
WIRES = 'Wires.txt'
EMBEDDINGINPUT = 'EmbeddingInput.txt'
EMBEDDINGOUTPUT = 'EmbeddingOutput.txt'


class ONOSEnv():
    def __init__(self, folder):
        self.folder = folder
        self.G = nx.Graph()
        self.active_nodes = 0
        self.node_embeddinged = []
        self.devices = []
        self.deviceId_to_arrayIndex = {}
        self.env_ports = []
        self.env_loads = []
        self.env_wires = []
        self.hosts = {}
        self.tracked_intent = {}
        self.set_up()

    def set_up(self):
        self.update_device()
        self.update_links()
        self.update_host()
        self.node_embedding()
        self.chose_intent()
        if len(self.devices) == 0 \
                or len(self.env_loads) == 0 \
                or len(self.tracked_intent.keys()) == 0:
            print("Set Up Error!")

    def node_embedding(self):
        self.node_embeddinged = np.full([self.active_nodes, REPESENTATTION_SIZE], 0.0, dtype=float)
        output = open(self.folder + EMBEDDINGINPUT, 'w')
        lines = []
        for s in range(self.active_nodes):
            for d in range(self.active_nodes):
                if self.env_wires[s][d] != -1:
                    line = "%s %s %s \n" % (s, d, self.env_wires[s][d])
                    lines.append(line)
        output.writelines(lines)
        output.close()
        root = os.getcwd() + "/" + self.folder
        embeddingInput = root + EMBEDDINGINPUT
        embeddingOutput = root + EMBEDDINGOUTPUT
        cmdline = "sudo python %s " \
                  "--method %s " \
                  "--input %s " \
                  "--graph-format edgelist " \
                  "--output %s " \
                  "--representation-size %s " \
                  "--workers %s " \
                  "--window-size %s " \
                  "--weighted " \
                  % (EMBEDDING_TOOL_DIR,
                     EMBEDDING_METHOD,
                     embeddingInput,
                     embeddingOutput,
                     REPESENTATTION_SIZE,
                     EMBEDDING_WORKERS,
                     EMBEDDING_WINDOWS_SIZE)
        # print(cmdline)
        result = os.popen(cmdline).read()
        if 'Error'in result:
            return
        else:
            protect_count = 0
            while True:
                protect_count += 1
                if os.path.isfile(embeddingOutput) or protect_count > 10000:
                    break
            if not os.path.isfile(embeddingOutput):
                print("embedding error")
                return
            file_input = open(embeddingOutput)
            line1_splits = file_input.readline().split(" ")
            if int(line1_splits[0]) == self.active_nodes and int(line1_splits[1]) == REPESENTATTION_SIZE:
                    for i in range(self.active_nodes):
                        line = file_input.readline()
                        split1 = line.split(" ", 1)
                        index = int(split1[0])
                        features_temp = split1[1].split(" ")
                        features = []
                        for feature in features_temp:
                            features.append(float(feature))
                        self.node_embeddinged[index] = features
        print("onde embedding successfully")
        return

    # need myself onos apps traffic-tracker
    def update_links(self):
        # init loads and wires
        self.env_loads = np.full([self.active_nodes] * 2, -1.0, dtype=float)
        self.env_wires = np.full([self.active_nodes] * 2, -1.0, dtype=float)
        self.env_ports = np.full([self.active_nodes] * 2, -1.0, dtype=float)
        reply = json_get_req('http://%s:%d/onos/v1/tm/tm/getLinksLoad' % (ONOS_IP, ONOS_PORT))
        if 'links' not in reply:
            return
        for link in reply['links']:
            src_port = link['src']['port']
            # dst_port = link['dst']['port']
            src = link['src']['device']
            src_index = self.deviceId_to_arrayIndex[src]
            dst = link['dst']['device']
            dst_index = self.deviceId_to_arrayIndex[dst]
            wire = link['wire']
            load = link['load']
            rest = link['rest']
            self.G.add_edge(src, dst, **{'wire': wire, 'load': load, 'bandwidth': rest})
            self.env_loads[src_index][dst_index] = load
            self.env_wires[src_index][dst_index] = wire
            self.env_ports[src_index][dst_index] = src_port
        # vector_to_file(matrix_to_onos_v(self.env_loads), self.folder + LOADS, 'w')
        vector_to_file(matrix_to_onos_v(self.env_wires), self.folder + WIRES, 'w')
        # vector_to_file(matrix_to_onos_v(self.env_ports), self.folder + PORTS, 'w')
        print("links update successfully")
        return

    def update_device(self):
        logging.info("Retrieving Topology...")
        reply = json_get_req('http://%s:%d/onos/v1/devices' % (ONOS_IP, ONOS_PORT))
        if 'devices' not in reply:
            return
        self.active_nodes = 0
        for dev in reply['devices']:
            # id is 'of:00000000000000a1',..., arrayIndex is 0,......
            self.deviceId_to_arrayIndex[dev['id']] = self.active_nodes
            self.G.add_node(dev['id'], type='device')
            self.devices.append(dev['id'])
            self.active_nodes += 1
        vector_to_file(self.devices, self.folder + DEVICES, 'w')
        print("device update successfully")
        return

    def update_host(self):
        reply = json_get_req('http://%s:%d/onos/v1/hosts' % (ONOS_IP, ONOS_PORT))
        if 'hosts' not in reply:
            return
        for host in reply['hosts']:
            self.G.add_node(host['id'], type='host')
            for location in host['locations']:
                self.G.add_edge(host['id'], location['elementId'], **{'bandwidth': DEFAULT_ACCESS_CAPACITY})
            self.hosts[host['id']] = host
        print("host update successfully")
        return

    # radmon chose a intent ro track,until get a intent with a ip protocol
    def chose_intent(self):
        reply = json_get_req('http://%s:%d/onos/v1/intents' % (ONOS_IP, ONOS_PORT))
        if 'intents' not in reply:
            return
        intents = reply['intents']
        if len(intents) == 0:
            print("no one intent")

        # chose a intent withe IP_PROTO
        while True:
            # radmon chose one intent
            intent_index = random.randint(0, len(intents)-1)
            intent = intents[intent_index]
            if intent['state'] != 'INSTALLED':
                continue

            # org.onosproject.eifwd
            self.tracked_intent['appId'] = intent['appId']
            # 56:60:C7:C8:CD:7B/None-72:06:C3:73:5C:A5/None
            self.tracked_intent['key'] = intent['key']
            self.tracked_intent['url_key'] = url_quote(intent['key'])

            # get host locations
            self.update_src_dst_locations()
            if 'src_locations' not in self.tracked_intent:
                continue

            # radom chose a switch in locations as src and dst location
            self.update_src_dst_location()
            if 'src_location' not in self.tracked_intent:
                continue

            # update intent information includes ip,port,protocol,flowid,deviceid
            self.update_tracked_intent()
            monitored = self.monitor_intent()
            if 'IP_PROTO' in self.tracked_intent and monitored:
                break
        print("chose intent successfully")
        return

    # need myself application eimr
    def monitor_intent(self):
        msg = dict()
        msg['name'] = self.tracked_intent['appId']
        msg['intentKey'] = self.tracked_intent['key']

        result = json_post_req(('http://%s:%d/onos/v1/eimr/eimr/startMonitorIntent'
                                % (ONOS_IP, ONOS_PORT)), json.dumps(msg))
        if 'Failed' in json.dumps(result):
            return False
        return True

    # the key of intent is src_host_mac-dst_host_mac
    def update_src_dst_locations(self):
        if 'key' not in self.tracked_intent:
            return
        splited_macs = self.tracked_intent['key'].split('-')
        src_mac = splited_macs[0]
        dst_mac = splited_macs[1]
        # check host mac exits
        if src_mac not in self.hosts or dst_mac not in self.hosts:
            return
        src_locations = self.hosts[src_mac]['locations']
        dst_locations = self.hosts[dst_mac]['locations']
        # check location if exits
        if src_locations is None or dst_locations is None\
                or len(src_locations) == 0 or len(dst_locations) == 0:
            return
        self.tracked_intent['src_locations'] = src_locations
        self.tracked_intent['dst_locations'] = dst_locations
        return

    # radom chose one location of locations as src_location and dst_location
    # location= {elementId,port} elementId==host connect deviceId,
    # embeding info = node_embeddinged[location['elementId']]
    def update_src_dst_location(self):
        src_index = random.randint(0, len(self.tracked_intent['src_locations']) - 1)
        dst_index = random.randint(0, len(self.tracked_intent['dst_locations']) - 1)
        self.tracked_intent['src_location'] = self.tracked_intent['src_locations'][src_index]
        self.tracked_intent['dst_location'] = self.tracked_intent['dst_locations'][dst_index]

    # update intent information includes ip,port,protocol,flowid,deviceid
    def update_tracked_intent(self):
        req_str = 'http://%s:%d/onos/v1/intents/relatedflows/%s/%s' \
                  % (ONOS_IP,
                     ONOS_PORT,
                     self.tracked_intent['appId'],
                     self.tracked_intent['url_key'])
        reply = json_get_req(req_str)
        if len(reply['paths']) == 0:
            return
        intent_flow = reply['paths'][0][0]
        self.tracked_intent['device_id'] = intent_flow['deviceId']
        self.tracked_intent['flow_id'] = intent_flow['id']
        criteria = intent_flow['selector']['criteria']
        for attribute in criteria:
            type = attribute['type']
            # print(type)
            value_key = criteria_type_key_to_value_key(type)
            self_key = criteria_type_key_to_self_key(type)
            if value_key != '' and self_key != '':
                self.tracked_intent[self_key] = attribute[value_key]
        return

    #
    def update_intent_load(self):
        req_str = 'http://%s:%d/onos/v1/eimr/eimr/intentLoad/%s/%s' \
                   % (ONOS_IP,
                      ONOS_PORT,
                      self.tracked_intent['appId'],
                      self.tracked_intent['url_key'])
        reply = json_get_req(req_str)
        if 'load' not in reply:
            return
        load = reply['load']

        print(bps_to_human_string(load))
        return


