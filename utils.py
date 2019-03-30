import base64
import logging
import urllib.request
import numpy as np
import os
import json
import time
from shutil import copyfile
from config import *

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')


def url_quote(data):
    return urllib.parse.quote(data).replace("/", "%2F")


def authenticated_http_req(url, user, pwd):
    request = urllib.request.Request(url)
    # print(url)
    payload = '%s:%s' % (user, pwd)
    base64string = base64.b64encode(payload.encode('utf-8')).decode("utf-8")
    request.add_header('Authorization', 'Basic %s' % base64string)
    return request


def json_get_req(url):
    try:
        request = authenticated_http_req(url, ONOS_USER, ONOS_PASS)
        response = urllib.request.urlopen(request)
        return json.loads(response.read().decode("utf-8"))
    except IOError as e:
        logging.error(e)
        return ''


def json_post_req(url, json_data):
    try:
        request = authenticated_http_req(url, ONOS_USER, ONOS_PASS)
        request.add_header('Content-Type', 'application/json')
        # json to bytes
        data = bytes(json_data, encoding='utf-8')
        print(data)
        response = urllib.request.urlopen(request, data=data)
        return json.loads(response.read().decode("utf-8"))
    except IOError as e:
        logging.error(e)
        return ''


def bps_to_human_string(value, to_byte_per_second=False):
    if to_byte_per_second:
        value = value/8.0
        suffix = 'B/s'
    else:
        suffix = 'bps'

    for unit in ['', 'K', 'M', 'G']:
        if abs(value) < 1000.0:
            return '%3.1f %s%s' % (value, unit, suffix)
        value /= 1000.0
    return '%.1f %s%s' % (value, 'T', suffix)


def scale(array):
    mean = array.mean()
    std = array.std()
    if std == 0:
        std = 1
    return np.asarray((array - mean)/std)


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def selu(x):
    from keras.activations import elu
    """Scaled Exponential Linear Unit. (Klambauer et al., 2017)

    # Arguments
        x: A tensor or variable to compute the activation function for.

    # References
        - [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
    """
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * elu(x, alpha)


def pretty(f):
    try:
        float(f)
        return str.format('{0:.3f}', f).rstrip('0').rstrip('.')
    except:
        return str(f)


def setup_exp(experiment=''):
    folder = FOLDER
    os.makedirs(folder, exist_ok=True)
    folder += experiment + '/'
    os.makedirs(folder, exist_ok=True)
    return folder


# 初始化运行，构建run文件夹，创建folder.ini及DDPG.json
def setup_run():
    # folder = runs
    folder = FOLDER
    epoch = 't%.6f/' % time.time()
    folder += epoch.replace('.', '')
    # 创建文件夹runs/时间序列
    os.makedirs(folder, exist_ok=True)
    # 在run中生成一个与时间相关的文件
    # 在里面写入folder.ini
    # 每次用w模式打开文件, 都会清空这个文件(坑)
    with open(folder + 'folder.ini', 'w') as ifile:
        ifile.write('[General]\n')
        ifile.write('**.folderName = "' + folder + '"\n')
    copyfile('./config.py', folder + "/config.py")
    return folder


# refer to
def criteria_type_key_to_value_key(argument):
    switcher = {
        'ETH_SRC': 'mac',
        'ETH_DST': 'mac',
        'ETH_TYPE': 'ethType',
        'IPV4_SRC': 'ip',
        'IPV4_DST': 'ip',
        'IP_PROTO': 'protocol',
        'TCP_SRC': 'tcpPort',
        'TCP_DST': 'tcpPort',
        'UDP_SRC': 'udpPort',
        'UDP_DST': 'udpPort',
    }
    return switcher.get(argument, "")


def criteria_type_key_to_self_key(argument):
    switcher = {
        'ETH_SRC': 'ETH_SRC',
        'ETH_DST': 'ETH_DST',
        'ETH_TYPE': 'ETH_TYPE',
        'IPV4_SRC': 'IP_SRC',
        'IPV4_DST': 'IP_DST',
        'IPV6_SRC': 'IP_SRC',
        'IPV6_DST': 'IP_DST',
        'IP_PROTO': 'IP_PROTO',
        'TCP_SRC': 'PORT_SRC',
        'TCP_DST': 'PORT_DST',
        'UDP_SRC': 'PORT_SRC',
        'UDP_DST': 'PORT_DST',
    }
    return switcher.get(argument, "")

