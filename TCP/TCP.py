# encoding=utf-8
import threading
from socket import *
import sys

sys.path.append("..")

import TCPdata  # import TCP.TCPdata as TCPdata
from time import ctime
import time

host = '127.0.0.1'
port = 8085
bufsiz = 2048 * 32  # 2048 * 16
addr = (host, port)

# ss = socket() # 创建服务器套接字
# ss.bind() # 把地址绑定到套接字上
# ss.listen() # 监听连接
# inf_loop: # 服务器无限循环
# cs = ss.accept() # 接受客户的连接
# comm_loop: # 通讯循环
# cs.recv()/cs.send() # 对话（接收与发送）
# cs.close() # 关闭客户套接字
# ss.close() # 关闭服务器套接字（可选）

# 创建tcp套接字，绑定，监听
tcpServerSock = socket(AF_INET, SOCK_STREAM)  # 创建TCP Socket
# AF_INET 服务器之间网络通信
# socket.SOCK_STREAM 流式socket , for TCP
tcpServerSock.bind(addr)  # 将套接字绑定到地址,
# 在AF_INET下,以元组（host,port）的形式表示地址.
tcpServerSock.listen(5)  # 操作系统可以挂起的最大连接数量，至少为1，大部分为5


#  输入推力
def input_control():
    global left, right

    while True:
        temp = input('Left:')
        left = temp
        temp = input('Right:')
        right = temp


while True:
    print('waiting for connection')
    # udp中是recvfrom(buffersize),tcp这里用accept()；
    # tcp这里接收到的是客户端的sock对象，后面接受数据时使用socket.recv()
    tcpClientSock, addr2 = tcpServerSock.accept()  # 接受客户的连接
    # 接受TCP连接并返回（conn,address）,其中conn是新的套接字对象，
    # 可以用来接收和发送数据。
    # address是连接客户端的地址。
    print('connected from :', addr2)
    left = 0
    right = 0
    # t1 = threading.Thread(target=input_control, name='T1')  # 输入推力控制
    # t1.start()

    while True:
        data = tcpClientSock.recv(bufsiz)  # 接收客户端发来的数据
        if not data:
            break
        # 接收数据
        ReceveData = data.decode()
        distance_terminal, posture, distance, warning, block, overturn, reach, velocity, destination_angle = TCPdata.deal(ReceveData)

        reset = 0
        msg = str('123') + ',' + str(left) + ',' + str(right) + ',' + str(0) + ',' + str(0) + ',' + \
              str(reach) + ',' + str(reset) + ',' + str('321')  # 测试传入UE4控制量是否正常

        tcpClientSock.send(msg.encode())  # 返回给客户端数据

    tcpClientSock.close()
tcpServerSock.close()
