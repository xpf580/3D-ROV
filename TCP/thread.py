# -*- coding: utf-8 -*-
# @Time    : 2022/3/16 13:49
# @Author  : 耀
import threading
import time
def thread_job():
    print("T1 start\n")
    for i in range(10):
        time.sleep(0.1)
    print('T1 finish\n' )

def T2_job():
    print('T2 start\n')
    print('T2 finish\n')

def main():
    added_thread=threading.Thread(target=thread_job,name='T1')#添加线程,target是这个线程要做的工作,target=后面的括号不能加
    added_thread.start()#激活线程
    thread2=threading.Thread(target=T2_job,name='T2')
    thread2.start()  # 激活线程
    thread2.join()
    added_thread.join()#等待所有线程运行完之后再运行
    print('all done')



#     print(threading.active_count())#当前激活了几个线程
#     print(threading.enumerate())#当前激活线程的名字
#     print(threading.current_thread())#当前运行程序的线程名字
if __name__=='__main__':
    main()