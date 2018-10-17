import os
import psutil,time

def getProcess(pName, level=5):
    process_lst = []

    # 获取当前系统所有进程id列表
    all_pids  = psutil.pids()

    # 遍历所有进程，名称匹配的加入process_lst
    for pid in all_pids:
        p = psutil.Process(pid)
        if p.name() == pName and p.cpu_percent(interval=1) > level:
            process_lst.append(p.pid)

    return process_lst

def reallocate_cpu(process_lst):
    cpu_num = psutil.cpu_count()
    # if cpu_num == 4:
    cpus = ['0-1', '1-2', '2-3', '4-5', '5-6', '6-7', '8-9', '9-10', '10-11', '12-13', '13-14', '14-15']
    for i in range(len(process_lst)):
        os.system('teskset -cp '+cpus[i]+ ' ' + process_lst[i])


if __name__ == '__main__':
    time.sleep(60)
    while True:
        if psutil.cpu_percent() <= 70:
            process_lst = getProcess('Python', 8)
            reallocate_cpu(process_lst)
            time.sleep(300)