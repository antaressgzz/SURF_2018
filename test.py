import psutil,time
import os

def getProcess(pName, level=5):
    process_lst = []

    # 获取当前系统所有进程id列表
    all_pids  = psutil.pids()

    for pid in all_pids:
        try:
            p = psutil.Process(pid)
            if p.cpu_percent(interval=0.1) > level:
                process_lst.append(p.pid)
        except:
            pass

    return process_lst

def reallocate_cpu(process_lst):
    cpu_num = psutil.cpu_count()
    # if cpu_num == 4:
#     cpus = ['0-1', '1-2', '2-3', '4-5', '5-6', '6-7' , '8-9', '9-10', '10-11', '12-13', '13-14', '14-15']
    cpus = ['0-1', '2-3', '4-5', '6-7', '8-9', '10', '11', '12', '13', '14', '15']
    for i in range(len(process_lst)):
        print('taskset -cp '+cpus[i]+ ' ' + str(process_lst[i]))
        os.system('taskset -cp '+cpus[i]+ ' ' + str(process_lst[i]))


if __name__ == '__main__':
    while True:
        print('cpu_usage:', psutil.cpu_percent(interval=1))
        if psutil.cpu_percent(interval=1) <= 90:
            process_lst = getProcess('Python', 10)
            print('process_lis:', process_lst)
            reallocate_cpu(process_lst)
        time.sleep(300)