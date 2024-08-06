import os


def pc_sleep():
    os.system("rundll32.exe powrprof.dll, SetSuspendState 0,1,0")


if __name__ == "__main__":
    pc_sleep()