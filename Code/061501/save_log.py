# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 09:22:29 2018

@author: Administrator
"""

import sys

def save_log():
    
    # make a copy of original stdout route
    stdout_backup = sys.stdout
    # define the log file that receives your log info
    log_file = open("message.log", "w")
    # redirect print output to log file
    sys.stdout = log_file
    
    print("Now all print info will be writtendddd to message.log")
    # any command line that you will execute
    #...
    
    log_file.close()
    
    # restore the output to initial pattern
    sys.stdout = stdout_backup

print("Now this will be presented on screen")