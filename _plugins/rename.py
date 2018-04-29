import os
import string
import argparse

def rename(path,old_name,new_name):
    for file in os.listdir(path):
        tmp = file
        if tmp.find(old_name) != -1:
            tmp.replace(old_name,new_name)
            os.rename(os.path.join(path,old_name), os.path.join(path,tmp))
    
    print 'Done.'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Rename Files')
    parser.add_argument('path', metavar='PATH', type=str,help='Input file directory')
    parser.add_argument('old_name', metavar='NAME_OLD', type=str, help='Old file names')
    parser.add_argument('new_name', metavar='NAME_NEW', type=str, help='New file names')
    input_dir = parser.parse_args().path
    old_name = parser.parse_args().old_name
    new_name = parser.parse_args().new_name
    rename(path,old_name,new_name)
    