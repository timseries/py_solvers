#!/usr/bin/python -tt
"""
Python script to prompt the user with a number of applications. Then allow selecting the configuration.
"""
import os
from py_utils.helpers import convertStr
import pdb

def main():
    #application selection
    ls_app_option=[]
    ls_prompts=['application','config']
    str_application_dir = ''
    print 'application/configuration selector, use q to break...\n'
    int_sel=''
    str_path = os.path.dirname(os.path.realpath(__file__)) + '/' + str_application_dir
    file_tuple_numbered = enumerate(os.walk(str_path))
    for app_index,file_tuple in file_tuple_numbered:
        print 'Select ' + ls_prompts[app_index] + '...\n'
        if app_index==0:
            ls_options = list(set(file_tuple[1]).difference(set(['data'])))
        else:
            pdb.set_trace()
            ls_options = [[file_dir+'/'+filename for filename in file_tuple[2]]
                          for file_dir in file_tuple[1] if file_dir=='config']
        for index, option in enumerate(ls_options):
            print str(index) + ': ' + option
        int_sel = convertStr(raw_input('Select number, q to quit: '))
        if int_sel == 'q':
            return
        if int_sel != 'error' and int_sel < len(ls_options) and int_sel >= 0:
            if str_application_dir=='':  
                str_application_dir = ls_options[int_sel]
            else:    
                str_configuration = ls_options[int_sel]
                lgc_valid_entry=1
        else:
            print 'invalid selection, try again...'
    int_sel=prompt_from_options(level)
    if lgc_valid_entry:
        str_full_config_path = str_path + '/' + str_configuration
        print 'selected: ' + str_full_config_path
        os.system(str_path + 'main.py ' + str_full_config_path)
    
if __name__ == "__main__":
    main()