#!/usr/bin/python -tt
"""
Python script to prompt the user with a number of applications. Then allow selecting the configuration.
"""
import os
from py_utils.helpers import convertStr
from py_utils.parameter_struct import ParameterStruct
import pdb

def main():
    #application selection
    ls_prompts=['application','config']
    str_application_dir = ''
    print 'application/configuration selector, use q to break...\n'
    int_sel=''
    str_path=os.path.dirname(os.path.realpath(__file__)) + '/'
    for ix_,prompt in enumerate(ls_prompts):
        print 'Select ' + prompt + '...\n'
        str_path+=prompt+'/'
        ls_options = list(set(os.listdir(str_path)).difference(set(['data','scripts','main.py'])))
        for index, option in enumerate(ls_options):
            print str(index) + ': ' + option
        int_sel=convertStr(raw_input('Select number, q to quit: '))
        if int_sel == 'q':
            return
        if int_sel!='error' and int_sel < len(ls_options) and int_sel >= 0:
            str_path+=ls_options[int_sel]+'/'
        else:
            print 'invalid selection, quitting...'
            return
    str_path=str_path[0:-1] 
    print 'selected: ' + str_path
    if '_all.ini' in str_path: #spawn lots of jobs in parallel
        print 'spawning parameter sweep configs from this master config'
        ps_params = ParameterStruct(str_path)
        ls_file_names = ps_params.generate_configs()
        for file_name in ls_file_names:
            os.system(ls_prompts[0]+'/main.py ' + file_name)
    
if __name__ == "__main__":
    main()