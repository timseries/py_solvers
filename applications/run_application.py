#!/usr/bin/python -tt
"""
Python script to prompt the user with a number of applications. Then allow selecting the configuration.
"""
import os
from py_utils.helpers import convertStr

def main():
    #application selection
    ls_app_option=[]
    ls_two_prompts=['application','configuration']
    str_application_dir = ''
    print 'application/configuration selector, use q to break...\n'
    for str_prompt in ls_two_prompts:
        print 'Select ' + str_prompt + '...\n'
        lgc_valid_entry = 0
        ls_possible_options = []
        while not lgc_valid_entry:
            str_path = os.path.dirname(os.path.realpath(__file__)) + '/' + str_application_dir
            for dirs, dirnames, filenames in os.walk(str_path):
                if str_application_dir=='':  
                    options = dirnames
                else:
                    options = filenames    
                for index, option in enumerate(options):
                    print str(index) + ': ' + option
                    ls_possible_options.append(option)
            int_sel = convertStr(raw_input('Select ' + str_prompt +  ' number, q to quit: '))
            if int_sel == 'q':
                break;
            if int_sel != 'error' and int_sel < len(ls_possible_options) and int_sel >= 0:
                if str_application_dir=='':  
                    str_application_dir = ls_possible_options[int_sel]
                else:    
                    str_configuration = ls_possible_options[int_sel]
                lgc_valid_entry=1
            else:
                print 'invalid selection, try again...'

    str_full_config_path = str_path + '/' + str_configuration
    print 'selected: ' + str_full_config_path
    os.system(str_path + 'main.py ' + str_full_config_path)
    
if __name__ == "__main__":
    main()