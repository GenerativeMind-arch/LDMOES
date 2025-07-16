import os
import shutil
def search_files(dir_path):

    file_list = os.listdir(dir_path)
    for file_name in file_list:
        complete_file_name = os.path.join(dir_path, file_name)
        if os.path.isdir(complete_file_name):
            search_files(complete_file_name)
        if os.path.isfile(complete_file_name):
            print(complete_file_name)
            shutil.copyfile(complete_file_name,"./churches/"+file_name)
if __name__ == '__main__':
    list1 = search_files('./imgs')

