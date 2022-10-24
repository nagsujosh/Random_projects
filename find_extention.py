import os
import shutil

dict_extensions = {
    'audio_extensions': ('.mp3', '.m4a', '.wav', '.flac'),
    'videos_extensions': ('.mp4', '.mkv', '.MKV', '.flv', '.mpeg'),
    'documents_extensions': ('.doc', '.pdf', '.txt'),
    'image_extensions': ('.png', '.jpg')
}

folderpath = input("Enter Folder Path:")


def file_finder(folder_path, file_extension):
    return [file for file in os.listdir(folder_path) for extension in file_extension
            if file.endswith(extension)]
for extension_type, extension_tuple in dict_extensions.items():
    folder_name = extension_type.split('_')[0] + 'File'
    folder_path = os.path.join(folderpath, folder_name)
    os.mkdir(folder_path)
    for item in file_finder(folderpath, extension_tuple):
        item_path = os.path.join(folderpath, item)
        item_new_path = os.path.join(folder_path,item)
        shutil.move(item_path, item_new_path)