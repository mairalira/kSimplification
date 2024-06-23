import os


def make_folder(folderName):
    # Create the new folder
    relative_folder_location = folderName
    os.makedirs(relative_folder_location, exist_ok=True)
