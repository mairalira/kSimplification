def make_folder(folderName):
    # Create the new folder
    parent_folder = "img/"
    relative_folder_location = parent_folder + folderName
    os.makedirs(relative_folder_location, exist_ok=True)
