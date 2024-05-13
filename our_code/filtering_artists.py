# Step 1: Count occurrences of each artist
artist_counts = df['artist'].value_counts()

# Step 3: Filter the dataset to keep only new_filenames
# 500 ÉS MASA POQUES IMATGES, potser millor fer 100
artists_above_99 = artist_counts[artist_counts >= 99].index
filtered_df = df[df['artist'].isin(artists_above_99)]

artist_counts

#%%
# AQUÍ SE'T CREARÀ UNA NOVA CARPETA ON MOUREM ELS FITXERS DELS ARTISTES CORRESPONENTS
# Keep files in directories of the filtered artists
import shutil
import os

# Define the input folder
input_folder = r"C:\Users\Mercè\Documents\UAB\XN\Projecte\input"

# Define the output folder where you want to move the files
output_folder = r"C:\Users\Mercè\Documents\UAB\XN\Projecte\input_new"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Iterate over the rows in the filtered DataFrame
for index, row in filtered_df.iterrows():
    # Extract the new_filename from the DataFrame
    new_filename = row['new_filename']
    
    # Determine the subdirectory of the file (train_1 or test)
    # TORNAR A EXECUTAR PER TEST <--------------------------------------------
    source_subdirectory = "train_1" 
    # source_subdirectory = "test"

    # Search for the file in the input folder and its subdirectories
    for root, dirs, files in os.walk(os.path.join(input_folder, source_subdirectory)):
        if new_filename in files:
            # Get the full path of the file
            file_path = os.path.join(root, new_filename)
            
            # Determine the destination subdirectory
            destination_subdirectory = os.path.join(output_folder, source_subdirectory)
            
            # Create the destination subdirectory if it doesn't exist
            os.makedirs(destination_subdirectory, exist_ok=True)
            
            # Define the destination path where you want to move the file
            destination_path = os.path.join(destination_subdirectory, new_filename)
            
            # Move the file to the output folder
            shutil.move(file_path, destination_path)


#%%

# globals

DATA_DIR = r"C:\Users\Mercè\Documents\UAB\XN\Projecte\input_new" # '/home/xnmaster/Project/input'
TRAIN_1_DIR =  r"C:\Users\Mercè\Documents\UAB\XN\Projecte\input_new\train_1\train_1" # '/home/xnmaster/Project/input/train_1'

TRAIN_DIRS = [TRAIN_1_DIR]
TEST_DIR = r"C:\Users\Mercè\Documents\UAB\XN\Projecte\input_new\test\test" # '/home/xnmaster/Project/input/test'


#%%
# Get the list of files in the folder
files_in_folder = os.listdir(TRAIN_1_DIR)

# Count the number of files in the folder
num_files_in_folder = len(files_in_folder)

print("Number of files in the folder TRAIN_1_DIR:", num_files_in_folder)


files_in_folder = os.listdir(TEST_DIR)

# Count the number of files in the folder
num_files_in_folder = len(files_in_folder)

print("Number of files in the folder TEST_DIR:", num_files_in_folder)

