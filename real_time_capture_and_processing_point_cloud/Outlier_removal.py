# Code to segment out the outliers from the point cloud data
# this is a for a small region where the main object is present wth some outliers
# Parameters nb_neighbors and std_ratio is tunable
import open3d as o3d
import numpy as np


input_filename = 'Different_Views_Multi_Coloured\View_1_Markers.ply'
output_filename = 'Different_Views_Multi_Coloured\View_1_outlier_remover.ply'

# Define the input and output file paths

# Define statistical outlier removal parameters
#Got value by trail and error working good now these values
'''
# For individual
nb_neighbors = 2000  # Number of neighbors to consider for each point
std_ratio = 1.0  # Standard deviation ratio
'''

#For complete (with all three spheres)

nb_neighbors = 500  # Number of neighbors to consider for each point
std_ratio = 2.0  # Standard deviation ratio




header_lines = []
with open(input_filename, "r") as input_file:
    header_done = False
    for line in input_file:
        if not header_done:
            header_lines.append(line)
            if "end_header" in line:
                header_done = True
        else:
            break

# Load the point cloud data from the file, including the RGB color information
data = np.loadtxt(input_filename, skiprows=len(header_lines))
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(data[:, 0:3])
pcd.colors = o3d.utility.Vector3dVector(data[:, 3:6] / 255.0)

# Run the statistical outlier removal algorithm to remove outliers
pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)

# Write the filtered point cloud to a new file, preserving the header information and RGB color information
with open(output_filename, "w") as output_file:
    output_file.writelines(header_lines)
    np.savetxt(output_file, np.hstack([np.asarray(pcd.points), np.asarray(pcd.colors)*255.0]), fmt="%0.6f %0.6f %0.6f %d %d %d")

with open(output_filename, 'r+') as f:
    # Read the contents of the PLY file
    contents = f.readlines()

    # Count the number of lines in the file
    a = len(contents)

    # Find the index of the "end_header" line
    b = 0
    for i, line in enumerate(contents):
        if line.startswith('end_header'):
            b = i + 1
            break

    # Replace the third entry in the fourth line with a-b
    line_parts = contents[3].split()
    line_parts[2] = str(a - b)
    contents[3] = ' '.join(line_parts) + '\n'

    # Move the file pointer to the beginning and truncate the file
    f.seek(0)
    f.truncate()

    # Write the modified contents back to the file
    f.writelines(contents)


