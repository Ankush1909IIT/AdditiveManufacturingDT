import numpy as np                     
import pyrealsense2 as rs             



Filename1 = "Different_Views_Multi_Coloured\Bunny_view_2.ply" # To save the point cloud data
Filename2 = "Different_Views_Multi_Coloured\Bunny_view_2_intermediate.ply" # To converst to ASCII format
Filename3 = "Different_Views_Multi_Coloured\Bunny_view_2_back_removed.ply" # To remove background
#Collect Data

print("Environment Ready")

pc = rs.pointcloud()
pipe = rs.pipeline()
#point = rs.points()

#Create a config and configure the pipeline to stream

config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

# Start streaming
profile = pipe.start(config)
align_to = rs.stream.color
align = rs.align(align_to)


frames = []
for x in range(2):
	frames = pipe.wait_for_frames()
	#frames = hole_filling.process(frames)
	#frames_filtered = threshold_filter.process(frames)
	
	aligned_frames = align.process(frames)

	# Get aligned frames
	aligned_depth_frame = aligned_frames.get_depth_frame() 
	color_frame = aligned_frames.get_color_frame()
	
	points = pc.calculate(aligned_depth_frame)
	pc.map_to(color_frame)
	
	print("Saving")
	points.export_to_ply(Filename1, color_frame)
	#points.export_to_ply("2.ply", frames_filtered)

print("Done")



#Convert to ASCII
# Load binary little endian PLY file
# Load PLY file and read header
with open(Filename1, 'rb') as f:
    # Read header
    header_lines = []
    while True:
        line = f.readline().decode('ascii')
        if line.strip() == 'format binary_little_endian 1.0':
            header_lines.append('format ascii 1.0\n')
            header_lines.append('\n')
        else:
            header_lines.append(line)
        if line.strip() == 'end_header':
            break
    
    # Read binary data into NumPy array
    data = np.fromfile(f, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('r', 'u1'), ('g', 'u1'), ('b', 'u1')])

# Convert to ASCII string
ascii_data = ''.join(header_lines) 
for item in data:
    ascii_data += str(item[0]) + ' ' + str(item[1]) + ' ' + str(item[2]) + ' ' + str(item[3]) + ' ' + str(item[4]) + ' ' + str(item[5]) + '\n'

# Write to file
with open(Filename2, 'w') as f:
    f.write(ascii_data)

print('Conversion to ASCII')

#Remove Background

with open(Filename2, 'r') as fp:
    x = len(fp.readlines())
    print('Total lines:', x) # 8

f = open(Filename2,'r')

g = open(Filename3,'w')

h = f.readlines()
for i in range(0,19,2):
    if i == 2:
        g.write('format ascii 1.0\n')
    else:
        g.write(h[i])
    #print(h[i])
g.write(h[24])
x = h[6].split()

no_of_entries = x[2]
no_of_entries = int(no_of_entries)
Counter = 0
for j in range (26,no_of_entries+26):
    All_info = h[j].split()
    x_dist = float(All_info[0])
    y_dist = float(All_info[1])
    z_dist = float(All_info[2])
    #g.write(h[j])
    #Counter = Counter + 1
    if z_dist < -0.26 and z_dist > -0.43 and x_dist < 0.16 and x_dist > -0.07  :
        g.write(h[j])
        Counter = Counter + 1


f.close()
g.close()

with open(Filename3, 'r') as file:
  
    # Reading the content of the file
    # using the read() function and storing
    # them in a new variable
    data = file.read()
  
    # Searching and replacing the text
    # using the replace() function
    data = data.replace(str(no_of_entries),str(Counter))

with open(Filename3, 'w') as file:
    file.write(data)
    # Writing the replaced data in our
    # text file
    
 
#print(Counter)
#print(no_of_entries)
#print(x)
print('Background Cleared')


