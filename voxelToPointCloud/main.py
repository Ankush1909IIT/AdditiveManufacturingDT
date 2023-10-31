import numpy as np
from VoxelGrid import VoxelGrid
from os.path import join
import sys
if __name__ == '__main__':
    
    path = 'data/bunny/voxRes_64/input_files'    # path to the folder containing *Level1InOut.raw, *Level1Normal.raw and *VoxelConfig.txt
    new_bunny       = False  # True or False
    plotPC_show     = False # True or False
    plotVox_show    = False # True or False
    showAxis_voxelPlot = False # True or False
    voxmodfilename  = 'bunnyLevel1InOut.raw' 
    cfgfilename     = 'bunnyVoxelConfig.txt'
    
    if len(sys.argv) > 1:
        printHeight = int(sys.argv[1])
    else:
        printHeight = -1
        
    voxgrid = VoxelGrid(join(path, voxmodfilename), join(path, cfgfilename))
    voxgrid.readConfig()
    voxgrid.printConfig()
    
    if new_bunny:
        voxgrid.RotateModby90('x')
    
    pcdensity = 10
    # voxgrid.genPointCloud(pcdensity, printHeight, plotPC_show)
    voxgrid.plotVoxelModel(printHeight, plotVox_show, not showAxis_voxelPlot)
    voxgrid.saveInOutVoxModel(join(path, 'voxmodel.npz'))