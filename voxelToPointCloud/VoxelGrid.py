import numpy as np
import os
class VoxelGrid:
    
    def __init__(self, voxmodpath, cfgpath):
        self.voxmodpath = voxmodpath
        self.cfgpath    = cfgpath
        self.modelname  = None
        self.bboxmin    = np.zeros(3, dtype=np.float32)
        self.bboxmax    = np.zeros(3, dtype=np.float32)
        self.numvoxels  = np.zeros(3, dtype=np.int32)
        self.voxsize    = np.zeros(3, dtype=np.float32)
        self.numvoxels_inside   = None
        self.numvoxels_boundary = None
        self.voxmodel   = None
        self.voxVal_in  = 127
        self.voxVal_bdry= 254
        self.voxVal_out = 0
        self.pointCloud = None
        self.order     = 'F'    # 'C' for C-style row-major array, 'F' for Fortran-style column-major array
        
        self.readConfig()
        self.readVoxModel()
        
    def readConfig(self):
        ''' 
        example config file:
        bunny                               # name of the model
        -31.5624	-47.9706	-9.13192    # bbox min
        30.0255	31.1302	69.2861             # bbox max
        200	256	256                         # number of voxels in each dimension
        0.307939	0.308987	0.30632     # voxel size
        3127372                             # number of inside voxels
        223144                              # number of boundary voxels
        '''
        ''' 
        Get the count of the type of voxels in the voxel model
        0   : outside
        127 : inside
        254 : boundary
        '''
        
        cfgfile = open(self.cfgpath, 'r')
        cfglines = cfgfile.readlines()
        cfgfile.close()
        
        self.modelname = cfglines[0].strip()                                            # name of the model
        self.bboxmin = np.array([float(x) for x in cfglines[1].strip().split('\t')])    # box min and max
        self.bboxmax = np.array([float(x) for x in cfglines[2].strip().split('\t')])    # box max
        self.numvoxels = np.array([int(x) for x in cfglines[3].strip().split('\t')])    # number of voxels in each dimension
        self.voxsize = np.array([float(x) for x in cfglines[4].strip().split('\t')])    # voxel size
        self.numvoxels_inside = int(cfglines[5].strip())                                # total number of voxels
        self.numvoxels_boundary = int(cfglines[6].strip())                              # total number of voxels
        
    def printConfig(self):
        print('Model name               : {}'.format(self.modelname))
        print('Bounding box min         : {}'.format(self.bboxmin))
        print('Bounding box max         : {}'.format(self.bboxmax))
        print('Number of voxels         : {}'.format(self.numvoxels))
        print('Voxel size               : {}'.format(self.voxsize))
        print('Number of inside voxels  : {}'.format(self.numvoxels_inside))
        print('Number of boundary voxels: {}'.format(self.numvoxels_boundary))
        
    def readVoxModel(self):
        flatarray = np.fromfile(self.voxmodpath, dtype=np.uint8)
        
        # Convert the flat array to a 3D array with the dimensions specified in the config file using numpy.reshape
        self.voxmodel = flatarray.reshape(self.numvoxels[0], self.numvoxels[1], self.numvoxels[2], order=self.order)
        print('Voxel model shape: {}'.format(self.voxmodel.shape))
        
    def saveInOutVoxModel(self, savepath):
        
        # Create a voxel model with in (1) and out (0) values. In include boundary voxels
        # Get max dimension of the voxel model
        maxdim = np.max(self.numvoxels)
        
        # inOutVoxModel = np.zeros((maxdim, maxdim, maxdim), dtype=np.uint8)
        inOutVoxModel = np.zeros(self.numvoxels, dtype=np.uint8)
        
        # Print Dimensions of the voxel model and the inOutVoxModel
        print('Voxel model shape: {}'.format(self.voxmodel.shape))      # (24, 32, 32)
        print('inOutVoxModel shape: {}'.format(inOutVoxModel.shape))    # (32, 32, 32)
        
        # Set the values of the inOutVoxModel to 1 for inside and boundary voxels
        inOutVoxModel[:self.numvoxels[0], :self.numvoxels[1], :self.numvoxels[2]] = self.voxmodel
        inOutVoxModel[inOutVoxModel != self.voxVal_out] = 1
        
        # take mirror image of the inOutVoxModel across the x-z plane
        inOutVoxModel = np.flip(inOutVoxModel, axis=1)
        
        # save as npz file
        np.savez(savepath, voxmodel=inOutVoxModel)
        
    def RotateModby90(self, axis):
        ''' Rotate the voxel model by 90 degrees about the specified axis '''
        
        if axis == 'x':
            self.voxmodel   = np.rot90(self.voxmodel, 1, (1, 2))
            self.numvoxels  = np.array([self.numvoxels[0], self.numvoxels[2], self.numvoxels[1]])
            self.voxSize    = np.array([self.voxsize[0], self.voxsize[2], self.voxsize[1]])
        elif axis == 'y':
            self.voxmodel  = np.rot90(self.voxmodel, 1, (0, 2))
            self.numvoxels = np.array([self.numvoxels[2], self.numvoxels[1], self.numvoxels[0]])
            self.voxsize   = np.array([self.voxsize[2], self.voxsize[1], self.voxsize[0]])
        elif axis == 'z':
            self.voxmodel   = np.rot90(self.voxmodel, 1, (0, 1))
            self.numvoxels  = np.array([self.numvoxels[1], self.numvoxels[0], self.numvoxels[2]])
            self.voxsize    = np.array([self.voxsize[1], self.voxsize[0], self.voxsize[2]]) 
        else:
            print('Invalid axis specified. Please specify x, y or z')
        return
        

    def genPointCloud(self, pcDensity=1, height=-1, plotShow=False):
        # Generate a point cloud from the voxel model
        unique, counts = np.unique(self.voxmodel, return_counts=True)
        print('Voxel counts             : {}'.format(dict(zip(unique, counts))))
        
        height = self.checkHeight(height)
        
        boundaryvoxels = np.argwhere(self.voxmodel[:, :, :height] == self.voxVal_bdry)  # Get the indices of the boundary voxels
        numboundaryvoxels = boundaryvoxels.shape[0]                                     # Get the number of boundary voxels
            
        # Generate a point cloud from the boundary voxels by sampling n points from each boundary voxel
        self.pointCloud = np.zeros((numboundaryvoxels*pcDensity, 3))
        for i in range(numboundaryvoxels):
            self.pointCloud[i*pcDensity:(i+1)*pcDensity, :] = self.bboxmin + boundaryvoxels[i, :] * self.voxsize + np.random.rand(pcDensity, 3) * self.voxsize
        print('Point cloud shape        : {}'.format(self.pointCloud.shape))
        
        if plotShow:
            # Plot the point cloud
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure(figsize=(60, 60))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(self.pointCloud[:, 0], self.pointCloud[:, 1], self.pointCloud[:, 2], s=0.7)
            # Show all the axes 
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            plt.show()
            plt.close()

        # Save the point cloud to a file in the same directory as the voxel model file
        pcpath = os.path.join(os.path.dirname(self.voxmodpath), self.modelname + '_pc_{}.xyz'.format(height))
            
        np.savetxt(pcpath, self.pointCloud, fmt='%.6f')
        print('Point cloud saved to {}'.format(pcpath))
        
    def plotVoxelModel(self, height=-1, plotShow=False, noAxes=True):
        # Plot the voxel model upto a specified height
        height = self.checkHeight(height)
        
        xlim = [0, self.numvoxels[0]]
        ylim = [0, self.numvoxels[1]]
        zlim = [0, self.numvoxels[2]]
        
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(60, 60))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.voxels(self.voxmodel[:, :, :height], edgecolor='k', facecolors='w')
        if noAxes:
            color_tuple = (1.0, 1.0, 1.0, 0.0)
            ax.w_xaxis.set_pane_color(color_tuple)
            ax.w_yaxis.set_pane_color(color_tuple)
            ax.w_zaxis.set_pane_color(color_tuple)
            ax.w_xaxis.line.set_color(color_tuple)
            ax.w_yaxis.line.set_color(color_tuple)
            ax.w_zaxis.line.set_color(color_tuple)

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
        else: 
            ax.set_xlabel('X')  # Show all the axes
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            
        ax.set_box_aspect((1, 1, 1))
        ax.auto_scale_xyz(xlim, ylim, zlim)
        ax.set_proj_type('ortho')
        # ax.view_init(elev=5, azim=180)
        plt.tight_layout()
        plotpath = os.path.join(os.path.dirname(self.voxmodpath), self.modelname + '_voxelmodel_{}.png'.format(height))
        plt.savefig(plotpath)        # Save the plot
        print('Voxel model plot saved to {}'.format(plotpath))
        if plotShow:
            plt.show()            # Show the plot
        plt.close()
        
    def checkHeight(self, height):
        if height == -1:
            height = self.numvoxels[2]
        elif height > self.numvoxels[2]:
            print('Specified height is greater than the voxel model height. Setting height to voxel model height')
            height = self.numvoxels[2]
        elif height < 0:
            print('Specified height is less than 0. Setting height to 1')
            height = 1
            
        return height