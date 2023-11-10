import torch.nn as nn
import torch
import math
import pdb
import numpy as np

def get_feature_leveller(feshape, num_channels, out_nodes=16, sf=1):
        ### YET TO IMPLEMENT group PARMAETER FEATURE: where feshape*num_channels reduced to num_channel branches with one output node each
    #     #### Can get specific shape inputs from the initModel function in utils.py to create apporpriate last FC layers for the feature extractor.
            return nn.Sequential(
                nn.Linear(int(feshape) * num_channels, out_nodes*num_channels),
                nn.ELU(inplace=True)
            )

# def patchify(images, n_patches= 16, slicewise=False):
#     n, c, h, w, d = images.shape
    
#     if slicewise:
#         output = 

#     assert h == w, "Patchify method is implemented for square images only"

#     patches = torch.zeros(n, n_patches ** 2, h * w * c // n_patches ** 2)
#     patch_size = h // n_patches

#     for idx, image in enumerate(images):
#         for i in range(n_patches):
#             for j in range(n_patches):
#                 patch = image[:, i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size]
#                 patches[idx, i * n_patches + j] = patch.flatten()
#     return patches


class MyViT(nn.Module):
  def __init__(self, num_channels=50, num_branches=1):
    # Super constructor
    super(MyViT, self).__init__()
    
    #Attributes
    self.num_channels = num_channels
    self.num_branches = num_branches
    
    
    # linear map
    self.levelers = nn.ModuleList([ self.get_feature_leveller(self.feshapes[bi], num_channels, out_nodes=self.feshapes[bi]) for bi in range(num_branches)])
    
    # class token
    self.class_token = nn.Parameter(torch.rand(1, num_branches))
    
    


  def forward(self, images):
    return





class ResNet_l3(nn.Module):

    def __init__(self, block, layers, num_classes):
        self.inplanes = 64
        super(ResNet_l3, self).__init__()
        self.conv1 = nn.Conv3d(1, 64, kernel_size=3, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool3d(3)
        self.fc = nn.Linear(9216 * block.expansion, 512)
        self.fc2 = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)        
        x = self.layer1(x)        
        x = self.layer2(x)        
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x1 = self.fc(x)
        x = self.fc2(x1)
        return [x,x1]


class BrASLnet(nn.Module):
    
    def __init__(self, num_classes=2, num_channels=50, num_groups=1, num_branches=1, sf=1, shapes=None, feshapes=None, skip_clf=False):
        super(BrASLnet, self).__init__()
        print("nc=%d, nch=%d, ngr=%d"%(num_classes,num_channels,num_groups))
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.num_groups = num_groups
        self.num_branches = num_branches
        self.sf = sf
        self.shapes = shapes
        self.feshapes = feshapes
        self.skip_clf = skip_clf
        self.features = nn.ModuleList([self.gen_feature_extractor(num_channels,num_groups,self.shapes[0][bi]) for bi in range(num_branches)])
        self.levelers = nn.ModuleList([ self.get_feature_leveller(self.feshapes[bi], num_channels) for bi in range(num_branches)])
        self.kernel_merger = nn.ModuleList([ self.get_feature_leveller(self.sf*1*num_channels, num_channels, out_nodes=1) for bi in range(num_branches)])
        
        self.filter = nn.ELU(inplace=False)
        self.clf_part1 = nn.Sequential(nn.Dropout(),
                                        nn.Linear(self.num_branches*1 *num_channels, 64*num_channels),
                                        nn.ELU(inplace=False),
                                        nn.Dropout(),
                                        nn.Linear(64*num_channels, self.num_branches),
                                        nn.ELU(inplace=False),
                                        nn.Dropout())
        
        self.clf_part2 = nn.Sequential(nn.Linear(self.num_branches, self.num_classes))


        self.classifier = nn.Sequential(nn.Dropout(),
                                        nn.Linear(self.num_branches*1 *num_channels, 64*num_channels),
                                        nn.ELU(inplace=False),
                                        nn.Dropout(),
                                        nn.Linear(64*num_channels, 64),
                                        nn.ELU(inplace=False),
                                        nn.Dropout(),
                                        nn.Linear(64, num_classes),
                                        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def gen_feature_extractor(self,num_channels, num_groups, curshape, feshape=None):
        sf = self.sf
        
        # if curshape.prod() <= 2048 or np.any(curshape < 8):
        if curshape.sum() <= 32 or np.any(curshape < 8):
            ks, st = 3,2
        # elif curshape.prod() <= 4096 or np.any(curshape < 12):
        elif curshape.sum() <= 64 or np.any(curshape < 12):
            ks, st = 5,3
        else:
            ks, st = 7,3
        ks = tuple(3 + (curshape // 10))
        pd = (1,1,1)
        st = tuple(1 + (curshape // 8))
        return nn.Sequential(
            nn.Conv3d(num_channels, sf*2*num_channels, kernel_size=ks, stride=1, padding=pd, groups=num_groups),
            nn.BatchNorm3d(sf*2*num_channels),
            nn.ELU(inplace=True),
            # nn.MaxPool3d(kernel_size=3, stride=1),

            nn.Conv3d(sf*2*num_channels, sf*4*num_channels, kernel_size=ks, stride=1, padding=pd, groups=num_groups),
            nn.BatchNorm3d(sf*4*num_channels),
            nn.ELU(inplace=True),
            # nn.MaxPool3d(kernel_size=3, stride=1),

            nn.Conv3d(sf*4*num_channels, sf*6*num_channels, kernel_size=ks, padding=pd, groups=num_groups),
            nn.BatchNorm3d(sf*6*num_channels),
            nn.ELU(inplace=True),

            nn.Conv3d(sf*6*num_channels, sf*6*num_channels, kernel_size=ks, padding=pd, groups=num_groups),
            nn.BatchNorm3d(sf*6*num_channels),
            nn.ELU(inplace=True),
            # nn.MaxPool3d(kernel_size=3, stride=1),

            nn.Conv3d(sf*6*num_channels, sf*1*num_channels, kernel_size=ks, padding=pd, groups=num_groups),
            nn.BatchNorm3d(sf*1*num_channels),
            nn.ELU(inplace=True),
            # nn.MaxPool3d(kernel_size=ks, stride=st),
            nn.AdaptiveMaxPool3d(1), 
            
            nn.Flatten(start_dim=1, end_dim = -1),
            nn.Linear(sf*1*num_channels, 1),
            nn.ELU(inplace=True)
        )
        
    def get_feature_leveller(self, feshape, num_channels, out_nodes=16, sf=1):
        ### YET TO IMPLEMENT group PARMAETER FEATURE: where feshape*num_channels reduced to num_channel branches with one output node each
    #     #### Can get specific shape inputs from the initModel function in utils.py to create apporpriate last FC layers for the feature extractor.
            return nn.Sequential(
                nn.Linear(int(feshape) * num_channels, out_nodes*num_channels),
                nn.ELU(inplace=True)
            )
            
        
    def forward(self, x):
        
        nsub, nch, nf = x.shape
        xp = []
        for i, br in enumerate(self.features):
            # ch here is not channel of the network, the fkey index in fkeys (or group in the CNN)
            # This is in case the shapes for different modalities are different.
            ch=0
            
            # Reshape flattened input image into multiple cuboidal images based on patches/ROIs/components.
            curshape = self.shapes[ch][i,:]
            imin = self.shapes[ch][0:i,:].prod(axis=1).sum()
            imax = imin + curshape.prod() 
            curx = torch.reshape(x[:,ch:ch+1,imin:imax], [nsub,1,curshape[0],curshape[1],curshape[2]])
            
            # Forward pass through feature extractor branches.
            xpi = br(curx)
            
            xpi = xpi.view(xpi.size(0),-1)
            ### Dense layer  (nch to 1) ###
            
            # xpi = self.levelers[i](xpi)
            xp.append(xpi)       
            
        # print([(j,xp[j].shape[-1]) for j in range(len(xp))])
        xp = torch.hstack(xp)
        identity = torch.clone(xp)
        # import pdb; pdb.set_trace()
        if self.skip_clf:
            out = self.clf_part1(xp)
            
            out += identity
            out = self.filter(out)
            out = self.clf_part2(out)
        else:
            out = self.classifier(xp)
        # return x 
        return [out, xp]
    


class meanASLnet(nn.Module):
    
    def __init__(self, num_classes=2, num_channels=50, num_groups=1, num_branches=1, sf=1, shapes=None, feshapes=None):
        super(meanASLnet, self).__init__()
        print("nc=%d, nch=%d, ngr=%d"%(num_classes,num_channels,num_groups))
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.num_groups = num_groups
        self.num_branches = num_branches
        self.sf = sf
        self.shapes = shapes
        self.feshapes = feshapes
        self.features = nn.ModuleList([self.gen_feature_extractor(num_channels,num_groups,self.shapes[0][bi]) for bi in range(num_branches)])
        self.levelers = nn.ModuleList([ self.get_feature_leveller(self.feshapes[bi], num_channels) for bi in range(num_branches)])
        self.kernel_merger = nn.ModuleList([ self.get_feature_leveller(self.sf*1*num_channels, num_channels, out_nodes=1) for bi in range(num_branches)])
        

        self.classifier = nn.Sequential(nn.Dropout(),
                                        nn.Linear(self.num_branches*1 *num_channels, 64*num_channels),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(),
                                        # nn.Linear(64*num_channels, 64),
                                        # nn.ReLU(inplace=True),
                                        # nn.Dropout(),
                                        nn.Linear(64, num_classes),
                                        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def gen_feature_extractor(self,num_channels, num_groups, curshape, feshape=None):
        # sf = self.sf
        
        # # if curshape.prod() <= 2048 or np.any(curshape < 8):
        # if curshape.sum() <= 32 or np.any(curshape < 8):
        #     ks, st = 3,2
        # # elif curshape.prod() <= 4096 or np.any(curshape < 12):
        # elif curshape.sum() <= 64 or np.any(curshape < 12):
        #     ks, st = 5,3
        # else:
        #     ks, st = 7,3
        # ks = tuple(3 + (curshape // 10))
        # pd = (1,1,1)
        # st = tuple(1 + (curshape // 8))
        
        return nn.Sequential(
            nn.AdaptiveAvgPool3d(1)
            
            # nn.Flatten(start_dim=1, end_dim = -1),
            # nn.Linear(sf*1*num_channels, 1),
            # nn.ReLU(inplace=True)
        )
        
    def get_feature_leveller(self, feshape, num_channels, out_nodes=16, sf=1):
        ### YET TO IMPLEMENT group PARMAETER FEATURE: where feshape*num_channels reduced to num_channel branches with one output node each
    #     #### Can get specific shape inputs from the initModel function in utils.py to create apporpriate last FC layers for the feature extractor.
            return nn.Sequential(
                nn.Linear(int(feshape) * num_channels, out_nodes*num_channels),
                nn.ReLU(inplace=True)
            )
            
        
    def forward(self, x):
        nsub, nch, nf = x.shape
        xp = []
        for i, br in enumerate(self.features):
            # ch here is not channel of the network, the fkey index in fkeys (or group in the CNN)
            # This is in case the shapes for different modalities are different.
            ch=0
            curshape = self.shapes[ch][i,:]
            
            imin = self.shapes[ch][0:i,:].prod(axis=1).sum()
            imax = imin + curshape.prod() 
            
            curx = torch.reshape(x[:,ch:ch+1,imin:imax], [nsub,1,curshape[0],curshape[1],curshape[2]])
            xpi = br(curx)
            xpi = xpi.view(xpi.size(0),-1)
            ### Dense layer  (nch to 1) ###
            
            # xpi = self.levelers[i](xpi)
            xp.append(xpi)
            
        # print([(j,xp[j].shape[-1]) for j in range(len(xp))])
        xp = torch.hstack(xp)
        # import pdb; pdb.set_trace()
        x = self.classifier(xp)
        # return x 
        return [x, xp]
    



## 

class AN3Ddr_lowresAvg(nn.Module):
    def __init__(self, num_classes=2, num_channels=1):
        super(AN3Ddr_lowresAvg, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(num_channels, 64, kernel_size=5, stride=1, padding=0), #5 kernel and padding 0
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=3),

            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=0), #3 kernel and padding 0
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=1),

            nn.Conv3d(128, 192, kernel_size=3, padding=1),
            nn.BatchNorm3d(192),
            nn.ReLU(inplace=True),

            nn.Conv3d(192, 192, kernel_size=3, padding=1),
            nn.BatchNorm3d(192),
            nn.ReLU(inplace=True),

            nn.Conv3d(192, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d([1,1,1]),
        )
        self.classifier = nn.Sequential(nn.Dropout(),
                                        nn.Linear(128, 64),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(),
                                        nn.Linear(64, num_classes),
                                        )
        # self.classifier = nn.Sequential(nn.Dropout(),
        #                                 nn.Linear(2048, 256),
        #                                 nn.ReLU(inplace=True),
        #                                 nn.Dropout(),
        #                                 nn.Linear(256, 64),
        #                                 nn.ReLU(inplace=True),
        #                                 nn.Dropout(),
        #                                 nn.Linear(64, num_classes),
        #                                 )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def forward(self, x):
        xp = self.features(x)
        x = xp.view(xp.size(0), -1)
        x = self.classifier(x)
        return [x, xp]


class AN3Ddr_highresMax(nn.Module):
    def __init__(self, num_classes=2, num_channels=1):
        super(AN3Ddr_highresMax, self).__init__()
    
        self.features = nn.Sequential(
            nn.Conv3d(num_channels, 64, kernel_size=5, stride=2, padding=0),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=3),

            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=3),

            nn.Conv3d(128, 192, kernel_size=3, padding=1),
            nn.BatchNorm3d(192),
            nn.ReLU(inplace=True),

            nn.Conv3d(192, 192, kernel_size=3, padding=1),
            nn.BatchNorm3d(192),
            nn.ReLU(inplace=True),

            nn.Conv3d(192, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=3),
        )

        self.classifier = nn.Sequential(nn.Dropout(),
                                        nn.Linear(256, 64),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(),
                                        nn.Linear(64, num_classes),
            )


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x):
        xp = self.features(x)
        x = xp.view(xp.size(0), -1)
        x = self.classifier(x)
        # return x
        return [x, xp]


        



class AN3Ddr_lowresMax(nn.Module):
    def __init__(self, num_classes=2, num_channels=50, num_groups=1):
        super(AN3Ddr_lowresMax, self).__init__()
        print("nc=%d, nch=%d, ngr=%d"%(num_classes,num_channels,num_groups))
        self.features = nn.Sequential(
            nn.Conv3d(num_channels, 64*num_channels, kernel_size=5, stride=2, padding=2, groups=num_groups),
            nn.BatchNorm3d(64*num_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=3),

            nn.Conv3d(64*num_channels, 128*num_channels, kernel_size=3, stride=1, padding=2, groups=num_groups),
            nn.BatchNorm3d(128*num_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=3),

            nn.Conv3d(128*num_channels, 192*num_channels, kernel_size=3, padding=1, groups=num_groups),
            nn.BatchNorm3d(192*num_channels),
            nn.ReLU(inplace=True),

            nn.Conv3d(192*num_channels, 192*num_channels, kernel_size=3, padding=1, groups=num_groups),
            nn.BatchNorm3d(192*num_channels),
            nn.ReLU(inplace=True),

            nn.Conv3d(192*num_channels, 128*num_channels, kernel_size=3, padding=1, groups=num_groups),
            nn.BatchNorm3d(128*num_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=3),
        )

        self.classifier = nn.Sequential(nn.Dropout(),
                                        # nn.Linear(2048*num_channels, 256*num_channels),
                                        # nn.ReLU(inplace=True),
                                        # nn.Dropout(),
                                        nn.Linear(128*num_channels, 64*num_channels),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(),
                                        # nn.Linear(64*num_channels, 64),
                                        # nn.ReLU(inplace=True),
                                        # nn.Dropout(),
                                        nn.Linear(64, num_classes),
                                        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x):
        xp = self.features(x)
        x = xp.view(xp.size(0), -1)
        # import pdb; pdb.set_trace()
        x = self.classifier(x)
        # return x
        return [x, xp]





class AN3Ddr_lowresMaxLight(nn.Module):
    def __init__(self, num_classes=2, num_channels=50, num_groups=1):
        super(AN3Ddr_lowresMaxLight, self).__init__()
        print("nc=%d, nch=%d, ngr=%d"%(num_classes,num_channels,num_groups))
        self.features = nn.Sequential(
            nn.Conv3d(num_channels, 32*num_channels, kernel_size=5, stride=2, padding=0, groups=num_groups),
            nn.BatchNorm3d(32*num_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=3),

            nn.Conv3d(32*num_channels, 64*num_channels, kernel_size=3, stride=1, padding=0, groups=num_groups),
            nn.BatchNorm3d(64*num_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=1),

            nn.Conv3d(64*num_channels, 96*num_channels, kernel_size=3, padding=1, groups=num_groups),
            nn.BatchNorm3d(96*num_channels),
            nn.ReLU(inplace=True),

            nn.Conv3d(96*num_channels, 96*num_channels, kernel_size=3, padding=1, groups=num_groups),
            nn.BatchNorm3d(96*num_channels),
            nn.ReLU(inplace=True),

            nn.Conv3d(96*num_channels, 64*num_channels, kernel_size=3, padding=1, groups=num_groups),
            nn.BatchNorm3d(64*num_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=1),
        )

        self.classifier = nn.Sequential(nn.Dropout(),
                                        nn.Linear(1024*num_channels, 256*num_channels),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(),
                                        nn.Linear(256*num_channels, 64*num_channels),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(),
                                        nn.Linear(64*num_channels, 64),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(),
                                        nn.Linear(64, num_classes),
                                        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x):
        xp = self.features(x)
        x = xp.view(xp.size(0), -1)
        x = self.classifier(x)
        return [x, xp]






class AN3Ddr_lowresMaxLight_ASL(nn.Module):
    def __init__(self, num_classes=2, num_channels=50, num_groups=1):
        super(AN3Ddr_lowresMaxLight_ASL, self).__init__()
        print("nc=%d, nch=%d, ngr=%d"%(num_classes,num_channels,num_groups))
        self.features = nn.Sequential(
            nn.Conv3d(num_channels, 8*num_channels, kernel_size=5, stride=1, padding=0, groups=num_groups),
            nn.BatchNorm3d(8*num_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),

            nn.Conv3d(8*num_channels, 16*num_channels, kernel_size=3, stride=1, padding=0, groups=num_groups),
            nn.BatchNorm3d(16*num_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),

            # nn.Conv3d(64*num_channels, 96*num_channels, kernel_size=3, padding=1, groups=num_groups),
            # nn.BatchNorm3d(96*num_channels),
            # nn.ReLU(inplace=True),

            # nn.Conv3d(96*num_channels, 96*num_channels, kernel_size=3, padding=1, groups=num_groups),
            # nn.BatchNorm3d(96*num_channels),
            # nn.ReLU(inplace=True),

            # nn.Conv3d(96*num_channels, 32*num_channels, kernel_size=3, padding=1, groups=num_groups),
            # nn.BatchNorm3d(32*num_channels),
            # nn.ReLU(inplace=True),
            
            # nn.Conv3d(64*num_channels, 64*num_channels, kernel_size=3, padding=1, groups=num_groups),
            # nn.BatchNorm3d(64*num_channels),
            # nn.ReLU(inplace=True),
            
            # nn.Conv3d(64*num_channels, 32*num_channels, kernel_size=3, padding=1, groups=num_groups),
            # nn.BatchNorm3d(32*num_channels),
            # nn.ReLU(inplace=True),
            
            nn.Conv3d(16*num_channels, 2*num_channels, kernel_size=3, padding=1, groups=num_groups),
            nn.BatchNorm3d(2*num_channels),
            nn.ReLU(inplace=True),


            
            nn.MaxPool3d(kernel_size=3, stride=1),
        )

        self.classifier = nn.Sequential(nn.Dropout(),
                                        nn.Linear(84*num_channels,64*num_channels),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(),
                                        nn.Linear(64*num_channels, 64),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(),
                                        nn.Linear(64, num_classes),
                                        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x):
        # import pdb; pdb.set_trace()
        xp = self.features(x)
        x = xp.view(xp.size(0), -1)
        x = self.classifier(x)
        return [x, xp]







class AlexNet3D_Dropout(nn.Module):

    def __init__(self, fkey=None, num_classes=2, num_channels=1):
        super(AlexNet3D_Dropout, self).__init__()

        if fkey == 'highres_smriPath':
            self.features = self.get_module_config('highres_smri_cnn_part', num_classes)
            self.classifier = self.get_module_config('highres_smri_lnn_part', num_classes)
        elif fkey == 'lowres_smriPath':
            self.features = self.get_module_config('lowres_smri_cnn_part', num_classes)
            self.classifier = self.get_module_config('lowres_smri_lnn_part', num_classes)

        # self.features = nn.Sequential(
        #     nn.Conv3d(1, 64, kernel_size=5, stride=2, padding=0),
        #     nn.BatchNorm3d(64),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool3d(kernel_size=3, stride=3),

        #     nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=0),
        #     nn.BatchNorm3d(128),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool3d(kernel_size=3, stride=1),

        #     nn.Conv3d(128, 192, kernel_size=3, padding=1),
        #     nn.BatchNorm3d(192),
        #     nn.ReLU(inplace=True),

        #     nn.Conv3d(192, 192, kernel_size=3, padding=1),
        #     nn.BatchNorm3d(192),
        #     nn.ReLU(inplace=True),

        #     nn.Conv3d(192, 128, kernel_size=3, padding=1),
        #     nn.BatchNorm3d(128),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool3d(kernel_size=3, stride=1),
        # )

        # self.features = nn.Sequential(
        #     nn.Conv3d(1, 64, kernel_size=5, stride=2, padding=0),
        #     nn.BatchNorm3d(64),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool3d(kernel_size=3, stride=3),

        #     nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=0),
        #     nn.BatchNorm3d(128),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool3d(kernel_size=3, stride=3),

        #     nn.Conv3d(128, 192, kernel_size=3, padding=1),
        #     nn.BatchNorm3d(192),
        #     nn.ReLU(inplace=True),

        #     nn.Conv3d(192, 192, kernel_size=3, padding=1),
        #     nn.BatchNorm3d(192),
        #     nn.ReLU(inplace=True),

        #     nn.Conv3d(192, 128, kernel_size=3, padding=1),
        #     nn.BatchNorm3d(128),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool3d(kernel_size=3, stride=3),
        # )

        # self.classifier = nn.Sequential(nn.Dropout(),
        #                                 nn.Linear(256, 64),
        #                                 nn.ReLU(inplace=True),
        #                                 nn.Dropout(),
        #                                 nn.Linear(64, num_classes),
        #                                 )

        # self.classifier = nn.Sequential(nn.Dropout(),
        #                                 nn.Linear(2048, 256),
        #                                 nn.ReLU(inplace=True),
        #                                 nn.Dropout(),
        #                                 nn.Linear(256, 64),
        #                                 nn.ReLU(inplace=True),
        #                                 nn.Dropout(),
        #                                 nn.Linear(64, num_classes),
        #                                 )

        

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        xp = self.features(x)
        x = xp.view(xp.size(0), -1)
        x = self.classifier(x)
        return [x, xp]


    def get_module_config(self, name, num_classes):
        if name == 'lowres_smri_cnn_part':
            return nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=5, stride=2, padding=0),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=3),

            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=1),

            nn.Conv3d(128, 192, kernel_size=3, padding=1),
            nn.BatchNorm3d(192),
            nn.ReLU(inplace=True),

            nn.Conv3d(192, 192, kernel_size=3, padding=1),
            nn.BatchNorm3d(192),
            nn.ReLU(inplace=True),

            nn.Conv3d(192, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=1),
        )

        elif name == 'lowres_smri_lnn_part':
            return nn.Sequential(nn.Dropout(),
                                        nn.Linear(2048, 256),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(),
                                        nn.Linear(256, 64),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(),
                                        nn.Linear(64, num_classes),
                                        )


        elif name == 'highres_smri_cnn_part':
            return nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=5, stride=2, padding=0),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=3),

            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=3),

            nn.Conv3d(128, 192, kernel_size=3, padding=1),
            nn.BatchNorm3d(192),
            nn.ReLU(inplace=True),

            nn.Conv3d(192, 192, kernel_size=3, padding=1),
            nn.BatchNorm3d(192),
            nn.ReLU(inplace=True),

            nn.Conv3d(192, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=3),
        )

        elif name == 'highres_smri_lnn_part':
            return nn.Sequential(nn.Dropout(),
                                        nn.Linear(256, 64),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(),
                                        nn.Linear(64, num_classes),
            )
               

        
