import torch.nn as nn
import math

class AlexNet3D_Dropout(nn.Module):

    def __init__(self, fkey=None, num_classes=2):
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
               

        
