import torch
from torch import nn

class build_resnet_block(nn.Module):
    """
    A ResNet block which includes two general_conv3d layers.
    """
    def __init__(self, channels, layers=2, do_batch_norm=False):
        super(build_resnet_block, self).__init__()
        self._channels = channels
        self._layers = layers

        # Define a sequence of 3D convolutional layers
        self.res_block = nn.Sequential(*[
            general_conv3d(in_channels=self._channels,
                           out_channels=self._channels,
                           strides=1,
                           do_batch_norm=do_batch_norm)
            for _ in range(self._layers)
        ])

    def forward(self, input_res):
        inputs = input_res.clone()
        input_res = self.res_block(input_res)
        return input_res + inputs

class upsample_conv3d_and_predict_flow(nn.Module):
    """
    An upsample convolution layer which includes a nearest interpolate and a general_conv3d.
    """
    def __init__(self, in_channels, out_channels, ksize=3, do_batch_norm=False):
        super(upsample_conv3d_and_predict_flow, self).__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._ksize = ksize
        self._do_batch_norm = do_batch_norm

        # Define a 3D convolutional layer for upsampling
        self.general_conv3d = general_conv3d(in_channels=self._in_channels,
                                             out_channels=self._out_channels,
                                             ksize=self._ksize,
                                             strides=1,
                                             do_batch_norm=self._do_batch_norm,
                                             padding=0)
        
        self.pad = nn.ReplicationPad3d(padding=(int((self._ksize-1)/2), int((self._ksize-1)/2),
                                                int((self._ksize-1)/2), int((self._ksize-1)/2),
                                                int((self._ksize-1)/2), int((self._ksize-1)/2)))

        # Define a 3D convolutional layer for predicting flow
        self.predict_flow = general_conv3d(in_channels=self._out_channels,
                                           out_channels=2,
                                           ksize=1,
                                           strides=1,
                                           padding=0,
                                           activation='tanh')

    def forward(self, conv):
        shape = conv.shape
        # Upsample the input using nearest neighbor interpolation
        conv = nn.functional.interpolate(conv, size=[shape[2]*2, shape[3]*2, shape[4]*2], mode='trilinear', align_corners=False)
        conv = self.pad(conv)
        conv = self.general_conv3d(conv)

        flow = self.predict_flow(conv) * 256.  # Scale flow values
        
        return torch.cat([conv, flow.clone()], dim=1), flow

def general_conv3d(in_channels, out_channels, ksize=3, strides=2, padding=1, do_batch_norm=False, activation='relu'):
    """
    A general 3D convolution layer which includes a Conv3d, an activation function, and optional batch normalization.
    """
    if activation == 'relu':
        if do_batch_norm:
            conv3d = nn.Sequential(
                nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=ksize,
                          stride=strides, padding=padding),
                nn.ReLU(inplace=True),
                nn.BatchNorm3d(out_channels, eps=1e-5, momentum=0.99)
            )
        else:
            conv3d = nn.Sequential(
                nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=ksize,
                          stride=strides, padding=padding),
                nn.ReLU(inplace=True)
            )
    elif activation == 'tanh':
        if do_batch_norm:
            conv3d = nn.Sequential(
                nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=ksize,
                          stride=strides, padding=padding),
                nn.Tanh(),
                nn.BatchNorm3d(out_channels, eps=1e-5, momentum=0.99)
            )
        else:
            conv3d = nn.Sequential(
                nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=ksize,
                          stride=strides, padding=padding),
                nn.Tanh()
            )
    return conv3d
