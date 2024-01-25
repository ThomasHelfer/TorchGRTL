# A small default model
class SuperResolution3DNet(torch.nn.Module):
    def __init__(self):
        super(SuperResolution3DNet, self).__init__()

        # Encoder
        # The encoder consists of two 3D convolutional layers.
        # The first conv layer expands the channel size from 25 to 64.
        # The second conv layer further expands the channel size from 64 to 128.
        # ReLU activation functions are used for non-linearity.
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv3d(25, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv3d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )

        # Decoder
        # The decoder uses a transposed 3D convolution (or deconvolution) to upsample the feature maps.
        # The channel size is reduced from 128 back to 64.
        # A final 3D convolution reduces the channel size back to the original size of 25.
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv3d(64, 25, kernel_size=3, padding=1),
        )

    def forward(self, x):
        # Forward pass of the network
        # x is the input tensor

        # Save the original input for later use
        tmp = x

        # Apply the encoder
        x = self.encoder(x)

        # Apply the decoder
        x = self.decoder(x)

        # Reusing the input data for faster learning
        # Here, every 2nd element in the spatial dimensions of x is replaced by the corresponding element in the original input.
        # This is a form of skip connection, which helps in retaining high-frequency details from the input.
        x[:, :, ::2, ::2, ::2] = tmp[:, :, ...]

        return x
