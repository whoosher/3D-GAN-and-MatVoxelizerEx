from torchsummary import summary
import torch
from Generator import Generator
from Discriminator import Discriminator

def test_gan3d(print_summary=True):
    noise_dim = 200 # latent space vector dim
    input_dim = 512 # convolutional channels
    dim = 64  # cube volume
    model_generator = Generator(input_dim=input_dim, out_dim=dim, out_channels=1, noise_dim=noise_dim)
    noise = torch.rand(1, noise_dim)
    generated_volume = model_generator(noise)
    print("Generator output shape", generated_volume.shape)
    model_discriminator = Discriminator(in_channels=1 , dim=64, out_conv_channels=512)
    out = model_discriminator(generated_volume)
    print("Discriminator output", out.item())
    if print_summary:
      print("\n\nGenerator summary\n\n")
      summary(model_generator, (1, noise_dim))
      print("\n\nDiscriminator summary\n\n")
      summary(model_discriminator, (1,dim,dim,dim))

test_gan3d()