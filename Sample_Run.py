import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
import numpy as np
from Mat_voxelizer import EX_airplane
from Generator import Generator
from Discriminator import Discriminator
from matplotlib import pyplot as plt


#Data Load
DIR_PATH = r'D:\3DShapeNets\volumetric_data\airplane\30\train' #Your file path
Airplane_Data = EX_airplane(DIR_PATH,object_ratio=0.1)
print(Airplane_Data)


#Hyper-Parameters
num_epochs = int(1000)
lr_D = 0.0002
lr_G = 0.0002
beta1=0.5
device = torch.device("cuda:0" if (torch.cuda.is_available() and 64 > 0) else "cpu")
batch_size=64

# Initialize BCELoss function
criterion = nn.BCELoss().to(device)
size = len(Airplane_Data)
print(size)

#Creat model_generator and model_discriminator
noise_dim = 200  # latent space vector dim
input_dim = 512  # convolutional channels
dim = 64  # cube volume
noise = torch.rand(1, noise_dim).to(device)
print(noise.is_cuda)

model_generator = Generator(input_dim=input_dim, out_dim=dim, out_channels=1, noise_dim=noise_dim).to(device)
generated_volume = model_generator(noise)
print("model_generator output shape", generated_volume.shape,generated_volume)
model_discriminator = Discriminator(in_channels=1, dim=64, out_conv_channels=512).to(device)
out = model_discriminator(generated_volume)
print("model_discriminator output", out.item())



# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
# fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(model_discriminator.parameters(), lr=lr_D, betas=(beta1, 0.999))
optimizerG = optim.Adam(model_generator.parameters(), lr=lr_G, betas=(beta1, 0.999))

shape_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for idx in range(len(Airplane_Data)):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        model_discriminator.zero_grad()
        # Format batch
        real_cuda = torch.Tensor(Airplane_Data[idx]).to(device)
        b_size = batch_size
        label = torch.full((1,1), real_label, dtype=torch.float).to(device)
        # Forward pass real batch through D
        output = model_discriminator(real_cuda)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.rand(1, noise_dim).to(device)
        fake = model_generator(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = model_discriminator(fake.detach())
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        model_generator.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = model_discriminator(fake)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if idx % 1 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, idx, size,
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (idx == size - 1)):
            with torch.no_grad():
                fake = model_generator(noise).detach().cpu()
            shape_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1

#Saving model weight
SAVE_PATH = r'C:\Users\USER\PycharmProjects\Generative_design\pretrained' #Your SAVE PATH
torch.save(model_generator, SAVE_PATH + '\model_G.pt')
torch.save({
    'model': model_generator.state_dict(),
    'optimizer': optimizerG.state_dict()
}, SAVE_PATH + r'\all.tar')
torch.save(model_discriminator, SAVE_PATH + '\model_D.pt')
torch.save({
    'model': model_discriminator.state_dict(),
    'optimizer': optimizerD.state_dict()
}, SAVE_PATH + r'\all.tar')

#Generated Shape plotting
generated_volume = model_generator(torch.rand(1, noise_dim).to(device))
generated_volume_cpu = generated_volume.cpu()
volume_list = generated_volume_cpu.detach().numpy()
volume_list=np.reshape(volume_list,(64,64,64))

# 3D Voxel Visualization
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
# ax.set_aspect('equal')
ax.voxels(volume_list, edgecolor='red')
plt.show()

