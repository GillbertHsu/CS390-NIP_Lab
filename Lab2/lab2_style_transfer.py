import os, sys
import argparse
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models




# Values that need to be changed for different pictures.
# Make sure image is a square before using.
s_image_name = "starry_night_full.jpg"  # Style image.
c_image_name = "city.jpg"               # Content image.
output_file_name = "modernStarry.png"

image_size = 1024
loader = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])


content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
unloader = transforms.ToPILImage()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#=============================<Helper Fuctions>=================================



def gram_matrix(input):
    #get batch size
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL
    G = torch.mm(features, features.t())  # compute gram product

    #normalize return value
    return G.div(a * b * c * d)



#========================<Loss Function Builder Functions>======================



class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input






class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input






class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std






def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                            style, content,
                            content_layers=content_layers_default,
                            style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)
    normalization = Normalization(normalization_mean, normalization_std).to(device)
    # Just in order to have an iterable access to or list of content/syle losses.
    content_losses = []
    style_losses = []
    # assuming that nn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization).to(device)
    #indext used for for loop
    i = 0
    #loop used to check for layers and input correct format
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
        #add to model
        model.add_module(name, layer)
        #check if content layer
        if name in content_layers:
            # add content loss:
            target = model(content).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)
        #check if style layer
        if name in style_layers:
            # add style loss:
            target_feature = model(style).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)
    # remove unneeded layers
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break
    model = model[:(i + 1)]
    return model, style_losses, content_losses






def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer






def run_style_transfer(cnn, normalization_mean, normalization_std,
                    content, style, input_img, num_steps=200,
                    style_weight=1000000, content_weight=1):
    """Run the style transfer."""
    print('Building model.....')
    model, style_losses, content_losses = get_style_model_and_losses(cnn, normalization_mean, normalization_std, style, content)
    optimizer = get_input_optimizer(input_img)
    print('Optimizing model and images for style transfer..')
    run = [0]
    while run[0] <= num_steps:
        def closure():
            with torch.no_grad():
                input_img.clamp_(0, 1)   # Correct the values of updated input image.

            #TODO: zero optimizer and forward prop the model.
            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            #TODO: calculate full loss value.
            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss
            style_score *= style_weight
            content_score *= content_weight
            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            #TODO: print progress.
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score
        # Call optimizer.
        optimizer.step(closure)
    with torch.no_grad():
        input_img.clamp_(0, 1)
    return input_img






def imshow(tensor, title=None):
    image = tensor.cpu().clone() 
    image = image.squeeze(0)
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


#=========================<Pipeline Functions>==================================

def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


def getRawData():
    print("   Loading images.")

    style = image_loader(s_image_name)
    content = image_loader(c_image_name)
    print(style.size())
    print(content.size())
    #check for image size match, other wise through error.
    #If this happens check on image and make sure formatted to square
    assert style.size() == content.size(), \
        "Make sure both input images are of the same size"

    #plot images to be show to user before running the style transfer
    plt.ion()
    plt.figure()
    imshow(style, title='Style')
    plt.figure()
    imshow(content, title='Content')
    return style, content



def styleTransfer(style, content):
    print("   Building transfer model.")

    #create model with proper layers
    cnn = models.vgg19(pretrained=True).features.to(device).eval()
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    #get image and then create a copy
    input_img = content.clone()
    plt.figure()
    imshow(input_img, title='Input Image')

    #start style transfer through function call
    output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std, content, style, input_img)

    #show the image and then save it
    plt.figure()
    imshow(output, title='Output Image')
    fname = './' + output_file_name
    plt.savefig(fname)
    plt.ioff()
    plt.show()
    print("   Transfer complete.")





#=========================<Main>================================================

def main():
    print("Starting style transfer program.")
    style, content = getRawData()
    styleTransfer(style, content)
    print("Done. Goodbye.")



if __name__ == "__main__":
    main()
