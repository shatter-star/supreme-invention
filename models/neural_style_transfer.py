import os
import cv2 as cv
import numpy as np
import torch
from torch.autograd import Variable
from torch.optim import LBFGS
from torchvision import transforms

from utils.image import load_image, prepare_img, save_image
from utils.loss import build_loss, make_tuning_step
from utils.config import load_config
from utils.logging import setup_logging

logger = setup_logging(__name__)

IMAGENET_MEAN_255 = [123.675, 116.28, 103.53]
IMAGENET_STD_NEUTRAL = [1, 1, 1]

class NeuralStyleTransfer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def prepare_model(self):
        '''
        Load VGG19 model into local cache.
        '''
        from models.definitions.vgg19 import VggModel
        model = VggModel(requires_grad=False, show_progress=True)
        content_feature_maps_index = model.content_feature_maps_index
        style_feature_maps_indices = model.style_feature_maps_indices
        layer_names = model.layer_names
        content_fms_index_name = (content_feature_maps_index, layer_names[content_feature_maps_index])
        style_fms_indices_names = (style_feature_maps_indices, layer_names)
        return model.to(self.device).eval(), content_fms_index_name, style_fms_indices_names

    def run(self, content_img_path, style_img_paths):
        '''
        The main Neural Style Transfer method.
        '''
        content_img = prepare_img(content_img_path, self.config['height'], self.device)
        style_imgs = [prepare_img(style_img_path, self.config['height'], self.device) for style_img_path in style_img_paths]

        dump_path = os.path.join(self.config['output_img_dir'], self.config['content_img_name'].split('.')[0] + '_' + '_'.join([os.path.basename(style_name).split('.')[0] for style_name in self.config['style_img_name']]))
        os.makedirs(dump_path, exist_ok=True)

        loss_history = []  # Initialize loss_history

        init_img = content_img
        optimizing_img = Variable(init_img, requires_grad=True)

        neural_net, content_feature_maps_index_name, style_feature_maps_indices_names = self.prepare_model()

        logger.info('Using VGG19 in the optimization procedure.')

        content_img_set_of_feature_maps = neural_net(content_img)
        style_img_set_of_feature_maps = [neural_net(style_img) for style_img in style_imgs]

        target_content_representation = content_img_set_of_feature_maps[content_feature_maps_index_name[0]].squeeze(axis=0)
        target_style_representation = [
            [self.gram_matrix(x) for cnt, x in enumerate(style_img_set) if cnt in style_feature_maps_indices_names[0]]
            for style_img_set in style_img_set_of_feature_maps
        ]
        target_representations = [target_content_representation, target_style_representation]

        num_of_iterations = self.config['num_iterations']

        optimizer = LBFGS((optimizing_img,), max_iter=num_of_iterations, line_search_fn='strong_wolfe')
        cnt = 0
            
        def closure():
            nonlocal cnt
            if torch.is_grad_enabled():
                optimizer.zero_grad()
            total_loss, content_loss, style_loss, tv_loss = build_loss(neural_net, optimizing_img, target_representations, content_feature_maps_index_name[0], style_feature_maps_indices_names[0], self.config)
            if total_loss.requires_grad:
                total_loss.backward()
            with torch.no_grad():
                logger.info(f'L-BFGS | iteration: {cnt:03}, total loss={total_loss.item():12.4f}, content_loss={self.config["content_weight"] * content_loss.item():12.4f}, style loss={self.config["style_weight"] * style_loss.item():12.4f}, tv loss={self.config["tv_weight"] * tv_loss.item():12.4f}')
                self.save_and_maybe_display(optimizing_img, dump_path, cnt, num_of_iterations)
                # Append the loss values to loss_history
                loss_history.append((total_loss.item(), content_loss.item(), style_loss.item(), tv_loss.item()))
            cnt += 1
            return total_loss
        
        optimizer.step(closure)
        return dump_path, loss_history

    def gram_matrix(self, x, should_normalize=True):
        '''
        Generate gram matrices of the representations of content and style images.
        '''
        (b, ch, h, w) = x.size()
        features = x.view(b, ch, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t)
        if should_normalize:
            gram /= ch * h * w
        return gram

    def total_variation(self, y):
        '''
        Calculate total variation.
        '''
        return torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) + torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :]))

    def save_and_maybe_display(self, optimizing_img, dump_path, img_id, num_of_iterations):
        '''
        Save the generated image.
        If saving_freq == -1, only the final output image will be saved.
        Else, intermediate images can be saved too.
        '''
        saving_freq = self.config.get('saving_freq', -1)
        out_img = optimizing_img.squeeze(axis=0).to('cpu').detach().numpy()
        out_img = np.moveaxis(out_img, 0, 2)

        if img_id == num_of_iterations-1 or (saving_freq != -1 and img_id % saving_freq == 0):
            img_format = self.config['img_format']
            out_img_name = str(img_id).zfill(img_format[0]) + img_format[1] if saving_freq != -1 else self.generate_out_img_name()
            dump_img = np.copy(out_img)
            dump_img += np.array(IMAGENET_MEAN_255).reshape((1, 1, 3))
            dump_img = np.clip(dump_img, 0, 255).astype('uint8')
            save_image(dump_img, os.path.join(dump_path, out_img_name))

    def generate_out_img_name(self):
        '''
        Generate a name for the output image.
        Example: 'content_style.jpg'
        '''
        content_basename = os.path.basename(self.config['content_img_name']).split('.')[0]
        style_basenames = [os.path.basename(style_name).split('.')[0] for style_name in self.config['style_img_name']]
        style_names = "_".join(style_basenames)
        prefix = f"{content_basename}_{style_names}"
        suffix = f"{self.config['img_format'][1]}"
        return prefix + suffix

if __name__ == "__main__":
    config = load_config()
    nst = NeuralStyleTransfer(config)
    dump_path, loss_history = nst.run()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    neural_net, _, _ = nst.prepare_model()

    torch.save(neural_net, 'model.pth')