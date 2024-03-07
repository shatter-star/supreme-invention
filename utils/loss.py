import torch

def build_loss(neural_net, optimizing_img, target_representations, content_feature_maps_index, style_feature_maps_indices, config):
    '''
    Calculate content_loss, style_loss, and total_variation_loss.
    '''
    target_content_representation, target_style_representation = target_representations
    current_set_of_feature_maps = neural_net(optimizing_img)
    current_content_representation = current_set_of_feature_maps[content_feature_maps_index].squeeze(axis=0)
    content_loss = torch.nn.MSELoss(reduction='mean')(target_content_representation, current_content_representation)
    
    style_loss = 0.0
    current_style_representation = [gram_matrix(x) for cnt, x in enumerate(current_set_of_feature_maps) if cnt in style_feature_maps_indices]
    for gram_gt, gram_hat in zip(target_style_representation, current_style_representation):
        style_loss += torch.nn.MSELoss(reduction='sum')(gram_gt[0].squeeze(0), gram_hat[0].squeeze(0))  
    style_loss /= len(target_style_representation)
    
    tv_loss = total_variation(optimizing_img)
    total_loss = config['content_weight'] * content_loss + config['style_weight'] * style_loss + config['tv_weight'] * tv_loss
    return total_loss, content_loss, style_loss, tv_loss

def make_tuning_step(neural_net, optimizer, target_representations, content_feature_maps_index, style_feature_maps_indices, config):
    '''
    Performs a step in the tuning loop.
    (We are tuning only the pixels, not the weights.)
    '''
    def tuning_step(optimizing_img):
        total_loss, content_loss, style_loss, tv_loss = build_loss(neural_net, optimizing_img, target_representations, content_feature_maps_index, style_feature_maps_indices, config)
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return total_loss, content_loss, style_loss, tv_loss
    return tuning_step

def gram_matrix(x, should_normalize=True):
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

def total_variation(y):
    '''
    Calculate total variation.
    '''
    return torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) + torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :]))