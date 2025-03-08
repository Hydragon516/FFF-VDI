import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from RAFT.raft import RAFT


def initialize_RAFT(model_path='weights/raft-things.pth'):
    """Initializes the RAFT model.
    """
    args = argparse.ArgumentParser()
    args.raft_model = model_path
    args.small = False
    args.mixed_precision = False
    args.alternate_corr = False
    model = RAFT(args)

    loaded_state_dict = torch.load(args.raft_model, map_location=torch.device('cpu'))
    new_state_dict = {}
    for n, v in loaded_state_dict.items():
        name = n.replace("module.","")
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

    return model


class RAFT_bi(nn.Module):
    """Flow completion loss"""
    def __init__(self, model_path='weights/raft-things.pth'):
        super().__init__()
        self.fix_raft = initialize_RAFT(model_path)

    def dtype(self):
        return next(self.parameters()).dtype

    def forward(self, gt_local_frames, iters=20):
        b, l_t, c, h, w = gt_local_frames.size()
        # print(gt_local_frames.shape)

        with torch.no_grad():
            gtlf_1 = gt_local_frames[:, :-1, :, :, :].reshape(-1, c, h, w)
            gtlf_2 = gt_local_frames[:, 1:, :, :, :].reshape(-1, c, h, w)

            # get type of gtlf_1 and gtlf_2
            _, gt_flows_forward = self.fix_raft(gtlf_1, gtlf_2, iters=iters, test_mode=True)
            _, gt_flows_backward = self.fix_raft(gtlf_2, gtlf_1, iters=iters, test_mode=True)

        
        gt_flows_forward = gt_flows_forward.view(b, l_t-1, 2, h, w)
        gt_flows_backward = gt_flows_backward.view(b, l_t-1, 2, h, w)

        return gt_flows_forward, gt_flows_backward


##################################################################################
def smoothness_loss(flow, cmask):
    delta_u, delta_v, mask = smoothness_deltas(flow)
    loss_u = charbonnier_loss(delta_u, cmask)
    loss_v = charbonnier_loss(delta_v, cmask)
    return loss_u + loss_v


def smoothness_deltas(flow):
    """
    flow: [b, c, h, w]
    """
    mask_x = create_mask(flow, [[0, 0], [0, 1]])
    mask_y = create_mask(flow, [[0, 1], [0, 0]])
    mask = torch.cat((mask_x, mask_y), dim=1)
    mask = mask.to(flow.device)
    filter_x = torch.tensor([[0, 0, 0.], [0, 1, -1], [0, 0, 0]])
    filter_y = torch.tensor([[0, 0, 0.], [0, 1, 0], [0, -1, 0]])
    weights = torch.ones([2, 1, 3, 3])
    weights[0, 0] = filter_x
    weights[1, 0] = filter_y
    weights = weights.to(flow.device)

    flow_u, flow_v = torch.split(flow, split_size_or_sections=1, dim=1)
    delta_u = F.conv2d(flow_u, weights, stride=1, padding=1)
    delta_v = F.conv2d(flow_v, weights, stride=1, padding=1)
    return delta_u, delta_v, mask


def second_order_loss(flow, cmask):
    delta_u, delta_v, mask = second_order_deltas(flow)
    loss_u = charbonnier_loss(delta_u, cmask)
    loss_v = charbonnier_loss(delta_v, cmask)
    return loss_u + loss_v


def charbonnier_loss(x, mask=None, truncate=None, alpha=0.45, beta=1.0, epsilon=0.001):
    """
    Compute the generalized charbonnier loss of the difference tensor x
    All positions where mask == 0 are not taken into account
    x: a tensor of shape [b, c, h, w]
    mask: a mask of shape [b, mc, h, w], where mask channels must be either 1 or the same as
    the number of channels of x. Entries should be 0 or 1
    return: loss
    """
    b, c, h, w = x.shape
    norm = b * c * h * w
    error = torch.pow(torch.square(x * beta) + torch.square(torch.tensor(epsilon)), alpha)
    if mask is not None:
        error = mask * error
    if truncate is not None:
        error = torch.min(error, truncate)
    return torch.sum(error) / norm


def second_order_deltas(flow):
    """
    consider the single flow first
    flow shape: [b, c, h, w]
    """
    # create mask
    mask_x = create_mask(flow, [[0, 0], [1, 1]])
    mask_y = create_mask(flow, [[1, 1], [0, 0]])
    mask_diag = create_mask(flow, [[1, 1], [1, 1]])
    mask = torch.cat((mask_x, mask_y, mask_diag, mask_diag), dim=1)
    mask = mask.to(flow.device)

    filter_x = torch.tensor([[0, 0, 0.], [1, -2, 1], [0, 0, 0]])
    filter_y = torch.tensor([[0, 1, 0.], [0, -2, 0], [0, 1, 0]])
    filter_diag1 = torch.tensor([[1, 0, 0.], [0, -2, 0], [0, 0, 1]])
    filter_diag2 = torch.tensor([[0, 0, 1.], [0, -2, 0], [1, 0, 0]])
    weights = torch.ones([4, 1, 3, 3])
    weights[0] = filter_x
    weights[1] = filter_y
    weights[2] = filter_diag1
    weights[3] = filter_diag2
    weights = weights.to(flow.device)

    # split the flow into flow_u and flow_v, conv them with the weights
    flow_u, flow_v = torch.split(flow, split_size_or_sections=1, dim=1)
    delta_u = F.conv2d(flow_u, weights, stride=1, padding=1)
    delta_v = F.conv2d(flow_v, weights, stride=1, padding=1)
    return delta_u, delta_v, mask

def create_mask(tensor, paddings):
    """
    tensor shape: [b, c, h, w]
    paddings: [2 x 2] shape list, the first row indicates up and down paddings
    the second row indicates left and right paddings
    |            |
    |       x    |
    |     x * x  |
    |       x    |
    |            |
    """
    shape = tensor.shape
    inner_height = shape[2] - (paddings[0][0] + paddings[0][1])
    inner_width = shape[3] - (paddings[1][0] + paddings[1][1])
    inner = torch.ones([inner_height, inner_width])
    torch_paddings = [paddings[1][0], paddings[1][1], paddings[0][0], paddings[0][1]]  # left, right, up and down
    mask2d = F.pad(inner, pad=torch_paddings)
    mask3d = mask2d.unsqueeze(0).repeat(shape[0], 1, 1)
    mask4d = mask3d.unsqueeze(1)
    return mask4d.detach()