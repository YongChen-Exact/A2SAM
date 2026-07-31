import numpy as np
import torch


def prune_by_percentile_gradient_perCell(log, model, time_para=1):
    statistic = {}
    new_masks = {}

    for name, param in model.named_parameters():
        if "image_encoder" in name:
            if "norm" in name or "pos_embed" in name or "pos" in name:
                new_mask = np.ones_like(param.data.cpu().numpy())
            elif 'head' in name or "bias" in name:
                new_mask = np.zeros_like(param.data.cpu().numpy())
            elif "neck" in name and len(param.shape) == 1:
                new_mask = np.ones_like(param.data.cpu().numpy())
            else:
                if "patch_embed" in name or ("neck" in name and len(param.shape) == 4):
                    tensor = param.grad.data.cpu().numpy()
                    B, C, H, W = tensor.shape
                    tensor = np.reshape(tensor, [B, -1])
                else:
                    tensor = param.grad.data.cpu().numpy()

                new_mask = np.ones_like(tensor)
                for ind in range(time_para):
                    max_index = abs(tensor).argsort(1)[:, -(ind + 1)]
                    one_hot_temp = ~np.eye(max(tensor.shape))[max_index][:, :tensor.shape[1]].astype(bool)
                    new_mask_temp = one_hot_temp.astype(np.float32)
                    new_mask = new_mask.astype(int) & new_mask_temp.astype(int)
                    new_mask = new_mask.astype(np.float32)

                if "patch_embed" in name or ("neck" in name and len(param.shape) == 4):
                    new_mask = np.reshape(new_mask, (B, C, H, W))
        else:
            new_mask = np.zeros_like(param.data.cpu().numpy())

        trainable_param = len(new_mask.reshape(-1)) - len(np.nonzero(new_mask)[0])
        total_para = len(new_mask.reshape(-1))
        statistic[name] = [trainable_param, total_para]
        print(name, ": ", trainable_param, "/", total_para, "(", np.round((trainable_param / total_para) * 100, 4),
              "%)", new_mask.shape)

        new_masks[name] = torch.from_numpy(new_mask).cuda()

    print("---------------------------------------------------------------")
    trainable_withouthead = 0
    total_withouthead = 0
    trainable_head = 0
    total_head = 0
    for na, [trainable_p, t_p] in statistic.items():
        if "head" not in na:
            trainable_withouthead = trainable_withouthead + trainable_p
            total_withouthead = total_withouthead + t_p
        else:
            trainable_head = trainable_head + trainable_p
            total_head = total_head + t_p
    print("---------------------------------------------------------------")

    print("---------------------------------------------------------------")
    log("Trainable parameter / Total (without head): " + str(trainable_withouthead) + "/" + str(
        total_withouthead) + "(" + str(np.round((trainable_withouthead / total_withouthead) * 100, 4)) + "%)")
    log("Trainable parameter / Total (total): " + str(trainable_head + trainable_withouthead) + "/" + str(
        total_head + total_withouthead) + "(" + str(
        np.round(((trainable_head + trainable_withouthead) / (total_head + total_withouthead)) * 100, 4)) + "%)")

    print("#######################################################################")
    return new_masks


def calculate_contrastive_score(log, model, alpha=0.4, keep_ratio=0.965):
    influence = {}
    magnitude = {}
    valid_params = []
    for name, param in model.named_parameters():
        if ("iou_prediction_head" not in name) and ("discriminator" not in name) and (param.grad is not None):
            valid_params.append(name)
            influence[name] = torch.square(param.grad).data
            magnitude[name] = torch.abs(param.data)

    all_influence = torch.cat([i.flatten() for i in influence.values()])
    all_magnitude = torch.cat([m.flatten() for m in magnitude.values()])

    i_min, i_max = all_influence.min(), all_influence.max()
    p_min, p_max = all_magnitude.min(), all_magnitude.max()

    for name in influence:
        if i_max > i_min:
            influence[name] = (influence[name] - i_min) / (i_max - i_min)
        if p_max > p_min:
            magnitude[name] = (magnitude[name] - p_min) / (p_max - p_min)

    contrastive_score = {}

    for name, param in model.named_parameters():
        contrastive_score[name] = torch.zeros_like(param)
        if name in valid_params:
            i = influence[name]
            rho = magnitude[name]
            contrastive_score[name] = alpha * (i / (rho + 1)) + (1 - alpha) * i

    model.zero_grad()
    sizes = {}
    tensors = []
    all_params_size = 0

    for name in valid_params:
        v = contrastive_score[name]
        sizes[name] = v.shape
        tensors.append(v.view(-1))
        all_params_size += torch.prod(torch.tensor(v.shape)).item()

    tensors = torch.cat(tensors, 0)

    keep_num = int(all_params_size * keep_ratio)

    assert keep_num > 0

    top_pos = torch.argsort(tensors)[:keep_num]

    masks = torch.zeros_like(tensors)

    masks[top_pos] = 1

    assert masks.long().sum() == len(top_pos)

    mask_dict = {}

    now_idx = 0
    for k, v in sizes.items():
        end_idx = now_idx + torch.prod(torch.tensor(v))
        mask_dict[k] = masks[now_idx: end_idx].reshape(v)
        now_idx = end_idx

    for name, param in model.named_parameters():
        if "image_encoder" in name:
            if "norm" in name or "pos_embed" in name or "pos" in name:
                mask_dict[name] = torch.tensor(np.ones_like(param.data.cpu().numpy())).cuda()
            elif "neck" in name and len(param.shape) == 1:
                mask_dict[name] = torch.tensor(np.ones_like(param.data.cpu().numpy())).cuda()

    assert now_idx == len(masks)

    all_params_size = 0
    pretrain_weight_size = 0

    for k, v in mask_dict.items():
        pretrain_weight_size += (v == 0).sum().item()
        all_params_size += torch.prod(torch.tensor(v.shape)).item()

    log("pretrain_weight_size: " + str(pretrain_weight_size) + ", all_params_size: " + str(all_params_size))
    log("trainable parameters: " + str((pretrain_weight_size / all_params_size * 100)) + "%")

    return mask_dict
