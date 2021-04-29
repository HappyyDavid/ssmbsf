import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import torch
import torch.optim as optim

torch.set_default_tensor_type(torch.cuda.FloatTensor)

from torch.utils.data import DataLoader

from athena_dataloader import get_data_path, KittiCycleDataset
from athena_config import Config
from athena_resnet_encoder import ResnetEncoder
from athena_depth_decoder import DepthDecoder
from athena_loss import set_train, image_sequence, model_inputs, scale_disps, compute_loss, image_sequences_
from athena_logger import tbx_logger
from athena_utils import save_model

C = Config()

data_path = get_data_path(C.filename_path, C.kitti_raw_dataset_path)
dataset = KittiCycleDataset(data_path, {'h': C.input_height, 'w': C.input_width}, True, C.num_scales)
dataloader = DataLoader(dataset, batch_size=C.batch_size, shuffle=True, pin_memory=True, num_workers=C.num_thread, drop_last=False)

models = {}
parms_to_train = []

# encoder part of network
models["encoder"] = ResnetEncoder(C.num_layers, C.is_pretrained, 2).cuda()
parms_to_train += list(models["encoder"].parameters())
# decoder part of network
models["decoder"] = DepthDecoder(models["encoder"].num_ch_enc, C.scales, num_output_channels=2).cuda()
parms_to_train += list(models["decoder"].parameters())

# optimizer
optimizer = optim.Adam(parms_to_train, C.learning_rate)
# scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, 20, 0.1)

# ***************************- TEST -*************************** #
# load weights from checkpoint
'''
# load model
ckp_load_dir = os.path.join(C.checkpoint_path, C.load_ckp_name, "weights_{}".format(79))
encoder_path = os.path.join(ckp_load_dir, "encoder.pth")
decoder_path = os.path.join(ckp_load_dir, "decoder.pth")

encoder_dict = torch.load(encoder_path)
h = encoder_dict['height']
w = encoder_dict['width']
del encoder_dict['height']
del encoder_dict['width']

models['encoder'].load_state_dict({k: v for k, v in encoder_dict.items()})
models['decoder'].load_state_dict(torch.load(decoder_path))
'''
# ***************************- TEST -*************************** #
# test saving model
if C.checkpoint_name != 'test':
    print("test saving model")
    save_model(models, optimizer, -1)

index = 0
for epo in range(C.num_epoch):

    scheduler.step()
    set_train(models)

    for idx, data in enumerate(dataloader):

        optimizer.zero_grad()

        for key, item in data.items():
            data[key] = item.cuda()

        # generate model inputs
        formers, latters = image_sequences_(data, 'stereo-flow-cross', 4)
        former, latter = formers[('former', 0)], latters[('latter', 0)]

        forward_in, backward_in = model_inputs(former, latter)
        # generate model outputs
        forward_features = models["encoder"](forward_in)
        forward_out = models["decoder"](forward_features)
        backward_features = models["encoder"](backward_in)
        backward_out = models["decoder"](backward_features)

        scale_disps(forward_out, backward_out, C.num_scales, {'h': C.input_height, 'w': C.input_width})
        losses = compute_loss(forward_out, backward_out, formers, latters, C.num_scales)
        losses['total'].backward()
        optimizer.step()

        if index % 100 == 0:
            tbx_logger(C, index/100, data, models, forward_out, backward_out, losses)

        index = index + 1

        print("epo: {}, idx: {}, loss: {:.3f}".format(epo, idx, losses['total']))

    save_model(models, optimizer, epo)

print('Train Done, Thanks to Server.')


# ***************************- TEST -*************************** #
'''
print("weight: {:.4f}, grad: {:.4f}".format(models["encoder"].encoder.conv1.weight[0, 0, 0, 0],
                                            models["encoder"].encoder.conv1.weight.grad[0, 0, 0, 0]))
'''
# ***************************- TEST -*************************** #
