import os
import torch
from athena_config import Config
C = Config()

def save_model(models, models_optimizer, idx):
    ckp_dir = os.path.join(C.checkpoint_path, C.checkpoint_name)

    if not os.path.exists(ckp_dir):
        os.makedirs(ckp_dir)

    save_dir = os.path.join(ckp_dir, "weights_{}".format(idx))

    # iterative remove test checkpoint saved directory
    # ***************************- TEST -*************************** #
    # if C.checkpoint_name == 'test' and os.path.exists(save_dir):
    #         for root, dirs, files in os.walk(save_dir, topdown=False):
    #             for name in files:
    #                 os.remove(os.path.join(root, name))
    #             for name in dirs:
    #                 os.rmdir(os.path.join(root, name))
    # ***************************- TEST -*************************** #

    os.makedirs(save_dir)

    for model_name, model in models.items():
        save_path = os.path.join(save_dir, "{}.pth".format(model_name))
        to_save = model.state_dict()
        if model_name == "encoder":
            # save the sizes - these are needed at prediction time
            to_save['height'] = C.input_height
            to_save['width'] = C.input_width
        torch.save(to_save, save_path)

    save_path = os.path.join(save_dir, "{}.pth".format("adam"))
    torch.save(models_optimizer.state_dict(), save_path)

    print("models_epoch_{} has been saved.".format(idx))