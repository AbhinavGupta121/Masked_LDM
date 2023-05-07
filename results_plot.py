# %%
import os
import matplotlib.pyplot as plt
import glob
import pandas as pd
import csv
from textwrap import wrap
# 291, train, samples, train, 058470, 000001, 021000
# 291, train, samples, train, 078240, 000002, 003300
# 290, train, samples, val,   142410, 000003, 030000
# 292, train, samples, val,   153480, 000004, 003600
# 292, train, samples, val,   137610, 000003, 025200
# 292, fid_val, step149880,   149880, 000004, 000008, 000 
# 292, fid_val, step149880,   149880, 000004, 000319, 001
# 292, fid_val, step149880,   149880, 000004, 000471, 000
# 291, fid_val, step449640,   449640, 000012, 000131, 001
# <...>
# image_log/version291/train/samples/val_batch_samples_gs-162480_e-000004_b-012600.png
# image_log/version291/train/samples/val_batch_samples_gs-175380_e-000004_b-025500.png
# image_log/version291/train/samples/val_batch_samples_gs-186780_e-000004_b-036900.png
# image_log/version291/train/samples/val_batch_samples_gs-222750_e-000005_b-035400.png
# image_log/version291/train/samples/val_batch_samples_gs-231120_e-000006_b-006300.png
# image_log/version291/train/samples/val_batch_samples_gs-234120_e-000006_b-009300.png
# image_log/version291/train/samples/val_batch_samples_gs-261720_e-000006_b-036900.png
# image_log/version291/train/samples/val_batch_samples_gs-344430_e-000009_b-007200.png
# image_log/version291/train/samples/train_batch_samples_gs-180180_e-000004_b-030300.png

# convert to above format
# convert above to list of lists, dont omit the 0s
samples = [[291, 'train', 'samples', 'train', "058470", "000001", "021000"],
           [291, 'train', 'samples', 'train', "078240", "000002", "003300"],
           [290, 'train', 'samples', 'val',   "142410", "000003", "030000"],
           [292, 'train', 'samples', 'val',   "153480", "000004", "003600"],
           [292, 'train', 'samples', 'val',   "137610", "000003", "025200"],
           [292, 'fid_val', 'step149880',   "149880", "000004", "000008", "000"],
           [292, 'fid_val', 'step149880',   "149880", "000004", "000319", "001"],
           [292, 'fid_val', 'step149880',   "149880", "000004", "000471", "000"],
           [291, 'fid_val', 'step449640',   "449640", "000012", "000131", "001"],
            [292, "train", "samples", "train", "111540", "000002", "036600"],
            # [291, "train", "samples", "val", "162480", "000004", "012600"],
            [291, "train", "samples", "val", "175380", "000004", "025500"],
            # [291, "train", "samples", "val", "186780", "000004", "036900"],
            [291, "train", "samples", "val", "222750", "000005", "035400"],
            [291, "train", "samples", "val", "231120", "000006", "006300"],
            [291, "train", "samples", "val", "234120", "000006", "009300"],
            [291, "train", "samples", "val", "261720", "000006", "036900"],
            # [291, "train", "samples", "val", "344430", "000009", "007200"],
            [291, "train", "samples", "train", "180180", "000004", "030300"]]

# image_log/version291/train/samples/val_batch_samples_gs-390300_e-000010_b-015600.png
# image_log/version291/train/samples/val_batch_samples_gs-392100_e-000010_b-017400.png
# image_log/version291/train/samples/train_batch_samples_gs-180780_e-000004_b-030900.png
# image_log/version291/train/samples/train_batch_samples_gs-190050_e-000005_b-002700.png

samples_failure = [[291, "train", "samples", "val", "390300", "000010", "015600"],
                   [291, "train", "samples", "val", "392100", "000010", "017400"],
                     [291, "train", "samples", "train", "180780", "000004", "030900"],
                     [291, "train", "samples", "train", "190050", "000005", "002700"]]

def get_img_details(sample):
    # print("--------------------")
    path = "/home/phebbar/Documents/ControlNet"
    # normal: image_log/version291/train/samples
    # fid:    image_log/version291/fid_val/step449640
    path = os.path.join(path, "image_log", "version" + str(sample[0]))
    path = os.path.join(path, sample[1], sample[2])
    if(sample[3]=="train" or sample[3]=="val"):
        img_name = sample[3] + "_batch_samples_gs-" + sample[4] + "_e-" + sample[5] + "_b-" + sample[6] + ".png"
        file_name = glob.glob(path + "/"+ sample[3]+"*.csv")[0]
        path = os.path.join(path, img_name)
        img_type = "sample"
        # read file as pandas dataframe
        global_step = int(sample[4])
        epoch = int(sample[5])
        batch_index = int(sample[6])
        caption = ""
        with open(file_name, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if(row[0]==str(global_step) and row[1]==str(epoch) and row[2]==str(batch_index)):
                    caption = str(row[3:])
                    # remove the square brackets, quotes and commas
                    caption = caption.replace("[", "")
                    caption = caption.replace("]", "")
                    caption = caption.replace("'", "")
                    caption = caption.replace('"', "")
                    caption = caption.replace(",", "")
                    break
    else:
        # samples_cfg_gs-149880_e-000004_b-000333_idx000.png
        img_name = "samples_cfg_gs-" + sample[3] + "_e-" + sample[4] + "_b-" + sample[5] + "_idx" + sample[6] + ".png"
        # print(path)
        file_name = glob.glob(path + "/"+"*.txt")[0]   
        path = os.path.join(path, img_name)
        img_type = "fid"
        global_step = int(sample[3])
        epoch = int(sample[4])
        batch_index = int(sample[5])
        idx = int(sample[6])
        caption = ""
        count_row = 0
        with open(file_name, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if(row[0]==str(global_step) and row[1]==str(epoch) and row[2]==str(batch_index)):
                    if(count_row!=idx):
                        count_row += 1
                        continue
                    count_row += 1
                    caption = str(row[3:])
                    # remove the square brackets, quotes and commas
                    caption = caption.replace("[", "")
                    caption = caption.replace("]", "")
                    caption = caption.replace("'", "")
                    caption = caption.replace('"', "")
                    caption = caption.replace(",", "")
                    break
    return path, img_type, caption

def save_generated_images(samples, folder_name):
    captions = []
    # check if pwd/results/generated exists, if not create it
    if(not os.path.exists("./results/"+folder_name+"/generated")):
        os.makedirs("./results/"+folder_name+"/generated")

    for id, sample in enumerate(samples):
        path, img_type, caption = get_img_details(sample)
        img = plt.imread(path)
        if(img_type=="sample"):
            img = img[2:514, 2:514, :]
        fig = plt.imshow(img)
        plt.axis('off')
        plt.imsave("results/"+folder_name+"/generated/"+str(id)+".png", img)
        captions.append(caption)

    # save captions as captions.txt
    with open("results/"+folder_name+"/generated/captions.txt", 'w') as f:
        for cap in captions:
            f.write(cap+"\n")
    return captions
    
def forward_model(captions, folder_name):
    from fid_mldm import model_sample, create_model_and_sampler
    import torch
    from PIL import Image

    gpu_id = 0
    device = "cuda"+":"+str(gpu_id)
    torch.cuda.set_device(device)

    version = None # default stable diffusion
    use_control  = False
    strength = 1

    model, ddim_sampler = create_model_and_sampler(version, use_control=use_control)

    model.control_scales = [strength] * 13 
    model.eval()
    if(not os.path.exists("./results/"+folder_name+"/stable_diffusion")):
        os.makedirs("./results/"+folder_name+"/stable_diffusion")
    with torch.no_grad():
        for id, cap in enumerate(captions):
            prompt = cap + ", visible face"
            n_prompt = ""
            num_samples = 1
            ddim_steps = 50
            results = model_sample(prompt, n_prompt, model, num_samples, ddim_sampler, ddim_steps)
            img = Image.fromarray(results[0])
            img.save("./results/"+folder_name+"/stable_diffusion/"+str(id)+".png")

def save_plots(folder_name):
    import glob
    import matplotlib.pyplot as plt
    import textwrap
    # read images from results/captioned_images and results/generated (read in numerical order)
    generated_imgs = glob.glob("./results/"+folder_name+"/generated/*.png")
    sd_images = glob.glob("./results/"+folder_name+"/stable_diffusion/*.png")
    # sort according to numerical order
    generated_imgs.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    sd_images.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    # print(generated_imgs)
    # print(sd_images)
    # read captions from ./results/captioned_images/captions.txt
    with open("./results/"+folder_name+"/generated/captions.txt", 'r') as f:
        captions = f.readlines()
        captions = [x.strip() for x in captions]

    # plot in batches of 4
    for i in range(0, len(generated_imgs)//4):
        fig, axs = plt.subplots(2, 4, figsize=(12,6.7))
        for j in range(4):
            img = plt.imread(generated_imgs[4*i+j])
            axs[0, j].imshow(img)
            axs[0, j].axis('off')
            axs[0, j].set_title(textwrap.fill(captions[4*i+j], 28), fontsize=13)
            img = plt.imread(sd_images[4*i+j])
            axs[1, j].imshow(img)
            axs[1, j].axis('off')

        plt.tight_layout()
        # plt.imsave("./results/"+folder_name+"/plots/"+str(i)+".png", fig)
        plt.savefig("./results/"+folder_name+"/" +folder_name + "_results_"+str(i)+".png")


# good_captions = save_generated_images(samples, "good")
# # print(good_captions)
# bad_captions = save_generated_images(samples_failure, "bad")
# # print(bad_captions)
# forward_model(good_captions, "good")
# forward_model(bad_captions, "bad")

save_plots("good")
save_plots("bad")

