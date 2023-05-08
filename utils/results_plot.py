# %%
import os
import matplotlib.pyplot as plt
import glob
import pandas as pd
import csv
from textwrap import wrap
import numpy as np
import pytorch_lightning as pl
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

# image_log/version291/train/samples/val_batch_samples_gs-244620_e-000006_b-019800.png
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
                    # [291, "train", "samples", "val", "222750", "000005", "035400"],
           [291, "train", "samples", "val", "231120", "000006", "006300"],
            [291, "train", "samples", "val", "234120", "000006", "009300"],
            [291, "train", "samples", "val", "261720", "000006", "036900"],
                    # [291, "train", "samples", "val", "344430", "000009", "007200"],
            [291, "train", "samples", "val", "244620", "000006", "019800"],
            [291, "train", "samples", "train", "180180", "000004", "030300"]]

# image_log/version291/train/samples/val_batch_samples_gs-390300_e-000010_b-015600.png
# image_log/version291/train/samples/val_batch_samples_gs-392100_e-000010_b-017400.png
# image_log/version291/train/samples/train_batch_samples_gs-180780_e-000004_b-030900.png
# image_log/version291/train/samples/train_batch_samples_gs-190050_e-000005_b-002700.png

samples_failure = [[291, "train", "samples", "val", "390300", "000010", "015600"],
                   [291, "train", "samples", "val", "392100", "000010", "017400"],
                     [291, "train", "samples", "train", "180780", "000004", "030900"],
                     [291, "train", "samples", "train", "190050", "000005", "002700"]]

# image_log/version291/train/samples/train_batch_samples_gs-133410_e-000003_b-021000.png
# image_log/version291/train/samples/train_batch_samples_gs-173580_e-000004_b-023700.png
# image_log/version291/train/samples/train_batch_samples_gs-217350_e-000005_b-030000.png
# image_log/version291/train/samples/val_batch_samples_gs-320160_e-000008_b-020400.png
# image_log/version291/train/samples/val_batch_samples_gs-473040_e-000012_b-023400.png

abalation_samples = [[291, "train", "samples", "train", "133410", "000003", "021000"],
             [291, "train", "samples", "train", "173580", "000004", "023700"],
             [291, "train", "samples", "train", "217350", "000005", "030000"],
             [291, "train", "samples", "val", "320160", "000008", "020400"],
             [291, "train", "samples", "val", "473040", "000012", "023400"]]


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

def save_generated_images(samples, folder_name, save_gt=False):
    captions = []
    # check if pwd/results/generated exists, if not create it
    if(not os.path.exists("./results/"+folder_name+"/generated")):
        os.makedirs("./results/"+folder_name+"/generated")
    
    if(save_gt):
        if(not os.path.exists("./results/"+folder_name+"/gt")):
            os.makedirs("./results/"+folder_name+"/gt")

    for id, sample in enumerate(samples):
        path, img_type, caption = get_img_details(sample)
        img_orig = plt.imread(path)
        if(img_type=="sample"):
            img = img_orig[2:514, 2:514, :]
        if(save_gt):
            gt_img = img_orig[516:1028, 2:514, :]
            # print(gt_img.shape, "jjjjjjjjjjjjjjj")
            fig = plt.imshow(gt_img)
            plt.axis('off')
            plt.imsave("results/"+folder_name+"/gt/"+str(id)+".png", gt_img)
        fig = plt.imshow(img)
        plt.axis('off')
        plt.imsave("results/"+folder_name+"/generated/"+str(id)+".png", img)
        captions.append(caption)

    # save captions as captions.txt
    with open("results/"+folder_name+"/generated/captions.txt", 'w') as f:
        for cap in captions:
            f.write(cap+"\n")
    return captions
    
def forward_model(captions, folder_name, model_name="stable_diffusion", version = None, use_control = False):
    """
    Version None: use defualt stable diffusion model (use_control = False)
    Version <num> load model from version<num> folder
    """
    from fid_mldm import model_sample, create_model_and_sampler
    import torch
    from PIL import Image

    gpu_id = 0
    device = "cuda"+":"+str(gpu_id)
    torch.cuda.set_device(device)

    strength = 1

    model, ddim_sampler = create_model_and_sampler(version, use_control=use_control)

    model.control_scales = [strength] * 13 
    model.eval()
    if(not os.path.exists("./results/"+folder_name+"/" + model_name +"/")):
        os.makedirs("./results/"+folder_name+"/" + model_name +"/")
    with torch.no_grad():
        for id, cap in enumerate(captions):
            prompt = cap + ", visible face"
            n_prompt = ""
            num_samples = 1
            ddim_steps = 50
            results = model_sample(prompt, n_prompt, model, num_samples, ddim_sampler, ddim_steps)
            img = Image.fromarray(results[0])
            img.save("./results/"+folder_name+"/" + model_name +"/"+str(id)+".png")

def save_plots(folder_name):
    import glob
    import matplotlib.pyplot as plt
    import textwrap
    # read images from results/captioned_images and results/generated (read in numerical order)
    generated_imgs = glob.glob("./results/"+folder_name+"/generated/*.png")
    sd_images = glob.glob("./results/"+folder_name+f"/stable_diffusion/*.png")
    baseline_images = glob.glob("./results/"+folder_name+f"/baseline/*.png")
    # sort according to numerical order
    generated_imgs.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    sd_images.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    baseline_images.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    # print(generated_imgs)
    # print(sd_images)
    # read captions from ./results/captioned_images/captions.txt
    with open("./results/"+folder_name+"/generated/captions.txt", 'r') as f:
        captions = f.readlines()
        captions = [x.strip() for x in captions]

    # plot in batches of 4
    for i in range(0, len(generated_imgs)//4):
        fig, axs = plt.subplots(3, 4, figsize=(12,9.7))
        for j in range(4):
            img = plt.imread(generated_imgs[4*i+j])
            axs[0, j].imshow(img)
            axs[0, j].axis('off')
            axs[0, j].set_title(textwrap.fill(captions[4*i+j], 28), fontsize=13)
            img = plt.imread(sd_images[4*i+j])
            axs[2, j].imshow(img)
            axs[2, j].axis('off')
            img_bline = plt.imread(baseline_images[4*i+j])
            axs[1, j].imshow(img_bline)
            axs[1, j].axis('off')

        plt.tight_layout()
        # plt.imsave("./results/"+folder_name+"/plots/"+str(i)+".png", fig)
        plt.savefig("./results/"+folder_name+"/" +folder_name + "results"+str(i)+".png")


# good_captions = save_generated_images(samples, "good")
# forward_model(good_captions, "good")
# forward_model(good_captions, "good", model_name="baseline", version = 294, use_control=True) # for baseline
# save_plots("good")

# bad_captions = save_generated_images(samples_failure, "bad")
# forward_model(bad_captions, "bad")
# forward_model(bad_captions, "bad", model_name="baseline", version = 294, use_control=True) # for baseline
# save_plots("bad")

def ablation(forward_pass = False):
    # c1 : gt
    # c2 : best (291)
    # c3 : T = 400 (version 288)
    # c4 : lambda1 = 0.2 (version 290)
    # c5 : lambda2 = 0.7 (version 292)
    # abalation_captions = save_generated_images(abalation_samples, "ablation", save_gt=True)
    abalation_captions = [
                          "A man facing the camera with a smile on his face.",
                          "A woman facing the camera with a smile on her face.",
                          "An old man facing the camera with a smile on his face.",
                          "A baby facing the camera with a smile on his face.",
                          ]
    versions  = [290, 288, 291, 292]
    # versions  = [290, 292]
    if(forward_pass):
        for version in versions:
            # set a random seed
            pl.seed_everything(42)
            np.random.seed(42)
            forward_model(abalation_captions, "ablation", model_name="version"+str(version), version = version, use_control=True)
    
    # create a len(ablation_samples) x len(versions) + 1 matrix of images and plot
    import glob
    import matplotlib.pyplot as plt
    import textwrap
    # read images from results/captioned_images and results/generated (read in numerical order)
    version_imgs_names = []
    for version in versions:
        version_imgs_names.append(glob.glob("./results/ablation/version"+str(version)+"/*.png"))
        version_imgs_names[-1].sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    
    gt_imgs_names = glob.glob("./results/ablation/gt/*.png")
    gt_imgs_names.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    titles = ["Ground Truth", "$\lambda_1=0.2$, $\lambda_2=0.5$, $T=100$", "$T'=400$", "$\lambda_1'=0.1$", "$\lambda_2'=0.7$"]
    # make figure
    fig, axs = plt.subplots(len(abalation_captions), len(versions)+1, figsize=(12,10))
    for i in range(len(abalation_captions)):
        # plot gt
        # img = plt.imread(gt_imgs_names[i])
        axs[i, 0].text(0.1, 0.3, textwrap.fill(abalation_captions[i], 22), fontsize=11)
        axs[i, 0].axis('off')
        # if(i == 0):
        #     axs[i, 0].set_title("Ground Truth", fontsize=13)
        # axs[i, 0].set_title(textwrap.fill(abalation_captions[i], 28), fontsize=13)
        # plot versions
        for j in range(len(versions)):
            img = plt.imread(version_imgs_names[j][i])
            axs[i, j+1].imshow(img)
            axs[i, j+1].axis('off')
            # plot titles (latex)
            if(i == 0):
                axs[i, j+1].set_title(titles[j+1], fontsize=13)

    # plt.tight_layout()
    # set horizontal spacing between subplots
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    # remove whitespace from figure
    plt.subplots_adjust(left=0.005, right=0.995, top=0.97, bottom=0.01)
    plt.savefig("./results/ablation/ablation_results.png")

# make the same ablation function but plot row wise instead of column wise
def ablation_rowwise(forward_pass = False):
    # c1 : gt
    # c2 : best (291)
    # c3 : T = 400 (version 288)
    # c4 : lambda1 = 0.2 (version 290)
    # c5 : lambda2 = 0.7 (version 292)
    # abalation_captions = save_generated_images(abalation_samples, "ablation", save_gt=True)
    abalation_captions = [
                          "A man facing the camera with a smile on his face.",
                          "A woman facing the camera with a smile on her face.",
                          "An old man facing the camera with a smile on his face.",
                          "A baby facing the camera with a smile on his face.",
                          ]
    versions  = [290, 288, 291, 292]
    # versions  = [290, 292]
    if(forward_pass):
        for version in versions:
            # set a random seed
            pl.seed_everything(42)
            np.random.seed(42)
            forward_model(abalation_captions, "ablation", model_name="version"+str(version), version = version, use_control=True)
    
    # create a len(ablation_samples) x len(versions) + 1 matrix of images and plot
    import glob
    import matplotlib.pyplot as plt
    import textwrap
    # read images from results/captioned_images and results/generated (read in numerical order)
    version_imgs_names = []
    for version in versions:
        version_imgs_names.append(glob.glob("./results/ablation/version"+str(version)+"/*.png"))
        version_imgs_names[-1].sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    
    gt_imgs_names = glob.glob("./results/ablation/gt/*.png")
    gt_imgs_names.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    titles = ["Ground Truth", "$\lambda_1=0.2$, $\lambda_2=0.5$, $T=100$", "$T'=400$", "$\lambda_1'=0.1$", "$\lambda_2'=0.7$"]
    # make figure transposed
    fig, axs = plt.subplots(len(versions), len(abalation_captions)+1, figsize=(12,10))
    for i in range(len(abalation_captions)):
        # plot gt
        # img = plt.imread(gt_imgs_names[i])
        axs[i, 0].text(0.7, 0.3, textwrap.fill(titles[i+1], 22), fontsize=11)
        axs[i, 0].axis('off')
        # if(i == 0):
        #     axs[i, 0].set_title("Ground Truth", fontsize=13)
        # axs[i, 0].set_title(textwrap.fill(abalation_captions[i], 28), fontsize=13)
        # plot versions
        for j in range(len(versions)):
            img = plt.imread(version_imgs_names[i][j])
            axs[i, j+1].imshow(img)
            axs[i, j+1].axis('off')
            # plot titles (latex)
            if(i == 0):
                axs[i, j+1].set_title(textwrap.fill(abalation_captions[j], 22), fontsize=13)

        
    # set horizontal spacing between subplots
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    # remove whitespace from figure
    plt.subplots_adjust(left=0.00, right=0.995, top=0.9, bottom=0.01)
    # plt.tight_layout()
    plt.savefig("./results/ablation/ablation_results_rowwise.png")


# ablation_rowwise(forward_pass=False)
