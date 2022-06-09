"""
Test a trained autoencoder
"""
import argparse
import os
import random
import time

import cv2  # pytype: disable=import-error
import imgaug
import numpy as np
import torch as th

from ae.autoencoder import load_ae, preprocess_image
from ae.data_loader import CheckFliplrPostProcessor, get_image_augmenter

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--folder", help="Log folder", type=str, default="logs/recorded_data/")
parser.add_argument("-ae", "--ae-path", help="Path to saved AE", type=str, default="")
parser.add_argument("-n", "--n-samples", help="Max number of samples", type=int, default=20)
parser.add_argument("--seed", help="Random generator seed", type=int, default=0)
parser.add_argument("-augment", "--augment", action="store_true", default=False, help="Use image augmenter")
args = parser.parse_args()

if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    if th.cuda.is_available():
        th.backends.cudnn.deterministic = True
        th.backends.cudnn.benchmark = False


if not args.folder.endswith("/"):
    args.folder += "/"

latent_sizes = args.ae_path.split(',')

autoencoders = []
for sz in latent_sizes:
    autoencoder = load_ae(f'ae-{sz}-hpc.pkl')
    autoencoders.append( autoencoder )

images = [im for im in os.listdir(args.folder) if im.endswith(".jpg")]
images = np.array(images)
n_samples = len(images)

augmenter = None
if args.augment:
    augmenter = get_image_augmenter()

# Small benchmark
# start_time = time.time()
# for _ in range(args.n_samples):
#     # Load test image
#     image_idx = np.random.randint(n_samples)
#     image_path = args.folder + images[image_idx]
#     image = cv2.imread(image_path)
#     input_image = image

#     encoded = autoencoder.encode_from_raw_image(input_image)
#     reconstructed_image = autoencoder.decode(encoded)[0]

# time_per_image = (time.time() - start_time) / args.n_samples
# print(f"{time_per_image:.4f}s")
# print(f"{1 / time_per_image:.4f}Hz")

errors = []

for _ in range(args.n_samples):
    # Load test image
    image_idx = np.random.randint(n_samples)
    image_path = args.folder + images[image_idx]
    image = cv2.imread(image_path)

    postprocessor = CheckFliplrPostProcessor()

    if augmenter is not None:
        input_image = augmenter.augment_image(image, hooks=imgaug.HooksImages(postprocessor=postprocessor))
    else:
        input_image = image

    if postprocessor.flipped:
        image = imgaug.augmenters.Fliplr(1).augment_image(image)

<<<<<<< HEAD
    cropped_image = preprocess_image(image, convert_to_rgb=False, normalize=False)
    encoded = autoencoder.encode_from_raw_image(input_image)
    encoded = autoencoder.encode(cropped_image[:, :, ::-1])

    reconstructed_image = autoencoder.decode(encoded)[0]
=======
    final_images = []
    for autoencoder in autoencoders:
        cropped_image = preprocess_image(image, convert_to_rgb=False, normalize=False)
        encoded = autoencoder.encode_from_raw_image(input_image)
        reconstructed_image = autoencoder.decode(encoded)[0]
>>>>>>> ce47080bba7fa1a8436a61c918af1d34ab5aea4a

        error = np.mean((cropped_image - reconstructed_image) ** 2)
        errors.append(error)
        
        three_imgs = np.vstack((image,cropped_image,reconstructed_image))
        final_images.append(three_imgs)

    final_images = np.hstack(final_images)

    cv2.imshow( f"Orig,Crop,Recon - {args.ae_path}", final_images)
    # stop if escape is pressed
    k = cv2.waitKey(0) & 0xFF
    if k == 27:
        break

print(f"Min error: {np.min(errors):.2f}")
print(f"Max error: {np.max(errors):.2f}")
print(f"Mean error: {np.mean(errors):.2f} +/- {np.std(errors):.2f}")
print(f"Median error: {np.median(errors):.2f}")
