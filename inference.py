"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

"""


import os
from options.inference_options import InferenceOptions
from data import create_dataset
from models import create_model
from utils.visualizer import save_images
from utils import html
import time

if __name__ == '__main__':
    start_time = time.time()
    print('start time:', start_time)

    opt = InferenceOptions().parse()  
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1

    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)

    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))
    if opt.load_iter > 0:
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

    if opt.eval:
        model.eval()

    for i, data in enumerate(dataset):
        if i >= opt.num_test:
            break
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        img_paths = model.get_image_paths()

        # ensure img_path is a string
        if isinstance(img_paths, list):
            img_path = img_paths[0]
        else:
            img_path = img_paths

        if i % 5 == 0:
            print('processing (%04d)-th image... %s' % (i, img_path))

        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)

    webpage.save()

    end_time = time.time()
    print('cost time:', end_time - start_time)

