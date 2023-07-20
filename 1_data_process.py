from utils import *
from mifgsm import MIFGSM
from mifgsm import *
import argparse
from torch.nn import functional as F
import csv

def get_parser():
    parser = argparse.ArgumentParser(description='Generating transferable adversaria examples')
    parser.add_argument('-e', '--eval', action='store_true', help='attack/evluation')
    parser.add_argument('--epoch', default=10, type=int, help='the iterations for updating the adversarial patch')
    parser.add_argument('--batchsize', default=5, type=int, help='the bacth size')
    parser.add_argument('--eps', default=16/255, type=float, help='the stepsize to update the perturbation')
    parser.add_argument('--alpha', default=1.6/255, type=float, help='the stepsize to update the perturbation')
    parser.add_argument('--momentum', default=1., type=float, help='the decay factor for momentum based attack')
    parser.add_argument('--model', default='resnet18', type=str, help='the source surrogate model')
    parser.add_argument('--input_dir', default='./data/val_rs', type=str, help='the path for custom benign images, default: untargeted attack data')
    parser.add_argument('--input_csv', default='./data/val_rs.csv', type=str, help='the path for custom benign images, default: untargeted attack data')
    parser.add_argument('--output_dir', default='./results/mifgsm', type=str, help='the path to store the adversarial patches')
    parser.add_argument('--targeted', action='store_true', help='targeted attack')
    parser.add_argument('--GPU_ID', default='2', type=str)
    return parser.parse_args()


def eval_acc(images, label, model_list):
    batchsize = images.shape[0]
    images = images.cuda()
    label = label.cuda()
    asr = torch.zeros(batchsize)
    for model_name in model_list:
        model = get_model(model_name)
        with torch.no_grad():
            asr += (model(images).argmax(1)!=(label)).detach().cpu()
    return asr/len(model_list)

def main():
    source_model = 'mobilenetv2'
    eval_model = ['mobilenetv3', 'resnet50', 'resnet101', 'densenet201', 'vgg19', 'xception65', 'resnet152', 'efficientnet_b4', 'efficientnetv2_s', 'adv_inception_v3']
    args = get_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_ID
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    sample_header = ['filename', 'truelabel', 'transform_lists', 'asr']
    csv_name = 'random_sample2.csv'
    with open(csv_name, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(sample_header)
    for _ in range(10):
        dataset = load_images(args.input_dir, args.input_csv)
        dataloader = DataLoader(dataset, batch_size=args.batchsize, shuffle=False, pin_memory=True, num_workers=4)
        # attacker = MIFGSM(source_model)
        for filenames, images, true_label in tqdm(dataloader):
            transform_index = list(random.randint(1, 20) for _ in range(4))
            number_strings = [str(num) for num in transform_index]
            trans_lists_str = "_".join(number_strings)
            attacker = MIFGSM(source_model, transform_list=transform_index)
            perturbations = attacker(images, true_label)
            asr = eval_acc(images + perturbations.cpu(), true_label, eval_model)
            save_images(args.output_dir, images + perturbations.cpu(), filenames)
            with open(csv_name, 'a') as file:
                writer = csv.writer(file)
                img_asr_trans_list = []
                for i in range(args.batchsize):
                    temp_list = [filenames[i], true_label[i].item(), trans_lists_str, asr[i].item()]
                    print(f'filename:{filenames[i]}, true_label:{true_label[i].item()}, trans_list:{trans_lists_str}, attack_success_rate:{asr[i].item()}')
                    img_asr_trans_list.append(temp_list)
                writer.writerows(img_asr_trans_list)
            # print(f'filename:{filenames}, true_label:{true_label}, trans_list:{trans_lists_str}, attack_success_rate:{asr}')

        # images = images.cuda()
        # trans_images = transform_index(images, 20)
        # save_images(args.output_dir, trans_images, filenames)

if __name__=='__main__':
    main()
