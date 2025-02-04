import cv2
import time
import argparse
from image_datasets import *
from model_loader import setup_explainer
from torchtext.vocab import GloVe
import torchvision
import pathlib
from tqdm import tqdm
import wandb
import torch
from utils.visualization_helpers import *
import matplotlib.pyplot as plt


def inference(args):
    if args.wandb:
        wandb_id_file_path = pathlib.Path('outputs/{}/runid.txt'.format(args.name))
        if wandb_id_file_path.exists():
            resume_id = wandb_id_file_path.read_text()
            wandb.init(project="temporal_scale", name=args.name, resume=resume_id, config=args)
        else:
            print('Creating new wandb instance...', wandb_id_file_path)
            run = wandb.init(project="temporal_scale", name=args.name, config=args)
            wandb_id_file_path.write_text(str(run.id))
        wandb.config.update(args)

    method = args.method
    num_top_samples = args.p

    # prepare the pretrained word embedding vectors
    embedding_glove = GloVe(name='6B', dim=args.word_embedding_dim)
    embeddings = embedding_glove.vectors.T.cuda()

    # prepare the reference dataset
    if args.refer == 'vg':
        dataset = VisualGenome(transform=data_transforms['val'])
    elif args.refer == 'coco':
        # TODO: check if this should be segmentation!!! Not sure why it would be one or the other
        dataset = MyCocoDetection(root='./data/coco/val2017',
                                  annFile='./data/coco/annotations/instances_val2017.json',
                                  transform=data_transforms['val'])
        # dataset = MyCocoSegmentation(root='./data/coco/val2017',
        #                           annFile='./data/coco/annotations/instances_val2017.json',
        #                           transform=data_transforms['val'],
        #                           inference=True)
    else:
        raise NotImplementedError
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    # load the target model with a trained explainer
    model = setup_explainer(args, random_feature=args.random)
    if len(args.model_path) < 1:
        args.model_path = 'outputs/' + args.name + '/ckpt_best.pth.tar'
    if len(args.max_path) < 1:
        args.max_path = 'outputs/' + args.name + '/act_max_{}.pt'.format(args.method)
    ckpt = torch.load(args.model_path)
    state_dict = ckpt['state_dict']
    model.load_state_dict(state_dict)
    model = model.cuda()
    model.eval()

    # get the max activation of each example on the target filter

    if not os.path.exists(args.max_path):
        print('extracting max activations...')
        for k, batch in enumerate(tqdm(dataloader)):
            ## DEBUG
            # if k == 10:
            #     break
            ## DEBUG

            img = batch[0].cuda()
            x = img.clone().detach()
            for name, module in model._modules.items():
                x = module(x)
                if name == args.layer:
                    break
            x = x.cpu().detach().numpy()
            if k == 0:
                max_activations = np.zeros((x.shape[1],len(dataset)))
            max_activations[:,k] = np.max(x.squeeze(0), axis=(-1, -2))
            # max_activations[k] = np.max(x, axis=(-1, -2))[f]
        torch.save(max_activations, args.max_path)
        print('Activations of all filters saved to {}'.format(args.max_path))
    max_activations = torch.load(args.max_path)

    # sort images by their max activations
    sorted_samples = np.argsort(-max_activations, axis=1)

    # activation threshold
    threshold = args.mask_threshold

    if args.wandb:
        # create wandb table where row headers are filter numbers, then add each word as column
        columns_tmp = list(range(1,args.num_output+1))
        columns_tmp.insert(0,'filter')
        columns = [str(col) for col in columns_tmp]
        data = []
    for f in tqdm(args.f):
        if args.wandb:
            top_k_heatmaps = [0, 0, 0]
            max_weights = np.zeros(args.num_heatmap_log)
        with torch.no_grad():
            start = time.time()
            print('explaining filter %d with %d top activated images' % (f, num_top_samples))
            filter_dataset = torch.utils.data.Subset(dataset, sorted_samples[f, :num_top_samples])
            filter_dataloader = torch.utils.data.DataLoader(filter_dataset, batch_size=1,
                                                            shuffle=False, num_workers=args.num_workers)
            weights = 0
            for i, batch in enumerate(filter_dataloader):
                if not batch[1]:
                    continue
                data_, annotation = batch[0].cuda(), batch[1]
                x = data_.clone()
                for name, module in model._modules.items():
                    x = module(x)
                    if name == args.layer:
                        activation = x.detach().cpu().numpy()
                        break
                c = activation[:, f, :, :]
                c = c.reshape(c.shape[-1], c.shape[-1])
                xf = cv2.resize(c, (224, 224))
                weight = np.amax(c)
                if weight <= 0.:
                    continue

                # interpret the explainer's output with the specified method
                predict = explain(method, model, data_, activation, c, xf, threshold)
                # get score between visual feature and all possible word embeddings
                predict_score = torch.mm(predict, embeddings) / \
                                torch.mm(torch.sqrt(torch.sum(predict ** 2, dim=1, keepdim=True)),
                                         torch.sqrt(torch.sum(embeddings ** 2, dim=0, keepdim=True)))
                # ranking the similarity, sorted_predict are descending indices of most similar words
                sorted_predict_score, sorted_predict = torch.sort(predict_score, dim=1, descending=True)

                sorted_predict = sorted_predict[0, :].detach().cpu().numpy()
                # take the top s words, and then repeat them floor(weight) times, since we use frequency of a word to determine the final concept
                if args.activation_repeat:
                    select_rank = np.repeat(sorted_predict[:args.s], int(weight))
                else:
                    select_rank = sorted_predict[:args.s]

                if weights == 0:
                    filter_rank = select_rank
                else:
                    filter_rank = np.concatenate((filter_rank, select_rank))

                weights += weight

                # VISUALIZE HERE
                if args.wandb:
                    max_weights_copy = max_weights.copy()
                    for idx, weight_max in enumerate(max_weights_copy):
                        if weight > weight_max:
                            max_weights[idx] = weight
                            viz_img = unnorm(data_.cpu())
                            viz_img = np.array(viz_img.squeeze(0))
                            heatmap_vis = combine_heatmap_img(viz_img, xf)
                            top_k_heatmaps[idx] = torch.tensor(heatmap_vis)
                            ### DEBUGING ###
                            # plt.imshow(heatmap_vis.transpose(1,2,0))
                            # plt.show()
                            break

            # todo: check if there is an issue here where a lot of words are not checked
            with open('data/entities.txt') as file:
                all_labels = [line.rstrip() for line in file]
            # determine word with most frequency, this is the word used as the highest ranked concept
            (values, counts) = np.unique(filter_rank, return_counts=True)
            ind = np.argsort(-counts)
            sorted_predict_words = []
            for ii in ind[:args.num_output]:
                word = embedding_glove.itos[int(values[ii])]
                if word in all_labels:
                    sorted_predict_words.append(word)

            end = time.time()
            print('Elasped Time: %f s' % (end - start))

        # save the top-k words
        if args.wandb:
            data_row = ["{}".format(word) for word in sorted_predict_words]
            data_row.insert(0, f)
            if len(data_row) < len(columns):
                for i in range(len(columns)):
                    data_row.append('-')
                    if len(data_row) == len(columns):
                        break
            data.append(data_row)
            # log heatmaps to wandb
            filter_image_array = torchvision.utils.make_grid(top_k_heatmaps)
            caption = "Method: {} | Filter: {} | Concept: {}".format(args.method, f, sorted_predict_words[0])
            ##### debug
            plt.imshow(filter_image_array.numpy().transpose(1,2,0))
            plt.title(caption)
            plt.show()
            #### debug
            # images = wandb.Image(filter_image_array, caption=caption)
            # wandb.log({"Filter Heatmaps and Highest Concept": images})



        print('Sorted words for filter index {}: {}'.format(f, sorted_predict_words))
        print()
    if args.wandb:
        table = wandb.Table(data=data, columns=columns)
        wandb.log({"Top-{} Filter_Explanations".format(args.num_output): table})
    return sorted_predict_words


def explain(method, model, data_, activation, c, xf, threshold):
    img = data_.cpu().detach().numpy().squeeze()
    img = np.transpose(img, (1, 2, 0))
    # TODO: Implement saving of visualization of heatmaps
    if method == 'original':
        # original image
        data = data_.clone().requires_grad_(True)
        predict = model(data)
    elif method == 'projection':
        # filter attention projection
        # dot product of tensor and filter & c to get similarities, then normalized by sqrt of sum
        filter_embed = torch.tensor(
            np.mean(activation * c / (np.sum(c ** 2, axis=(0, 1)) ** .5), axis=(2, 3))).cuda()
        predict = model.fc(filter_embed)
    elif method == 'image':
        # image masking
        data = img * (xf[:, :, None] > threshold)
        data = torch.tensor(np.transpose(data, (2, 0, 1))).unsqueeze(0).cuda()
        predict = model(data)
    elif method == 'activation':
        # activation masking
        filter_embed = torch.tensor(np.mean(activation * (c > threshold), axis=(2, 3))).cuda()
        predict = model.fc(filter_embed)
    else:
        raise NotImplementedError

    return predict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer', type=str, default='layer4', help='target layer')
    # parser.add_argument('--f', type=list, default=[0,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000],
    parser.add_argument('--f', type=list, default=[0,50,100,150,200,250,300,350,400,450,500],
                        help='list of index of the target filters')
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--method', type=str, default='projection',
                        choices=('original', 'image', 'activation', 'projection'),
                        help='method used to explain the target filter')
    parser.add_argument('--word-embedding-dim', type=int, default=300)
    parser.add_argument('--refer', type=str, default='coco', choices=('vg', 'coco'), help='reference dataset')
    parser.add_argument('--num-output', type=int, default=10,
                        help='number of words used to explain the target filter')
    parser.add_argument('--random', type=bool, default=False, help='Use randomly initialized models instead of pretrained feature extractors')
    parser.add_argument('--model-path', type=str, default='', help='path to load the target model')
    parser.add_argument('--thresh-path', type=str, help='path to save/load the thresholds')
    parser.add_argument('--mask_threshold', type=float, default=0.04, help='path to save/load the thresholds')
    parser.add_argument('--max-path', type=str, default='', help='path to save/load the max activations of all examples')
    parser.add_argument('--pretrain', type=str, default=None, help='path to the pretrained model')
    parser.add_argument('--model', type=str, default='resnet50', help='target network')
    parser.add_argument('--classifier_name', type=str, default='fc', help='name of classifier layer')
    parser.add_argument('--name', type=str, default='baseline_layer4', help='experiment name, used for resuming wandb run')
    parser.add_argument('--activation_repeat', type=bool, default=True, help='Weight concepts by size of activation')
    parser.add_argument('--wandb', type=bool, default=True, help='Use wandb for logging')
    parser.add_argument('--save_heatmaps', type=bool, default=True, help='Option to save heatmaps to wandb')
    parser.add_argument('--num_heatmap_log', type=int, default=3, help='Number of heatmaps to save')


    # if filter activation projection is used
    parser.add_argument('--s', type=int, default=5,
                        help='number of semantics contributed by each top activated image')
    parser.add_argument('--p', type=int, default=25,
                        help='number of top activated images used to explain each filter')

    args = parser.parse_args()
    print(args)

    inference(args)
    print()