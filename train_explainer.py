import argparse
import matplotlib.pyplot as plt
from torchtext.vocab import GloVe
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.dataset import random_split
from torch.nn import Upsample

from image_datasets import *
from train_helpers import set_bn_eval, CSMRLoss
from model_loader import setup_explainer
import wandb
import pathlib
from typing import Optional, Dict


def train_one_epoch(epoch, model, loss_fn, optimizer, train_loader, embeddings, output_path, experiment_name,
                    train_label_idx, k=5):
    print("\nEpoch {} starting.".format(epoch))
    epoch_loss = 0.0
    num_batch = len(train_loader)
    correct = 0.0
    top_k_correct = 0.0
    model.train()
    model.apply(set_bn_eval)
    interp_on_fly = False
    for batch_index, batch in enumerate(train_loader):
        # DEBUG
        if batch_index == 1:
            break
        # DEBUG
        data, target, mask = batch[0].cuda(), batch[1].squeeze(0).cuda(), batch[2].squeeze(0).cuda()
        predict = data.clone()
        for name, module in model._modules.items():
            if name == 'fc':
                predict = torch.flatten(predict, 1) # flatten GAPed vector
            predict = module(predict)
            if name == args.layer:
                if torch.sum(mask) > 0:
                    # TODO: check if this is faster than implementing it in the dataloader, should be faster with batch_size >> num_workers
                    # if the sizes don't align, set condition for resizing to true on first iteration
                    if batch_index == 0:
                        if mask.shape[-1] != predict.shape[-1]:
                            interp_on_fly = True
                            interpolate = Upsample(size=(predict.shape[2], predict.shape[3]), mode='bilinear', align_corners=False)
                    if interp_on_fly:
                        predict = predict * interpolate(mask)
                    else:
                        predict = predict * mask
                else:
                    continue

        loss = loss_fn(predict, target, embeddings, train_label_idx)
        sorted_predict = torch.argsort(torch.mm(predict, embeddings) /
                                       torch.mm(torch.sqrt(torch.sum(predict ** 2, dim=1, keepdim=True)),
                                                torch.sqrt(torch.sum(embeddings ** 2,
                                                                     dim=0, keepdim=True))),
                                       dim=1, descending=True)[:, :k]
        for i, pred in enumerate(sorted_predict):
            correct += target[i, pred[0]].detach().item()
            top_k_correct += (torch.sum(target[i, pred]) > 0).detach().item()

        optimizer.zero_grad()
        if loss.requires_grad:
            loss.backward()
            optimizer.step()

        if batch_index % 10 == 0:
            train_log = 'Epoch {:2d}\tLoss: {:.6f}\tTrain: [{:4d}/{:4d} ({:.0f}%)]'.format(
                epoch, loss.cpu().item(),
                batch_index, num_batch,
                100. * batch_index / num_batch)
            print(train_log)
            if args.wandb:
                wandb.log({'Iter_Train_Loss': loss})

        if batch_index % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, os.path.join(output_path,
                            'ckpt_tmp.pth.tar'))

        epoch_loss += loss.data.detach().item()
        torch.cuda.empty_cache()

    epoch_loss /= len(train_loader.dataset)
    train_acc = correct / (len(train_loader) * train_loader.batch_size) * 100
    train_top_k_acc = top_k_correct / (len(train_loader) * train_loader.batch_size * k) * 100
    print()
    print("Train average loss: {:.6f}\t".format(epoch_loss))
    print("Train top-1 accuracy: {:.2f}%".format(train_acc))
    print("Train top-5 accuracy: {:.2f}%".format(train_top_k_acc))
    return epoch_loss, train_acc


def validate(model, loss_fn, valid_loader, embeddings, train_label_idx, k=5):
    model.eval()
    valid_loss = 0
    correct = 0.0
    top_k_correct = 0.0
    interp_on_fly = False
    for batch_index, batch in enumerate(valid_loader):
        with torch.no_grad():
            data, target, mask = batch[0].cuda(), batch[1].squeeze(0).cuda(), batch[2].squeeze(0).cuda()
            predict = data.clone()
            for name, module in model._modules.items():
                if name == 'classifier' or name == 'fc':
                    if args.model == 'mobilenet':
                        predict = torch.mean(predict, dim=[2, 3])
                    else:
                        predict = torch.flatten(predict, 1)
                predict = module(predict)
                if name == args.layer:
                    # predict = predict * mask
                    # if the sizes don't align, set condition for resizing to true on first iteration
                    if batch_index == 0:
                        if mask.shape[-1] != predict.shape[-1]:
                            interp_on_fly = True
                            interpolate = Upsample(size=(predict.shape[2], predict.shape[3]), mode='bilinear',
                                                   align_corners=False)
                    if interp_on_fly:
                        predict = predict * interpolate(mask)
                    else:
                        predict = predict * mask
            sorted_predict = torch.argsort(torch.mm(predict, embeddings) /
                                           torch.mm(torch.sqrt(torch.sum(predict ** 2, dim=1, keepdim=True)),
                                                    torch.sqrt(torch.sum(embeddings ** 2,dim=0, keepdim=True))),
                                           dim=1, descending=True)[:, :k]
            for i, pred in enumerate(sorted_predict):
                correct += target[i, pred[0]].detach().item()
                top_k_correct += (torch.sum(target[i, pred]) > 0).detach().item()

            valid_loss += loss_fn(predict, target, embeddings, train_label_idx).data.detach().item()
            # valid_loss += loss_fn(predict, target[:, train_label_idx], embeddings, train_label_idx).data.detach().item()
        torch.cuda.empty_cache()
        # break

    valid_loss /= len(valid_loader.dataset)
    valid_acc = correct / (len(valid_loader) * valid_loader.batch_size) * 100
    valid_top_k_acc = top_k_correct / (len(valid_loader) * valid_loader.batch_size * k) * 100
    print('Valid average loss: {:.6f}\t'.format(valid_loss))
    print("Valid top-1 accuracy: {:.2f}%".format(valid_acc))
    print("Valid top-5 accuracy: {:.2f}%".format(valid_top_k_acc))
    return valid_loss, valid_acc


def main(args, train_rate=0.9):
    if not args.name:
        args.name = 'vsf_%s_%s_%s_%.1f' % (args.refer, args.model, args.layer, args.anno_rate)
    if args.random:
        args.name += '_random'

    args.save_dir = args.save_dir + '/' + args.name + '/'
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)


    if args.wandb:
        wandb_id_file_path = pathlib.Path(args.save_dir + '/'  + 'runid.txt')
        print('Creating new wandb instance...', wandb_id_file_path)
        run = wandb.init(project="temporal_scale", name=args.name, config=args)
        wandb_id_file_path.write_text(str(run.id))

    word_embedding = GloVe(name='6B', dim=args.word_embedding_dim)
    torch.cuda.empty_cache()

    model = setup_explainer(args, random_feature=args.random)
    parameters = model.fc.parameters()
    model = model.cuda()


    optimizer = torch.optim.Adam(parameters, lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, verbose=True)

    if args.refer == 'vg':
        dataset = VisualGenome(root_dir='data', transform=data_transforms['val'])
        datasets = {}
        train_size = int(train_rate * len(dataset))
        test_size = len(dataset) - train_size
        torch.manual_seed(0)
        datasets['train'], datasets['val'] = random_split(dataset, [train_size, test_size])
        label_index_file = os.path.join('./data/vg', "vg_labels.pkl")
        with open(label_index_file, 'rb') as f:
            labels = pickle.load(f)
        label_index = []
        for label in labels:
            label_index.append(word_embedding.stoi[label])
        np.random.seed(0)
        train_label_index = np.random.choice(range(len(label_index)), int(len(label_index) * args.anno_rate))
        word_embeddings_vec = word_embedding.vectors[label_index].T.cuda()
    elif args.refer == 'coco':
        datasets = {'val': MyCocoSegmentation(root='./data/coco/val2017',
                                              annFile='./data/coco/annotations/instances_val2017.json',
                                              transform=data_transforms['val']),
                    'train': MyCocoSegmentation(root='./data/coco/train2017',
                                                annFile='./data/coco/annotations/instances_train2017.json',
                                                transform=data_transforms['train'])}
        label_embedding_file = "./data/coco/coco_label_embedding.pth"
        label_embedding = torch.load(label_embedding_file)
        label_index = list(label_embedding['itos'].keys())

        word_embeddings_vec = word_embedding.vectors[label_index].T.cuda() # saving all word embeddings of classes into an array
        train_label_index = None
        # train_label_index = np.ones(len(label_index))
    else:
        raise NotImplementedError

    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=args.batch_size,
                                                  shuffle=True, num_workers=args.num_workers)
                   for x in ['train', 'val']}

    loss_fn = CSMRLoss(margin=args.margin)
    print("Model setup...")

    # Train and validate
    best_valid_loss = 99999999.
    train_accuracies = []
    valid_accuracies = []
    with open(os.path.join(args.save_dir, 'valid.txt'), 'w') as f:
        for epoch in range(args.epochs):
            train_loss, train_acc = train_one_epoch(epoch, model, loss_fn, optimizer, dataloaders['train'],
                                                    word_embeddings_vec, args.save_dir, args.name, train_label_index)
            ave_valid_loss, valid_acc = validate(model, loss_fn, dataloaders['val'],
                                                 word_embeddings_vec, train_label_index)

            if args.wandb:
                wandb.log({'Epoch': epoch})
                wandb.log({'Epoch': epoch, 'Epoch_Ave_Train_Loss': train_loss})
                wandb.log({'Epoch': epoch, 'Epoch_Ave_Train_Acc': train_acc})
                wandb.log({'Epoch': epoch, 'Epoch_Ave_Valid_Loss': ave_valid_loss})
                wandb.log({'Epoch': epoch, 'Epoch_Ave_Valid_Acc': valid_acc})
                wandb.log({'Epoch': epoch, 'LR': optimizer.param_groups[0]['lr']})

            train_accuracies.append(train_acc)
            valid_accuracies.append(valid_acc)
            scheduler.step(ave_valid_loss)
            f.write('epoch: %d\n' % epoch)
            f.write('train loss: %f\n' % train_loss)
            f.write('train accuracy: %f\n' % train_acc)
            f.write('validation loss: %f\n' % ave_valid_loss)
            f.write('validation accuracy: %f\n' % valid_acc)

            if ave_valid_loss < best_valid_loss:
                best_valid_loss = ave_valid_loss
                print('==> new checkpoint saved')
                f.write('==> new checkpoint saved\n')
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, os.path.join(args.save_dir, 'ckpt_best.pth.tar'))
                plt.figure()
                plt.plot(train_loss, '-o', label='train')
                plt.plot(ave_valid_loss, '-o', label='valid')
                plt.xlabel('Epoch')
                plt.ylabel('Loss (')
                plt.legend(loc='upper right')
                plt.savefig(os.path.join(args.save_dir, 'losses.png'))
                plt.close()

    wandb.run.summary["best_validation_loss"] = best_valid_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--num-classes', type=int, default=2)
    parser.add_argument('--num-workers', type=int, default=16)
    parser.add_argument('--word-embedding-dim', type=int, default=300)
    parser.add_argument('--save-dir', type=str, default='./outputs')
    parser.add_argument('--save-every', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--random', type=bool, default=False, help='Use randomly initialized models instead of pretrained feature extractors')
    parser.add_argument('--wandb', type=bool, default=True, help='Use wandb for logging')
    parser.add_argument('--layer', type=str, default='layer4', help='target layer')
    parser.add_argument('--classifier_name', type=str, default='fc', help='name of classifier layer')
    parser.add_argument('--model', type=str, default='resnet50', help='target network')
    parser.add_argument('--refer', type=str, default='coco', choices=('vg', 'coco'), help='reference dataset')
    parser.add_argument('--pretrain', type=str, default=None, help='path to the pretrained model')
    parser.add_argument('--name', type=str, default='debug', help='experiment name')
    parser.add_argument('--anno-rate', type=float, default=0.1, help='fraction of concepts used for supervision')
    parser.add_argument('--margin', type=float, default=1., help='hyperparameter for margin ranking loss')
    args = parser.parse_args()
    print(args)

    main(args)
