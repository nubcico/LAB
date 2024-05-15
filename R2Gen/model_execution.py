import numpy as np
import torch
from torchvision import transforms
import main
# from models.r2gen import R2GenModel
from r2gen_model import R2GenModel
from modules.tokenizers import Tokenizer
from modules.dataloaders import R2DataLoader
from modules.metrics import compute_scores
from modules.optimizers import build_optimizer, build_lr_scheduler
from modules.trainer import Trainer
from modules.loss import compute_loss
import modules.utils as utils


def mn():
    args = main.parse_agrs()

    # fix random seeds
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    # create tokenize r
    tokenizer = Tokenizer(args)

    # create data loader
    train_dataloader = R2DataLoader(args, tokenizer, split='train', shuffle=True)
    val_dataloader = R2DataLoader(args, tokenizer, split='val', shuffle=False)
    test_dataloader = R2DataLoader(args, tokenizer, split='test', shuffle=False)

    # build model architecture
    model = R2GenModel(args, tokenizer)

    # get function handles of loss and metrics
    criterion = compute_loss
    metrics = compute_scores

    # build optimizer, learning rate scheduler
    optimizer = build_optimizer(args, model)
    lr_scheduler = build_lr_scheduler(args, optimizer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.load_state_dict((torch.load("model_iu_xray.pth", map_location=device)['state_dict']))
    # model.load_state_dict((torch.load("model_mimic_cxr.pth", map_location=device)['state_dict']))

    model.to(device)
    # print(model.eval())

    # for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(test_dataloader):
    #     if args.dataset_name == "iu_xray":
    #         output = model.forward_iu_xray(images, mode="sample")  # Assuming you want to generate a report
    #     else:
    #         output = model.forward_mimic_cxr(images, mode="sample")
    #     # output = model(images, reports_ids, mode='train')
    #     loss = criterion(output, reports_ids, reports_masks)
    #     optimizer.zero_grad()
    #     loss.backward()
    #     torch.nn.utils.clip_grad_value_(model.parameters(), 0.1)
    #     optimizer.step()
    #     # report = tokenizer.decode(output[0])  # Assuming the output is a token ID sequence
    #     report = tokenizer.decode(output[0].argmax(dim=1).cpu().numpy())
    #     print(images_id)
    #     print(report)




    model.eval()
    trainer = Trainer(model, criterion, metrics, optimizer, args, lr_scheduler, train_dataloader, val_dataloader,
                      test_dataloader)
    model.eval()


    with torch.no_grad():
        test_gts, test_res = [], []
        img = []
        id = 0
        for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(test_dataloader):
            images, reports_ids, reports_masks = images.to(device), reports_ids.to(
                device), reports_masks.to(device)
            output = model(images, mode='sample')
            # loss = criterion(output, reports_ids, reports_masks)
            # test_loss += loss.item()
            # optimizer.zero_grad()
            # loss.backward()
            # log = {'train_loss': test_loss / len(test_dataloader)}

            reports = model.tokenizer.decode_batch(output.cpu().numpy())
            ground_truths = model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
            test_res.extend(reports)
            test_gts.extend(ground_truths)
            img.extend(images_id)
            # print(id)
            # print(images_id)
            # print(reports)
            # print(ground_truths)
            # id=id+1
            # model.eval()
        # compute_scores(ground_truths, reports)


    for id, text in enumerate(test_res):
        print(id)
        print(img[id])
        print(text)
        print(test_gts[id])


    # with torch.no_grad():
    #     test_gts, test_res = [], []
    #     for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(test_dataloader):
    #         images, reports_ids, reports_masks = images.to(device), reports_ids.to(device), reports_masks.to(device)
    #         output = model(images, mode='sample')
    #         reports = model.tokenizer.decode_batch(output.cpu().numpy())
    #         ground_truths = model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
    #         test_res.extend(reports)
    #         test_gts.extend(ground_truths)
        # test_met = trainer.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
        #                                {i: [re] for i, re in enumerate(test_res)})
        # log = {'test_' + k: v for k, v in test_met.items()}
        # print(log)



if __name__ == '__main__':
    mn()
