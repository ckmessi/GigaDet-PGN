from __future__ import absolute_import
import time
import torch
from tqdm import tqdm

from pgn.core import model_loader, detect
from pgn.datasets.voc import VOCDataset, VOCTestDataset
from pgn.datasets.panda import PandaImageAndLabelDataset
from pgn.utils import array_tool, eval_tool
from pgn.utils import misc, parser

from pgn.utils.logger import DEFAULT_LOGGER


def eval_recall(cfg, test_data_loader, trainer, max_test_count=10000):
    DEFAULT_LOGGER.info(f"[Train] start evaluate...")

    # pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    # gt_bboxes, gt_labels, gt_difficults = list(), list(), list()

    pred_patches_array = list()
    gt_bboxes_array = list()

    # Iterate data_loader
    for batch_index, elements in tqdm(enumerate(test_data_loader), total=min(max_test_count, len(test_data_loader))):
        (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) = misc.get_test_batch_elements(elements, cfg.DATASETS.DATASET_TYPE)
        imgs = imgs.to(trainer.device).float()

        features = trainer.faster_rcnn.extractor(imgs)
        anchor, patch_counts = trainer.faster_rcnn.pgn(features, sizes, scale=1)

        patch_count = patch_counts[0]
        cur_gt_bboxes_ = gt_bboxes_[0]
        anchor_top_k, patch_count_top_k = detect.get_top_k_pgn_count_anchor(imgs[0],
                                                                            anchor,
                                                                            patch_count,
                                                                            top_k=512,
                                                                            pgn_patch_iou_threshold=0.2)


        # Note: both in resized image size
        pred_patches_array.append(anchor_top_k)
        gt_bboxes_array.append(cur_gt_bboxes_)

        if batch_index == max_test_count - 1:
            break

    # eval
    DEFAULT_LOGGER.info(f"[Train] Finish forward, and start to calculate bboxes recall...")
    recall_dict = eval_tool.calc_bboxes_recall_in_patches(pred_patches_array, gt_bboxes_array)
    DEFAULT_LOGGER.info(f"[Train] evaluate finished...")
    result = {
        'recall_dict': recall_dict,
        'recall': recall_dict[64]
    }
    return result


def get_data_loader(cfg):

    DEFAULT_LOGGER.info(f"using cfg uid={cfg.INFO.UID}")

    dataset_type = cfg.DATASETS.DATASET_TYPE
    if dataset_type == 'PANDA':
        train_dataset = PandaImageAndLabelDataset(cfg, cfg.DATASETS.DATASET_ROOT, split='train', target_key=cfg.DATASETS.PANDA.PERSON_KEY)
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=1,
                                                        shuffle=True,
                                                        num_workers=0,
                                                        pin_memory=True,
                                                        collate_fn=PandaImageAndLabelDataset.collate_fn
                                                        )
        test_dataset = PandaImageAndLabelDataset(cfg, cfg.DATASETS.DATASET_ROOT, split='test', target_key=cfg.DATASETS.PANDA.PERSON_KEY)
        test_data_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=1,
                                            num_workers=0,
                                            shuffle=False,
                                            pin_memory=True,
                                            collate_fn=PandaImageAndLabelDataset.collate_fn
                                            )
    else:
        # default voc scenario
        train_dataset = VOCDataset(cfg)
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=1,
                                             shuffle=True,
                                             # pin_memory=True,
                                             num_workers=0)
        testset = VOCTestDataset(cfg)
        test_data_loader = torch.utils.data.DataLoader(testset,
                                            batch_size=1,
                                            num_workers=0,
                                            shuffle=False,
                                            pin_memory=True
                                            )
    return train_data_loader, test_data_loader



def train(args):

    # opt._parse(args.kwargs)

    cfg = parser.setup_cfg(args)

    DEFAULT_LOGGER.debug('[Train] DEBUG log level enabled.')

    DEFAULT_LOGGER.info('start loading data loader...')
    train_data_loader, test_data_loader = get_data_loader(cfg)
    DEFAULT_LOGGER.info('finish loading data loader..')

    trainer = model_loader.build_pgn_trainer(cfg)

    device = misc.get_target_device()

    if args.eval_only:
        # eval
        eval_result = eval_recall(cfg, test_data_loader, trainer)

        # trainer.plot_eval_info(eval_result)
        trainer.output_eval_info(eval_result, title="eval_only")

        return

    max_epoch = cfg.SOLVER.MAX_EPOCH
    for epoch in range(1, max_epoch + 1):
        # train
        trainer.reset_meters()
        # print tqdm title for next epoch
        tqdm_title = ('\n' + '%12s' * 3) % ('Epoch', 'gpu_mem', 'count_loss')
        print(tqdm_title)
        progress_bar = tqdm(enumerate(train_data_loader), total=len(train_data_loader), ncols=108)
        t1 = 0
        for index, elements in progress_bar:
            # prepare data
            t0 = time.time()
            DEFAULT_LOGGER.debug(f"[PGN] data loader get item cost {time.time() - t1}")
            (img, bbox_, label_, scale) = misc.get_train_batch_elements(elements, cfg.DATASETS.DATASET_TYPE)
            DEFAULT_LOGGER.debug(f"[PGN] get_batch cost {time.time() - t0}")
            img, bbox, label = img.to(device).float(), bbox_.to(device), label_.to(device)
            scale = array_tool.scalar(scale)
            # train step
            DEFAULT_LOGGER.debug(f"[PGN] to device cost {time.time() - t0}")
            trainer.train_step(img, bbox, label, scale)
            DEFAULT_LOGGER.debug(f"[PGN] train_step cost {time.time() - t0}")
            # log/print
            trainer.output_step_info(cfg, progress_bar, epoch)
                # trainer.plot_train_info(opt, img, bbox_, label_)
            DEFAULT_LOGGER.debug(f"[PGN] plot step cost {time.time() - t0}")
            t1 = time.time()

        DEFAULT_LOGGER.info(trainer.get_average_meter_data())
        trainer.adjust_learning_rate(cfg, epoch)

        # eval
        eval_result = eval_recall(cfg, test_data_loader, trainer)

        # trainer.plot_eval_info(eval_result)
        trainer.output_eval_info(eval_result, title=f"evaluate on epoch {epoch}")

        # save
        trainer.save_checkpoint(cfg, eval_result, epoch, metric_key='recall')


    DEFAULT_LOGGER.info(f"training complete.\n\n")


if __name__ == '__main__':

    args = parser.default_argument_parser().parse_args()
    train(args)

    # train(voc_data_dir="D:/Data/Dataset/PascalVOC/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007")
    # train(dataset_type='panda', panda_data_dir="D:/Data/Dataset/PANDA_DATASET/PANDA/PANDA_IMAGE/image_train/", panda_annotation_path='D:/Data/Dataset/PANDA_DATASET/PANDA/PANDA_IMAGE/image_annos/person_bbox_train.json')
    # train(voc_data_dir="/data/chenkai/dataset/public/VOCData/VOCdevkit/VOC2007/")
