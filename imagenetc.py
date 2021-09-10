from typing import List
from custombench import load_imagenetc
from robustbench.utils import clean_accuracy as accuracy
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model

from torchvision import models as pt_models
import logging

from cifar10c import setup_source, setup_norm, setup_tent
from conf import cfg, load_cfg_fom_args

from custombench import normalize_model

logger = logging.getLogger(__name__)



def evaluate(data_dir):  
    load_cfg_fom_args()
    
    # TODO normalize_model
    #base_model = load_model(cfg.MODEL.ARCH, cfg.CKPT_DIR,
    #                   cfg.CORRUPTION.DATASET, ThreatModel.corruptions).cuda()
    base_model = pt_models.resnet50(pretrained=True).cuda()


    # FIXME
    # See https://github.com/RobustBench/robustbench/blob/5e8980aeb97f04a950ef128a890f8cb45f142f9b/robustbench/model_zoo/imagenet.py
    if cfg.MODEL.NORMALIZE:
        mu = (0.485, 0.456, 0.406)
        sigma = (0.229, 0.224, 0.225)
        base_model = normalize_model(base_model, mu, sigma).cuda()
        

    if cfg.MODEL.ADAPTATION == "source":
        logger.info("test-time adaptation: NONE")
        model = setup_source(base_model)
    if cfg.MODEL.ADAPTATION == "norm":
        logger.info("test-time adaptation: NORM")
        model = setup_norm(base_model)
    if cfg.MODEL.ADAPTATION == "tent":
        logger.info("test-time adaptation: TENT")
        model = setup_tent(base_model)

    # evaluate on each severity and type of corruption in turn
    for severity in cfg.CORRUPTION.SEVERITY:
        for corruption_type in cfg.CORRUPTION.TYPE:
            # reset adaptation for each combination of corruption x severity
            # note: for evaluation protocol, but not necessarily needed
            try:
                model.reset()
                logger.info("resetting model")
            except:
                logger.warning("not resetting model")

            x_test, y_test = load_imagenetc(n_examples=cfg.CORRUPTION.NUM_EX,
                                            severity=severity,
                                            data_path=data_dir,
                                            shuffle=True,
                                            corruption_name=corruption_type,
                                            minibatch_size=256)  # For loading only
        

            acc = accuracy(model, x_test, y_test, cfg.TEST.BATCH_SIZE, device='cuda')
            err = 1. - acc
            logger.info(f"error % [{corruption_type}{severity}]: {err:.2%}")

           
if __name__ == '__main__':
    evaluate(data_dir='./data')

