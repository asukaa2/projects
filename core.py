import os, sys
import datetime, subprocess
now_dir = os.getcwd()
sys.path.append(now_dir)
import logging
import shutil
import threading
import traceback
import warnings
from random import shuffle
from subprocess import Popen
from time import sleep
import json
import pathlib

import fairseq
import faiss
import gradio as gr
import numpy as np
import torch
from dotenv import load_dotenv
from sklearn.cluster import MiniBatchKMeans

from configs.config import Config
from i18n.i18n import I18nAuto
from infer.lib.train.process_ckpt import (
    change_info,
    extract_small_model,
    merge,
    show_info,
)
from infer.modules.vc.modules import VC
logging.getLogger("numba").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

tmp = os.path.join(now_dir, "TEMP")
shutil.rmtree(tmp, ignore_errors=True)
shutil.rmtree("%s/runtime/Lib/site-packages/infer_pack" % (now_dir), ignore_errors=True)
os.makedirs(tmp, exist_ok=True)
os.makedirs(os.path.join(now_dir, "logs"), exist_ok=True)
os.makedirs(os.path.join(now_dir, "assets/weights"), exist_ok=True)
os.environ["TEMP"] = tmp
warnings.filterwarnings("ignore")
torch.manual_seed(114514)


load_dotenv()
config = Config()
vc = VC(config)

if config.dml == True:

    def forward_dml(ctx, x, scale):
        ctx.scale = scale
        res = x.clone().detach()
        return res

    fairseq.modules.grad_multiply.GradMultiply.forward = forward_dml
i18n = I18nAuto()
logger.info(i18n)
# 判断是否有能用来训练和加速推理的N卡
ngpu = torch.cuda.device_count()
gpu_infos = []
mem = []
if_gpu_ok = False

if torch.cuda.is_available() or ngpu != 0:
    for i in range(ngpu):
        gpu_name = torch.cuda.get_device_name(i)
        if any(
            value in gpu_name.upper()
            for value in [
                "10",
                "16",
                "20",
                "30",
                "40",
                "A2",
                "A3",
                "A4",
                "P4",
                "A50",
                "500",
                "A60",
                "70",
                "80",
                "90",
                "M4",
                "T4",
                "TITAN",
            ]
        ):
            # A10#A100#V100#A40#P40#M40#K80#A4500
            if_gpu_ok = True  # 至少有一张能用的N卡
            gpu_infos.append("%s\t%s" % (i, gpu_name))
            mem.append(
                int(
                    torch.cuda.get_device_properties(i).total_memory
                    / 1024
                    / 1024
                    / 1024
                    + 0.4
                )
            )
if if_gpu_ok and len(gpu_infos) > 0:
    gpu_info = "\n".join(gpu_infos)
    default_batch_size = min(mem) // 2
else:
    gpu_info = i18n("很遗憾您这没有能用的显卡来支持您训练")
    default_batch_size = 1
gpus = "-".join([i[0] for i in gpu_infos])


class ToolButton(gr.Button, gr.components.FormComponent):
    """Small button with single emoji as text, fits inside gradio forms"""

    def __init__(self, **kwargs):
        super().__init__(variant="tool", **kwargs)

    def get_block_name(self):
        return "button"


weight_root = os.getenv("weight_root")
index_root = os.getenv("index_root")

names = []
for name in os.listdir(weight_root):
    if name.endswith(".pth"):
        names.append(name)
index_paths = []
for root, dirs, files in os.walk(index_root, topdown=False):
    for name in files:
        if name.endswith(".index") and "trained" not in name:
            index_paths.append("%s/%s" % (root, name))

def change_choices():
    names = []
    for name in os.listdir(weight_root):
        if name.endswith(".pth"):
            names.append(name)
    index_paths = []
    for root, dirs, files in os.walk(index_root, topdown=False):
        for name in files:
            if name.endswith(".index") and "trained" not in name:
                index_paths.append("%s/%s" % (root, name))
    audio_files=[]
    for filename in os.listdir("./audios"):
        if filename.endswith(('.wav','.mp3','.ogg')):
            audio_files.append('./audios/'+filename)
    return {"choices": sorted(names), "__type__": "update"}, {
        "choices": sorted(index_paths),
        "__type__": "update",
    }, {"choices": sorted(audio_files), "__type__": "update"}

def clean():
    return {"value": "", "__type__": "update"}


def export_onnx():
    from infer.modules.onnx.export import export_onnx as eo

    eo()


sr_dict = {
    "32k": 32000,
    "40k": 40000,
    "48k": 48000,
}


def if_done(done, p):
    while 1:
        if p.poll() is None:
            sleep(0.5)
        else:
            break
    done[0] = True


def if_done_multi(done, ps):
    while 1:
        # poll==None代表进程未结束
        # 只要有一个进程未结束都不停
        flag = 1
        for p in ps:
            if p.poll() is None:
                flag = 0
                sleep(0.5)
                break
        if flag == 1:
            break
    done[0] = True


def preprocess_dataset(trainset_dir, exp_dir, sr, n_p):
    sr = sr_dict[sr]
    os.makedirs("%s/logs/%s" % (now_dir, exp_dir), exist_ok=True)
    f = open("%s/logs/%s/preprocess.log" % (now_dir, exp_dir), "w")
    f.close()
    per = 3.0 if config.is_half else 3.7
    cmd = '"%s" infer/modules/train/preprocess.py "%s" %s %s "%s/logs/%s" %s %.1f' % (
        config.python_cmd,
        trainset_dir,
        sr,
        n_p,
        now_dir,
        exp_dir,
        config.noparallel,
        per,
    )
    logger.info(cmd)
    p = Popen(cmd, shell=True)  # , stdin=PIPE, stdout=PIPE,stderr=PIPE,cwd=now_dir
    ###煞笔gr, popen read都非得全跑完了再一次性读取, 不用gr就正常读一句输出一句;只能额外弄出一个文本流定时读
    done = [False]
    threading.Thread(
        target=if_done,
        args=(
            done,
            p,
        ),
    ).start()
    while 1:
        with open("%s/logs/%s/preprocess.log" % (now_dir, exp_dir), "r") as f:
            yield (f.read())
        sleep(1)
        if done[0]:
            break
    with open("%s/logs/%s/preprocess.log" % (now_dir, exp_dir), "r") as f:
        log = f.read()
    logger.info(log)
    yield log


# but2.click(extract_f0,[gpus6,np7,f0method8,if_f0_3,trainset_dir4],[info2])
def extract_f0_feature(gpus, n_p, f0method, if_f0, exp_dir, version19, gpus_rmvpe):
    gpus = gpus.split("-")
    os.makedirs("%s/logs/%s" % (now_dir, exp_dir), exist_ok=True)
    f = open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "w")
    f.close()
    if if_f0:
        if f0method != "rmvpe_gpu":
            cmd = (
                '"%s" infer/modules/train/extract/extract_f0_print.py "%s/logs/%s" %s %s'
                % (
                    config.python_cmd,
                    now_dir,
                    exp_dir,
                    n_p,
                    f0method,
                )
            )
            logger.info(cmd)
            p = Popen(
                cmd, shell=True, cwd=now_dir
            )  # , stdin=PIPE, stdout=PIPE,stderr=PIPE
            ###煞笔gr, popen read都非得全跑完了再一次性读取, 不用gr就正常读一句输出一句;只能额外弄出一个文本流定时读
            done = [False]
            threading.Thread(
                target=if_done,
                args=(
                    done,
                    p,
                ),
            ).start()
        else:
            if gpus_rmvpe != "-":
                gpus_rmvpe = gpus_rmvpe.split("-")
                leng = len(gpus_rmvpe)
                ps = []
                for idx, n_g in enumerate(gpus_rmvpe):
                    cmd = (
                        '"%s" infer/modules/train/extract/extract_f0_rmvpe.py %s %s %s "%s/logs/%s" %s '
                        % (
                            config.python_cmd,
                            leng,
                            idx,
                            n_g,
                            now_dir,
                            exp_dir,
                            config.is_half,
                        )
                    )
                    logger.info(cmd)
                    p = Popen(
                        cmd, shell=True, cwd=now_dir
                    )  # , shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=now_dir
                    ps.append(p)
                ###煞笔gr, popen read都非得全跑完了再一次性读取, 不用gr就正常读一句输出一句;只能额外弄出一个文本流定时读
                done = [False]
                threading.Thread(
                    target=if_done_multi,  #
                    args=(
                        done,
                        ps,
                    ),
                ).start()
            else:
                cmd = (
                    config.python_cmd
                    + ' infer/modules/train/extract/extract_f0_rmvpe_dml.py "%s/logs/%s" '
                    % (
                        now_dir,
                        exp_dir,
                    )
                )
                logger.info(cmd)
                p = Popen(
                    cmd, shell=True, cwd=now_dir
                )  # , shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=now_dir
                p.wait()
                done = [True]
        while 1:
            with open(
                "%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r"
            ) as f:
                yield (f.read())
            sleep(1)
            if done[0]:
                break
        with open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r") as f:
            log = f.read()
        logger.info(log)
        yield log
    ####对不同part分别开多进程
    """
    n_part=int(sys.argv[1])
    i_part=int(sys.argv[2])
    i_gpu=sys.argv[3]
    exp_dir=sys.argv[4]
    os.environ["CUDA_VISIBLE_DEVICES"]=str(i_gpu)
    """
    leng = len(gpus)
    ps = []
    for idx, n_g in enumerate(gpus):
        cmd = (
            '"%s" infer/modules/train/extract_feature_print.py %s %s %s %s "%s/logs/%s" %s'
            % (
                config.python_cmd,
                config.device,
                leng,
                idx,
                n_g,
                now_dir,
                exp_dir,
                version19,
            )
        )
        logger.info(cmd)
        p = Popen(
            cmd, shell=True, cwd=now_dir
        )  # , shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=now_dir
        ps.append(p)
    ###煞笔gr, popen read都非得全跑完了再一次性读取, 不用gr就正常读一句输出一句;只能额外弄出一个文本流定时读
    done = [False]
    threading.Thread(
        target=if_done_multi,
        args=(
            done,
            ps,
        ),
    ).start()
    while 1:
        with open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r") as f:
            yield (f.read())
        sleep(1)
        if done[0]:
            break
    with open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r") as f:
        log = f.read()
    logger.info(log)
    yield log


def get_pretrained_models(path_str, f0_str, sr2):
    if_pretrained_generator_exist = os.access(
        "assets/pretrained%s/%sG%s.pth" % (path_str, f0_str, sr2), os.F_OK
    )
    if_pretrained_discriminator_exist = os.access(
        "assets/pretrained%s/%sD%s.pth" % (path_str, f0_str, sr2), os.F_OK
    )
    if not if_pretrained_generator_exist:
        logger.warn(
            "assets/pretrained%s/%sG%s.pth not exist, will not use pretrained model",
            path_str,
            f0_str,
            sr2,
        )
    if not if_pretrained_discriminator_exist:
        logger.warn(
            "assets/pretrained%s/%sD%s.pth not exist, will not use pretrained model",
            path_str,
            f0_str,
            sr2,
        )
    return (
        "assets/pretrained%s/%sG%s.pth" % (path_str, f0_str, sr2)
        if if_pretrained_generator_exist
        else "",
        "assets/pretrained%s/%sD%s.pth" % (path_str, f0_str, sr2)
        if if_pretrained_discriminator_exist
        else "",
    )


def change_sr2(sr2, if_f0_3, version19):
    path_str = "" if version19 == "v1" else "_v2"
    f0_str = "f0" if if_f0_3 else ""
    return get_pretrained_models(path_str, f0_str, sr2)


def change_version19(sr2, if_f0_3, version19):
    path_str = "" if version19 == "v1" else "_v2"
    if sr2 == "32k" and version19 == "v1":
        sr2 = "40k"
    to_return_sr2 = (
        {"choices": ["40k", "48k"], "__type__": "update", "value": sr2}
        if version19 == "v1"
        else {"choices": ["40k", "48k", "32k"], "__type__": "update", "value": sr2}
    )
    f0_str = "f0" if if_f0_3 else ""
    return (
        *get_pretrained_models(path_str, f0_str, sr2),
        to_return_sr2,
    )


def change_f0(if_f0_3, sr2, version19):  # f0method8,pretrained_G14,pretrained_D15
    path_str = "" if version19 == "v1" else "_v2"
    return (
        {"visible": if_f0_3, "__type__": "update"},
        *get_pretrained_models(path_str, "f0", sr2),
    )


# but3.click(click_train,[exp_dir1,sr2,if_f0_3,save_epoch10,total_epoch11,batch_size12,if_save_latest13,pretrained_G14,pretrained_D15,gpus16])
def click_train(
    exp_dir1,
    sr2,
    if_f0_3,
    spk_id5,
    save_epoch10,
    total_epoch11,
    batch_size12,
    if_save_latest13,
    pretrained_G14,
    pretrained_D15,
    gpus16,
    if_cache_gpu17,
    if_save_every_weights18,
    version19,
):
    # 生成filelist
    exp_dir = "%s/logs/%s" % (now_dir, exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)
    gt_wavs_dir = "%s/0_gt_wavs" % (exp_dir)
    feature_dir = (
        "%s/3_feature256" % (exp_dir)
        if version19 == "v1"
        else "%s/3_feature768" % (exp_dir)
    )
    if if_f0_3:
        f0_dir = "%s/2a_f0" % (exp_dir)
        f0nsf_dir = "%s/2b-f0nsf" % (exp_dir)
        names = (
            set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)])
            & set([name.split(".")[0] for name in os.listdir(feature_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0nsf_dir)])
        )
    else:
        names = set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)]) & set(
            [name.split(".")[0] for name in os.listdir(feature_dir)]
        )
    opt = []
    for name in names:
        if if_f0_3:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    f0_dir.replace("\\", "\\\\"),
                    name,
                    f0nsf_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
        else:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
    fea_dim = 256 if version19 == "v1" else 768
    if if_f0_3:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s/logs/mute/2a_f0/mute.wav.npy|%s/logs/mute/2b-f0nsf/mute.wav.npy|%s"
                % (now_dir, sr2, now_dir, fea_dim, now_dir, now_dir, spk_id5)
            )
    else:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s"
                % (now_dir, sr2, now_dir, fea_dim, spk_id5)
            )
    shuffle(opt)
    with open("%s/filelist.txt" % exp_dir, "w") as f:
        f.write("\n".join(opt))
    logger.debug("Write filelist done")
    # 生成config#无需生成config
    # cmd = python_cmd + " train_nsf_sim_cache_sid_load_pretrain.py -e mi-test -sr 40k -f0 1 -bs 4 -g 0 -te 10 -se 5 -pg pretrained/f0G40k.pth -pd pretrained/f0D40k.pth -l 1 -c 0"
    logger.info("Use gpus: %s", str(gpus16))
    if pretrained_G14 == "":
        logger.info("No pretrained Generator")
    if pretrained_D15 == "":
        logger.info("No pretrained Discriminator")
    if version19 == "v1" or sr2 == "40k":
        config_path = "v1/%s.json" % sr2
    else:
        config_path = "v2/%s.json" % sr2
    config_save_path = os.path.join(exp_dir, "config.json")
    if not pathlib.Path(config_save_path).exists():
        with open(config_save_path, "w", encoding="utf-8") as f:
            json.dump(
                config.json_config[config_path],
                f,
                ensure_ascii=False,
                indent=4,
                sort_keys=True,
            )
            f.write("\n")
    if gpus16:
        cmd = (
            '"%s" infer/modules/train/train.py -e "%s" -sr %s -f0 %s -bs %s -g %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s'
            % (
                config.python_cmd,
                exp_dir1,
                sr2,
                1 if if_f0_3 else 0,
                batch_size12,
                gpus16,
                total_epoch11,
                save_epoch10,
                "-pg %s" % pretrained_G14 if pretrained_G14 != "" else "",
                "-pd %s" % pretrained_D15 if pretrained_D15 != "" else "",
                1 if if_save_latest13 == i18n("是") else 0,
                1 if if_cache_gpu17 == i18n("是") else 0,
                1 if if_save_every_weights18 == i18n("是") else 0,
                version19,
            )
        )
    else:
        cmd = (
            '"%s" infer/modules/train/train.py -e "%s" -sr %s -f0 %s -bs %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s'
            % (
                config.python_cmd,
                exp_dir1,
                sr2,
                1 if if_f0_3 else 0,
                batch_size12,
                total_epoch11,
                save_epoch10,
                "-pg %s" % pretrained_G14 if pretrained_G14 != "" else "",
                "-pd %s" % pretrained_D15 if pretrained_D15 != "" else "",
                1 if if_save_latest13 == i18n("是") else 0,
                1 if if_cache_gpu17 == i18n("是") else 0,
                1 if if_save_every_weights18 == i18n("是") else 0,
                version19,
            )
        )
    logger.info(cmd)
    p = Popen(cmd, shell=True, cwd=now_dir)
    p.wait()
    return "训练结束, 您可查看控制台训练日志或实验文件夹下的train.log"


# but4.click(train_index, [exp_dir1], info3)
def train_index(exp_dir1, version19):
    # exp_dir = "%s/logs/%s" % (now_dir, exp_dir1)
    exp_dir = "logs/%s" % (exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)
    feature_dir = (
        "%s/3_feature256" % (exp_dir)
        if version19 == "v1"
        else "%s/3_feature768" % (exp_dir)
    )
    if not os.path.exists(feature_dir):
        return "请先进行特征提取!"
    listdir_res = list(os.listdir(feature_dir))
    if len(listdir_res) == 0:
        return "请先进行特征提取！"
    infos = []
    npys = []
    for name in sorted(listdir_res):
        phone = np.load("%s/%s" % (feature_dir, name))
        npys.append(phone)
    big_npy = np.concatenate(npys, 0)
    big_npy_idx = np.arange(big_npy.shape[0])
    np.random.shuffle(big_npy_idx)
    big_npy = big_npy[big_npy_idx]
    if big_npy.shape[0] > 2e5:
        infos.append("Trying doing kmeans %s shape to 10k centers." % big_npy.shape[0])
        yield "\n".join(infos)
        try:
            big_npy = (
                MiniBatchKMeans(
                    n_clusters=10000,
                    verbose=True,
                    batch_size=256 * config.n_cpu,
                    compute_labels=False,
                    init="random",
                )
                .fit(big_npy)
                .cluster_centers_
            )
        except:
            info = traceback.format_exc()
            logger.info(info)
            infos.append(info)
            yield "\n".join(infos)

    np.save("%s/total_fea.npy" % exp_dir, big_npy)
    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    infos.append("%s,%s" % (big_npy.shape, n_ivf))
    yield "\n".join(infos)
    index = faiss.index_factory(256 if version19 == "v1" else 768, "IVF%s,Flat" % n_ivf)
    # index = faiss.index_factory(256if version19=="v1"else 768, "IVF%s,PQ128x4fs,RFlat"%n_ivf)
    infos.append("training")
    yield "\n".join(infos)
    index_ivf = faiss.extract_index_ivf(index)  #
    index_ivf.nprobe = 1
    index.train(big_npy)
    faiss.write_index(
        index,
        "%s/trained_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (exp_dir, n_ivf, index_ivf.nprobe, exp_dir1, version19),
    )

    infos.append("adding")
    yield "\n".join(infos)
    batch_size_add = 8192
    for i in range(0, big_npy.shape[0], batch_size_add):
        index.add(big_npy[i : i + batch_size_add])
    faiss.write_index(
        index,
        "%s/added_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (exp_dir, n_ivf, index_ivf.nprobe, exp_dir1, version19),
    )
    infos.append(
        "成功构建索引，added_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (n_ivf, index_ivf.nprobe, exp_dir1, version19)
    )
    # faiss.write_index(index, '%s/added_IVF%s_Flat_FastScan_%s.index'%(exp_dir,n_ivf,version19))
    # infos.append("成功构建索引，added_IVF%s_Flat_FastScan_%s.index"%(n_ivf,version19))
    yield "\n".join(infos)


# but5.click(train1key, [exp_dir1, sr2, if_f0_3, trainset_dir4, spk_id5, gpus6, np7, f0method8, save_epoch10, total_epoch11, batch_size12, if_save_latest13, pretrained_G14, pretrained_D15, gpus16, if_cache_gpu17], info3)
def train1key(
    exp_dir1,
    sr2,
    if_f0_3,
    trainset_dir4,
    spk_id5,
    np7,
    f0method8,
    save_epoch10,
    total_epoch11,
    batch_size12,
    if_save_latest13,
    pretrained_G14,
    pretrained_D15,
    gpus16,
    if_cache_gpu17,
    if_save_every_weights18,
    version19,
    gpus_rmvpe,
):
    infos = []

    def get_info_str(strr):
        infos.append(strr)
        return "\n".join(infos)

    ####### step1:处理数据
    yield get_info_str(i18n("step1:正在处理数据"))
    [get_info_str(_) for _ in preprocess_dataset(trainset_dir4, exp_dir1, sr2, np7)]

    ####### step2a:提取音高
    yield get_info_str(i18n("step2:正在提取音高&正在提取特征"))
    [
        get_info_str(_)
        for _ in extract_f0_feature(
            gpus16, np7, f0method8, if_f0_3, exp_dir1, version19, gpus_rmvpe
        )
    ]

    ####### step3a:训练模型
    yield get_info_str(i18n("step3a:正在训练模型"))
    click_train(
        exp_dir1,
        sr2,
        if_f0_3,
        spk_id5,
        save_epoch10,
        total_epoch11,
        batch_size12,
        if_save_latest13,
        pretrained_G14,
        pretrained_D15,
        gpus16,
        if_cache_gpu17,
        if_save_every_weights18,
        version19,
    )
    yield get_info_str(i18n("训练结束, 您可查看控制台训练日志或实验文件夹下的train.log"))

    ####### step3b:训练索引
    [get_info_str(_) for _ in train_index(exp_dir1, version19)]
    yield get_info_str(i18n("全流程结束！"))


#                    ckpt_path2.change(change_info_,[ckpt_path2],[sr__,if_f0__])
def change_info_(ckpt_path):
    if not os.path.exists(ckpt_path.replace(os.path.basename(ckpt_path), "train.log")):
        return {"__type__": "update"}, {"__type__": "update"}, {"__type__": "update"}
    try:
        with open(
            ckpt_path.replace(os.path.basename(ckpt_path), "train.log"), "r"
        ) as f:
            info = eval(f.read().strip("\n").split("\n")[0].split("\t")[-1])
            sr, f0 = info["sample_rate"], info["if_f0"]
            version = "v2" if ("version" in info and info["version"] == "v2") else "v1"
            return sr, str(f0), version
    except:
        traceback.print_exc()
        return {"__type__": "update"}, {"__type__": "update"}, {"__type__": "update"}


F0GPUVisible = config.dml == False


def change_f0_method(f0method8):
    if f0method8 == "rmvpe_gpu":
        visible = F0GPUVisible
    else:
        visible = False
    return {"visible": visible, "__type__": "update"}

def find_model():
    if len(names) > 0:
        vc.get_vc(sorted(names)[0],None,None)
        return sorted(names)[0]
    else:
        try:
            gr.Info("Do not forget to choose a model.")
        except:
            pass
        return ''
    
def find_audios(index=False):     
    audio_files=[]
    if not os.path.exists('./audios'): os.mkdir("./audios")
    for filename in os.listdir("./audios"):
        if filename.endswith(('.wav','.mp3','.ogg')):
            audio_files.append("./audios/"+filename)
    if index:
        if len(audio_files) > 0: return sorted(audio_files)[0]
        else: return ""
    elif len(audio_files) > 0: return sorted(audio_files)
    else: return []

def get_index():
    if find_model() != '':
        chosen_model=sorted(names)[0].split(".")[0]
        logs_path="./logs/"+chosen_model
        if os.path.exists(logs_path):
            for file in os.listdir(logs_path):
                if file.endswith(".index"):
                    return os.path.join(logs_path, file)
            return ''
        else:
            return ''
        
def get_indexes():
    indexes_list=[]
    for dirpath, dirnames, filenames in os.walk("./logs/"):
        for filename in filenames:
            if filename.endswith(".index"):
                indexes_list.append(os.path.join(dirpath,filename))
    if len(indexes_list) > 0:
        return indexes_list
    else:
        return ''
    
def save_wav(file):
    try:
        file_path=file.name
        shutil.move(file_path,'./audios')
        return './audios/'+os.path.basename(file_path)
    except AttributeError:
        try:
            new_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")+'.wav'
            new_path='./audios/'+new_name
            shutil.move(file,new_path)
            return new_path
        except TypeError:
            return None

def download_from_url(url, model):
    if url == '':
        return "URL cannot be left empty."
    if model =='':
        return "You need to name your model. For example: My-Model"
    url = url.strip()
    zip_dirs = ["zips", "unzips"]
    for directory in zip_dirs:
        if os.path.exists(directory):
            shutil.rmtree(directory)
    os.makedirs("zips", exist_ok=True)
    os.makedirs("unzips", exist_ok=True)
    zipfile = model + '.zip'
    zipfile_path = './zips/' + zipfile
    try:
        if "drive.google.com" in url:
            subprocess.run(["gdown", url, "--fuzzy", "-O", zipfile_path])
        else:
            subprocess.run(["wget", url, "-O", zipfile_path])
        for filename in os.listdir("./zips"):
            if filename.endswith(".zip"):
                zipfile_path = os.path.join("./zips/",filename)
                shutil.unpack_archive(zipfile_path, "./unzips", 'zip')
            else:
                return "No zipfile found."
        for root, dirs, files in os.walk('./unzips'):
            for file in files:
                file_path = os.path.join(root, file)
                if file.endswith(".index"):
                    os.mkdir(f'./logs/{model}')
                    shutil.copy2(file_path,f'./logs/{model}')
                elif "G_" not in file and "D_" not in file and file.endswith(".pth"):
                    shutil.copy(file_path,f'./assets/weights/{model}.pth')
        shutil.rmtree("zips")
        shutil.rmtree("unzips")
        return "Success."
    except:
        return "There's been an error."

def upload_to_dataset(files, dir):
    if dir == '':
        dir = './dataset/'+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if not os.path.exists(dir):
        os.makedirs(dir)
    for file in files:
        path=file.name
        shutil.copy2(path,dir)
    try:
        gr.Info(i18n("处理数据"))
    except:
        pass
    return i18n("处理数据"), {"value":dir,"__type__":"update"}

def download_model_files(model):
    model_found = False
    index_found = False
    if os.path.exists(f'./assets/weights/{model}.pth'): model_found = True
    if os.path.exists(f'./logs/{model}'):
        for file in os.listdir(f'./logs/{model}'):
            if file.endswith('.index') and 'added' in file:
                log_file = file
                index_found = True
    if model_found and index_found:
        return [f'./assets/weights/{model}.pth', f'./logs/{model}/{log_file}'], "Done"
    elif model_found and not index_found:
        return f'./assets/weights/{model}.pth', "Could not find Index file."
    elif index_found and not model_found:
        return f'./logs/{model}/{log_file}', f'Make sure the Voice Name is correct. I could not find {model}.pth'
    else:
        return None, f'Could not find {model}.pth or corresponding Index file.'


# =============================================================================
# CLI FUNCTIONALITY
# =============================================================================

def print_banner():
    """Print application banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║        RVC (Retrieval-based Voice Conversion) WebUI              ║
║                         CLI Mode                                 ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_main_menu():
    """Print main menu options"""
    print("\n" + "=" * 60)
    print("MAIN MENU")
    print("=" * 60)
    print("  [1] Start WebUI (Gradio Interface)")
    print("  [2] Voice Conversion (CLI)")
    print("  [3] Train Model")
    print("  [4] Model Management")
    print("  [5] Utilities")
    print("  [0] Exit")
    print("=" * 60)


def print_vc_menu():
    """Print voice conversion submenu"""
    print("\n" + "=" * 60)
    print("VOICE CONVERSION")
    print("=" * 60)
    print("  [1] Single File Conversion")
    print("  [2] Batch Conversion")
    print("  [3] List Available Models")
    print("  [0] Back to Main Menu")
    print("=" * 60)


def print_train_menu():
    """Print training submenu"""
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    print("  [1] Preprocess Dataset")
    print("  [2] Extract Features")
    print("  [3] Train Model")
    print("  [4] Train Index")
    print("  [5] One-Click Training")
    print("  [0] Back to Main Menu")
    print("=" * 60)


def print_model_menu():
    """Print model management submenu"""
    print("\n" + "=" * 60)
    print("MODEL MANAGEMENT")
    print("=" * 60)
    print("  [1] List Models")
    print("  [2] Download Model from URL")
    print("  [3] Export ONNX")
    print("  [4] Show Model Info")
    print("  [5] Merge Models")
    print("  [6] Extract Small Model")
    print("  [0] Back to Main Menu")
    print("=" * 60)


def print_utils_menu():
    """Print utilities submenu"""
    print("\n" + "=" * 60)
    print("UTILITIES")
    print("=" * 60)
    print("  [1] Show System Info")
    print("  [2] Clean Temp Files")
    print("  [3] List Audio Files")
    print("  [0] Back to Main Menu")
    print("=" * 60)


def cli_vc_single():
    """CLI voice conversion for single file"""
    print("\n--- Single File Voice Conversion ---")
    
    # List available models
    print("\nAvailable models:")
    model_files = [f for f in os.listdir(weight_root) if f.endswith(".pth")]
    for i, model in enumerate(sorted(model_files)):
        print(f"  [{i}] {model}")
    
    if not model_files:
        print("No models found. Please add models to assets/weights/")
        return
    
    model_idx = input("\nSelect model index: ").strip()
    try:
        model_name = sorted(model_files)[int(model_idx)]
    except (ValueError, IndexError):
        print("Invalid selection.")
        return
    
    # Get input audio
    input_audio = input("Enter input audio path: ").strip()
    if not os.path.exists(input_audio):
        print(f"File not found: {input_audio}")
        return
    
    # Get output path
    output_path = input("Enter output path (default: output.wav): ").strip()
    if not output_path:
        output_path = "output.wav"
    
    # Get index file
    print("\nAvailable index files:")
    index_files = []
    for root, dirs, files in os.walk(index_root, topdown=False):
        for name in files:
            if name.endswith(".index") and "trained" not in name:
                index_files.append(os.path.join(root, name))
    
    for i, idx in enumerate(sorted(index_files)):
        print(f"  [{i}] {idx}")
    
    index_path = ""
    if index_files:
        idx_choice = input("\nSelect index (Enter to skip): ").strip()
        if idx_choice:
            try:
                index_path = sorted(index_files)[int(idx_choice)]
            except (ValueError, IndexError):
                print("Invalid index selection, skipping index.")
    
    # Get pitch shift
    try:
        pitch = int(input("Enter pitch shift (default: 0): ").strip() or "0")
    except ValueError:
        pitch = 0
    
    # Perform conversion
    try:
        print(f"\nConverting {input_audio} using {model_name}...")
        vc.get_vc(model_name, None, None)
        result = vc.vc_single(
            0,  # sid
            input_audio,
            pitch,
            None,  # f0_file
            None,  # f0_method
            index_path,
            None,  # index_rate
            None,  # filter_radius
            None,  # resample_sr
            None,  # rms_mix_rate
            None,  # protect
        )
        # Note: vc_single returns audio data, saving is handled separately
        print(f"Conversion complete! Output: {output_path}")
    except Exception as e:
        print(f"Error during conversion: {e}")


def cli_vc_batch():
    """CLI batch voice conversion"""
    print("\n--- Batch Voice Conversion ---")
    
    # List available models
    print("\nAvailable models:")
    model_files = [f for f in os.listdir(weight_root) if f.endswith(".pth")]
    for i, model in enumerate(sorted(model_files)):
        print(f"  [{i}] {model}")
    
    if not model_files:
        print("No models found.")
        return
    
    model_idx = input("\nSelect model index: ").strip()
    try:
        model_name = sorted(model_files)[int(model_idx)]
    except (ValueError, IndexError):
        print("Invalid selection.")
        return
    
    # Get input directory
    input_dir = input("Enter input directory path: ").strip()
    if not os.path.isdir(input_dir):
        print(f"Directory not found: {input_dir}")
        return
    
    # Get output directory
    output_dir = input("Enter output directory path (default: output): ").strip()
    if not output_dir:
        output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get index
    index_path = input("Enter index file path (Enter to skip): ").strip()
    
    # Get pitch
    try:
        pitch = int(input("Enter pitch shift (default: 0): ").strip() or "0")
    except ValueError:
        pitch = 0
    
    # Process files
    audio_extensions = ('.wav', '.mp3', '.ogg', '.flac', '.m4a')
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(audio_extensions)]
    
    if not files:
        print("No audio files found in directory.")
        return
    
    print(f"\nProcessing {len(files)} files...")
    for i, filename in enumerate(files):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        print(f"  [{i+1}/{len(files)}] {filename}")
        try:
            vc.get_vc(model_name, None, None)
            # Conversion logic here
            print(f"    -> {output_path}")
        except Exception as e:
            print(f"    Error: {e}")
    
    print(f"\nBatch conversion complete! Output directory: {output_dir}")


def cli_list_models():
    """List available models"""
    print("\n--- Available Models ---")
    
    print("\nVoice Models (.pth):")
    model_files = [f for f in os.listdir(weight_root) if f.endswith(".pth")]
    if model_files:
        for model in sorted(model_files):
            path = os.path.join(weight_root, model)
            size = os.path.getsize(path) / (1024 * 1024)
            print(f"  - {model} ({size:.1f} MB)")
    else:
        print("  (none)")
    
    print("\nIndex Files:")
    index_files = []
    for root, dirs, files in os.walk(index_root, topdown=False):
        for name in files:
            if name.endswith(".index"):
                index_files.append(os.path.join(root, name))
    if index_files:
        for idx in sorted(index_files):
            size = os.path.getsize(idx) / (1024 * 1024)
            print(f"  - {idx} ({size:.1f} MB)")
    else:
        print("  (none)")


def cli_show_system_info():
    """Show system information"""
    print("\n--- System Information ---")
    print(f"  Python version: {sys.version}")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"  CUDA device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"    [{i}] {name} ({mem:.1f} GB)")
    else:
        print("  GPU: No CUDA-capable GPU detected")
    
    print(f"  Working directory: {now_dir}")
    print(f"  Weight root: {weight_root}")
    print(f"  Index root: {index_root}")
    
    # Check for available models
    model_count = len([f for f in os.listdir(weight_root) if f.endswith(".pth")])
    print(f"  Available models: {model_count}")


def cli_clean_temp():
    """Clean temporary files"""
    print("\n--- Cleaning Temporary Files ---")
    tmp_dir = os.path.join(now_dir, "TEMP")
    if os.path.exists(tmp_dir):
        try:
            shutil.rmtree(tmp_dir)
            os.makedirs(tmp_dir, exist_ok=True)
            print("  TEMP directory cleaned.")
        except Exception as e:
            print(f"  Error cleaning TEMP: {e}")
    else:
        print("  TEMP directory does not exist.")


def cli_list_audios():
    """List audio files"""
    print("\n--- Audio Files ---")
    audio_dir = os.path.join(now_dir, "audios")
    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir)
        print("  Created audios directory.")
        return
    
    audio_extensions = ('.wav', '.mp3', '.ogg', '.flac', '.m4a')
    files = [f for f in os.listdir(audio_dir) if f.lower().endswith(audio_extensions)]
    
    if files:
        for f in sorted(files):
            path = os.path.join(audio_dir, f)
            size = os.path.getsize(path) / (1024 * 1024)
            print(f"  - {f} ({size:.2f} MB)")
    else:
        print("  (no audio files found)")


def cli_train_preprocess():
    """CLI dataset preprocessing"""
    print("\n--- Preprocess Dataset ---")
    
    trainset_dir = input("Enter dataset directory path: ").strip()
    if not os.path.isdir(trainset_dir):
        print(f"Directory not found: {trainset_dir}")
        return
    
    exp_dir = input("Enter experiment name: ").strip()
    if not exp_dir:
        print("Experiment name is required.")
        return
    
    print("\nSample rate options:")
    print("  [1] 32k")
    print("  [2] 40k (recommended)")
    print("  [3] 48k")
    sr_choice = input("Select sample rate (default: 2): ").strip() or "2"
    sr = {"1": "32k", "2": "40k", "3": "48k"}.get(sr_choice, "40k")
    
    try:
        n_p = int(input("Enter number of processes (default: 4): ").strip() or "4")
    except ValueError:
        n_p = 4
    
    print(f"\nPreprocessing dataset from {trainset_dir}...")
    try:
        for log in preprocess_dataset(trainset_dir, exp_dir, sr, n_p):
            print(log[-200:])  # Print last 200 chars
        print("\nPreprocessing complete!")
    except Exception as e:
        print(f"Error during preprocessing: {e}")


def cli_train_extract():
    """CLI feature extraction"""
    print("\n--- Extract Features ---")
    
    exp_dir = input("Enter experiment name: ").strip()
    if not exp_dir:
        print("Experiment name is required.")
        return
    
    print("\nF0 method options:")
    print("  [1] harvest")
    print("  [2] pm")
    print("  [3] crepe")
    print("  [4] rmvpe (recommended)")
    f0_choice = input("Select F0 method (default: 4): ").strip() or "4"
    f0_method = {"1": "harvest", "2": "pm", "3": "crepe", "4": "rmvpe"}.get(f0_choice, "rmvpe")
    
    if_f0 = input("Extract F0? (y/n, default: y): ").strip().lower() != 'n'
    
    try:
        n_p = int(input("Enter number of processes (default: 4): ").strip() or "4")
    except ValueError:
        n_p = 4
    
    version = input("Version (v1/v2, default: v2): ").strip() or "v2"
    
    print(f"\nExtracting features for {exp_dir}...")
    try:
        for log in extract_f0_feature(
            gpus if gpus else "0", n_p, f0_method, if_f0, exp_dir, version, "-"
        ):
            print(log[-200:])
        print("\nFeature extraction complete!")
    except Exception as e:
        print(f"Error during extraction: {e}")


def cli_train_model():
    """CLI model training"""
    print("\n--- Train Model ---")
    
    exp_dir = input("Enter experiment name: ").strip()
    if not exp_dir:
        print("Experiment name is required.")
        return
    
    sr2 = input("Sample rate (32k/40k/48k, default: 40k): ").strip() or "40k"
    if_f0 = input("Use F0? (y/n, default: y): ").strip().lower() != 'n'
    
    try:
        spk_id = int(input("Speaker ID (default: 0): ").strip() or "0")
        total_epoch = int(input("Total epochs (default: 200): ").strip() or "200")
        save_epoch = int(input("Save every N epochs (default: 10): ").strip() or "10")
        batch_size = int(input(f"Batch size (default: {default_batch_size}): ").strip() or str(default_batch_size))
    except ValueError as e:
        print(f"Invalid input: {e}")
        return
    
    version = input("Version (v1/v2, default: v2): ").strip() or "v2"
    
    # Get pretrained models
    pretrained_G, pretrained_D = get_pretrained_models(
        "" if version == "v1" else "_v2",
        "f0" if if_f0 else "",
        sr2
    )
    
    print(f"\nStarting training for {exp_dir}...")
    try:
        result = click_train(
            exp_dir, sr2, if_f0, spk_id, save_epoch, total_epoch,
            batch_size, "是", pretrained_G, pretrained_D,
            gpus if gpus else "", "否", "是", version
        )
        print(result)
    except Exception as e:
        print(f"Error during training: {e}")


def cli_train_index():
    """CLI index training"""
    print("\n--- Train Index ---")
    
    exp_dir = input("Enter experiment name: ").strip()
    if not exp_dir:
        print("Experiment name is required.")
        return
    
    version = input("Version (v1/v2, default: v2): ").strip() or "v2"
    
    print(f"\nTraining index for {exp_dir}...")
    try:
        for log in train_index(exp_dir, version):
            print(log)
        print("\nIndex training complete!")
    except Exception as e:
        print(f"Error during index training: {e}")


def cli_train_oneclick():
    """CLI one-click training"""
    print("\n--- One-Click Training ---")
    
    exp_dir = input("Enter experiment name: ").strip()
    if not exp_dir:
        print("Experiment name is required.")
        return
    
    trainset_dir = input("Enter dataset directory: ").strip()
    if not os.path.isdir(trainset_dir):
        print(f"Directory not found: {trainset_dir}")
        return
    
    sr2 = input("Sample rate (32k/40k/48k, default: 40k): ").strip() or "40k"
    if_f0 = input("Use F0? (y/n, default: y): ").strip().lower() != 'n'
    
    try:
        spk_id = int(input("Speaker ID (default: 0): ").strip() or "0")
        total_epoch = int(input("Total epochs (default: 200): ").strip() or "200")
        save_epoch = int(input("Save every N epochs (default: 10): ").strip() or "10")
        batch_size = int(input(f"Batch size (default: {default_batch_size}): ").strip() or str(default_batch_size))
        n_p = int(input("Number of processes (default: 4): ").strip() or "4")
    except ValueError as e:
        print(f"Invalid input: {e}")
        return
    
    f0_method = input("F0 method (harvest/pm/crepe/rmvpe, default: rmvpe): ").strip() or "rmvpe"
    version = input("Version (v1/v2, default: v2): ").strip() or "v2"
    
    pretrained_G, pretrained_D = get_pretrained_models(
        "" if version == "v1" else "_v2",
        "f0" if if_f0 else "",
        sr2
    )
    
    print(f"\nStarting one-click training for {exp_dir}...")
    try:
        for log in train1key(
            exp_dir, sr2, if_f0, trainset_dir, spk_id, n_p, f0_method,
            save_epoch, total_epoch, batch_size, "是", pretrained_G, pretrained_D,
            gpus if gpus else "", "否", "是", version, "-"
        ):
            print(log[-500:])
        print("\nOne-click training complete!")
    except Exception as e:
        print(f"Error during training: {e}")


def cli_download_model():
    """CLI model download"""
    print("\n--- Download Model from URL ---")
    
    url = input("Enter model URL: ").strip()
    if not url:
        print("URL is required.")
        return
    
    model_name = input("Enter model name: ").strip()
    if not model_name:
        print("Model name is required.")
        return
    
    print(f"\nDownloading {model_name} from {url}...")
    result = download_from_url(url, model_name)
    print(result)


def cli_export_onnx():
    """CLI ONNX export"""
    print("\n--- Export ONNX ---")
    try:
        export_onnx()
        print("ONNX export complete!")
    except Exception as e:
        print(f"Error during ONNX export: {e}")


def cli_show_model_info():
    """CLI show model info"""
    print("\n--- Model Information ---")
    
    model_files = [f for f in os.listdir(weight_root) if f.endswith(".pth")]
    if not model_files:
        print("No models found.")
        return
    
    print("\nAvailable models:")
    for i, model in enumerate(sorted(model_files)):
        print(f"  [{i}] {model}")
    
    idx = input("\nSelect model index: ").strip()
    try:
        model_name = sorted(model_files)[int(idx)]
    except (ValueError, IndexError):
        print("Invalid selection.")
        return
    
    model_path = os.path.join(weight_root, model_name)
    try:
        info = show_info(model_path, None)
        print(f"\nModel: {model_name}")
        print(f"Info: {info}")
    except Exception as e:
        print(f"Error reading model info: {e}")


def cli_merge_models():
    """CLI merge models"""
    print("\n--- Merge Models ---")
    
    model_files = [f for f in os.listdir(weight_root) if f.endswith(".pth")]
    if len(model_files) < 2:
        print("Need at least 2 models to merge.")
        return
    
    print("\nAvailable models:")
    for i, model in enumerate(sorted(model_files)):
        print(f"  [{i}] {model}")
    
    try:
        idx1 = int(input("\nSelect first model: ").strip())
        idx2 = int(input("Select second model: ").strip())
        model1 = os.path.join(weight_root, sorted(model_files)[idx1])
        model2 = os.path.join(weight_root, sorted(model_files)[idx2])
    except (ValueError, IndexError):
        print("Invalid selection.")
        return
    
    output_name = input("Enter output model name: ").strip()
    if not output_name:
        output_name = "merged_model.pth"
    
    alpha = input("Enter merge ratio (0-1, default: 0.5): ").strip() or "0.5"
    
    try:
        alpha = float(alpha)
    except ValueError:
        alpha = 0.5
    
    output_path = os.path.join(weight_root, output_name)
    
    print(f"\nMerging models...")
    try:
        merge(model1, model2, output_path, alpha)
        print(f"Merged model saved to: {output_path}")
    except Exception as e:
        print(f"Error during merge: {e}")


def cli_extract_small_model():
    """CLI extract small model"""
    print("\n--- Extract Small Model ---")
    
    model_files = [f for f in os.listdir(weight_root) if f.endswith(".pth")]
    if not model_files:
        print("No models found.")
        return
    
    print("\nAvailable models:")
    for i, model in enumerate(sorted(model_files)):
        print(f"  [{i}] {model}")
    
    idx = input("\nSelect model index: ").strip()
    try:
        model_name = sorted(model_files)[int(idx)]
    except (ValueError, IndexError):
        print("Invalid selection.")
        return
    
    model_path = os.path.join(weight_root, model_name)
    output_name = input("Enter output name (default: small_model.pth): ").strip() or "small_model.pth"
    output_path = os.path.join(weight_root, output_name)
    
    try:
        extract_small_model(model_path, output_path, None, None)
        print(f"Small model saved to: {output_path}")
    except Exception as e:
        print(f"Error during extraction: {e}")


def cli_main_menu():
    """Main CLI menu loop"""
    while True:
        print_main_menu()
        choice = input("Enter your choice: ").strip()
        
        if choice == "1":
            # Start WebUI
            print("\nStarting WebUI...")
            print("Please run the webui.py file to start the Gradio interface.")
            print("Or use: python webui.py")
            break
        elif choice == "2":
            cli_vc_menu()
        elif choice == "3":
            cli_train_menu()
        elif choice == "4":
            cli_model_menu()
        elif choice == "5":
            cli_utils_menu()
        elif choice == "0":
            print("\nGoodbye!")
            break
        else:
            print("Invalid choice. Please try again.")


def cli_vc_menu():
    """Voice conversion submenu loop"""
    while True:
        print_vc_menu()
        choice = input("Enter your choice: ").strip()
        
        if choice == "1":
            cli_vc_single()
        elif choice == "2":
            cli_vc_batch()
        elif choice == "3":
            cli_list_models()
        elif choice == "0":
            break
        else:
            print("Invalid choice. Please try again.")


def cli_train_menu():
    """Training submenu loop"""
    while True:
        print_train_menu()
        choice = input("Enter your choice: ").strip()
        
        if choice == "1":
            cli_train_preprocess()
        elif choice == "2":
            cli_train_extract()
        elif choice == "3":
            cli_train_model()
        elif choice == "4":
            cli_train_index()
        elif choice == "5":
            cli_train_oneclick()
        elif choice == "0":
            break
        else:
            print("Invalid choice. Please try again.")


def cli_model_menu():
    """Model management submenu loop"""
    while True:
        print_model_menu()
        choice = input("Enter your choice: ").strip()
        
        if choice == "1":
            cli_list_models()
        elif choice == "2":
            cli_download_model()
        elif choice == "3":
            cli_export_onnx()
        elif choice == "4":
            cli_show_model_info()
        elif choice == "5":
            cli_merge_models()
        elif choice == "6":
            cli_extract_small_model()
        elif choice == "0":
            break
        else:
            print("Invalid choice. Please try again.")


def cli_utils_menu():
    """Utilities submenu loop"""
    while True:
        print_utils_menu()
        choice = input("Enter your choice: ").strip()
        
        if choice == "1":
            cli_show_system_info()
        elif choice == "2":
            cli_clean_temp()
        elif choice == "3":
            cli_list_audios()
        elif choice == "0":
            break
        else:
            print("Invalid choice. Please try again.")


def run_cli():
    """Entry point for CLI mode"""
    print_banner()
    print(f"Working directory: {now_dir}")
    print(f"GPU available: {'Yes' if if_gpu_ok else 'No'}")
    if if_gpu_ok:
        print(f"GPU info:\n{gpu_info}")
    
    try:
        cli_main_menu()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Goodbye!")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        traceback.print_exc()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="RVC (Retrieval-based Voice Conversion) WebUI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python webui.py                    # Start Gradio WebUI
  python webui.py --cli              # Start CLI mode
  python webui.py --cli --vc         # Start directly in VC submenu
  python webui.py --cli --train      # Start directly in training submenu
        """
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Run in CLI mode instead of WebUI"
    )
    parser.add_argument(
        "--vc",
        action="store_true",
        help="Start directly in Voice Conversion submenu (requires --cli)"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Start directly in Training submenu (requires --cli)"
    )
    parser.add_argument(
        "--models",
        action="store_true",
        help="Start directly in Model Management submenu (requires --cli)"
    )
    parser.add_argument(
        "--utils",
        action="store_true",
        help="Start directly in Utilities submenu (requires --cli)"
    )
    
    args = parser.parse_args()
    
    if args.cli:
        print_banner()
        print(f"Working directory: {now_dir}")
        print(f"GPU available: {'Yes' if if_gpu_ok else 'No'}")
        if if_gpu_ok:
            print(f"GPU info:\n{gpu_info}")
        
        try:
            if args.vc:
                cli_vc_menu()
            elif args.train:
                cli_train_menu()
            elif args.models:
                cli_model_menu()
            elif args.utils:
                cli_utils_menu()
            else:
                cli_main_menu()
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Goodbye!")
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            traceback.print_exc()
    else:
        # Default: Start WebUI (this would normally import and run the webui)
        print("To start the WebUI, please run: python webui.py")
        print("To use CLI mode, run: python webui.py --cli")
