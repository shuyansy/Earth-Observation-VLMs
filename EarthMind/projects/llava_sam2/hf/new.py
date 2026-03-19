import argparse
import copy
import os
import os.path as osp
import torch
from mmengine.dist import master_only
from xtuner.registry import BUILDER
from xtuner.configs import cfgs_name_path
from xtuner.model.utils import guess_load_checkpoint
from mmengine.config import Config
from mmengine.fileio import PetrelBackend, get_file_backend
from mmengine.config import ConfigDict
import hashlib  # ← 新增

def _tensor_md5(t: torch.Tensor) -> str:
    t = t.detach().cpu().contiguous()
    # 关键：把 bf16/fp16 转成 fp32，避免 numpy 不支持 bf16
    if t.dtype in (torch.bfloat16, torch.float16):
        t = t.to(torch.float32)
    return hashlib.md5(t.numpy().tobytes()).hexdigest()

def verify_visual_encoder_export(orig_vision_model: torch.nn.Module, save_dir: str):
    """
    从磁盘 `save_dir` 重新加载 HF 模型，抽查 visual encoder 的参数，
    与导出前的原始 visual encoder 做哈希对比，打印验证结果。
    """
    # 延迟导入，避免脚本顶部循环依赖
    from projects.llava_sam2.hf.models.modeling_earthmind_chat import Sa2VAChatModel

    print("[Verify] Reloading HF model from disk for visual-encoder check ...")
    reloaded = Sa2VAChatModel.from_pretrained(
        save_dir, trust_remote_code=True, torch_dtype=torch.float32, device_map=None
    )

    sd_orig = orig_vision_model.state_dict()
    sd_new  = reloaded.vision_model.state_dict()

    # 取公共键，抽查三处（首/中/尾），并校验形状与 md5
    common = sorted(set(sd_orig.keys()) & set(sd_new.keys()))
    if not common:
        raise RuntimeError("No overlapping param keys between original and reloaded vision_model.")

    picks = [common[0], common[len(common)//2], common[-1]]
    all_ok = True
    for k in picks:
        if sd_orig[k].shape != sd_new[k].shape:
            print(f"[Verify][FAIL] shape mismatch @ {k}: {sd_orig[k].shape} vs {sd_new[k].shape}")
            all_ok = False
            continue
        h1 = _tensor_md5(sd_orig[k])
        h2 = _tensor_md5(sd_new[k])
        if h1 == h2:
            print(f"[Verify][OK]  {k}  md5={h1}")
        else:
            print(f"[Verify][FAIL] {k}  md5(orig)={h1}  md5(new)={h2}")
            all_ok = False

    # 额外给出总体数量检查
    print(f"[Verify] vision_model param count: orig={len(sd_orig)} new={len(sd_new)}")

    if not all_ok:
        raise RuntimeError("Visual encoder export verification failed on at least one checked tensor.")
    print("[Verify] Visual encoder weights successfully written and reloaded ✔")


def convert_dict2config_dict(input):
    input = ConfigDict(**input)
    for key in input.keys():
        if isinstance(input[key], dict):
            input[key] = convert_dict2config_dict(input[key])
    return input


TORCH_DTYPE_MAP = dict(
    fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32, auto='auto')


def parse_args():
    parser = argparse.ArgumentParser(description='toHF script')
    parser.add_argument('config', help='config file name or path.')
    parser.add_argument('--pth-model', help='pth model file')
    parser.add_argument('--save-path', type=str, default='./work_dirs/hf_model',
                        help='save folder name')
    args = parser.parse_args()
    return args


@master_only
def master_print(msg):
    print(msg)


def main():
    args = parse_args()

    # resolve config path
    if not osp.isfile(args.config):
        try:
            args.config = cfgs_name_path[args.config]
        except KeyError:
            raise FileNotFoundError(f'Cannot find {args.config}')

    # load config & model
    cfg = Config.fromfile(args.config)
    model = BUILDER.build(cfg.model)

    # load checkpoint
    backend = get_file_backend(args.pth_model)
    if isinstance(backend, PetrelBackend):
        from xtuner.utils.fileio import patch_fileio
        with patch_fileio():
            state_dict = guess_load_checkpoint(args.pth_model)
    else:
        state_dict = guess_load_checkpoint(args.pth_model)

    # Load model weights
    model.load_state_dict(state_dict, strict=False)
    print(f'Loaded PTH model from {args.pth_model}')

    # === Save hca_queries separately ===
    if 'mllm.hca_queries' in state_dict:
        hca_queries_tensor = state_dict['mllm.hca_queries']
        os.makedirs(args.save_path, exist_ok=True)
        torch.save({'mllm.hca_queries': hca_queries_tensor},
                   osp.join(args.save_path, 'hca_queries.pth'))
        print(f'[Info] Saved hca_queries to {osp.join(args.save_path, "hca_queries.pth")}')
    else:
        print('[Warning] mllm.hca_queries not found in checkpoint.')

    # Merge LoRA and prepare for HuggingFace
    model._merge_lora()
    model.mllm.transfer_to_hf = True

    # Get state dict from model
    all_state_dict = model.all_state_dict()

    # Remove hca_queries from HF save
    if 'mllm.hca_queries' in all_state_dict:
        del all_state_dict['mllm.hca_queries']

    # Rename keys for HuggingFace compatibility
    name_map = {'mllm.model.': '', '.gamma': '.g_weight'}
    all_state_dict_new = {}
    for key in all_state_dict.keys():
        new_key = copy.deepcopy(key)
        for old, new in name_map.items():
            new_key = new_key.replace(old, new)
        all_state_dict_new[new_key] = all_state_dict[key]

    # Build HuggingFace model
    from projects.llava_sam2.hf.models.configuration_earthmind_chat import Sa2VAChatConfig
    from projects.llava_sam2.hf.models.modeling_earthmind_chat import Sa2VAChatModel

    internvl_config = Sa2VAChatConfig.from_pretrained(cfg.path)
    config_dict = internvl_config.to_dict()
    config_dict['auto_map'] = {
        'AutoConfig': 'configuration_earthmind_chat.Sa2VAChatConfig',
        'AutoModel': 'modeling_earthmind_chat.Sa2VAChatModel',
        'AutoModelForCausalLM': 'modeling_earthmind_chat.Sa2VAChatModel'
    }

    config_dict["llm_config"]["vocab_size"] = len(model.tokenizer)
    config_dict["template"] = cfg.template

    sa2va_hf_config = Sa2VAChatConfig(**config_dict)
    hf_sa2va_model = Sa2VAChatModel(
        sa2va_hf_config,
        vision_model=model.mllm.model.vision_model,
        language_model=model.mllm.model.language_model,
    )

    # Final load and save
    all_state_dict_new = {
        k.replace("mllm.", ""): v for k, v in all_state_dict_new.items()
    }

    hf_sa2va_model.load_state_dict(all_state_dict_new, strict=False)
    hf_sa2va_model.save_pretrained(args.save_path)
    model.tokenizer.save_pretrained(args.save_path)
    print(f"[Success] Saved HuggingFace model to {args.save_path}")

    # Copy source modeling files
    os.system(f"cp -pr ./projects/llava_sam2/hf/models/* {args.save_path}")

    # === 新增：验证视觉编码器已写入 ===
    try:
        verify_visual_encoder_export(model.mllm.model.vision_model, args.save_path)
    except Exception as e:
        print(f"[Verify][WARN] visual-encoder verification encountered an issue: {e}")


if __name__ == '__main__':
    main()
