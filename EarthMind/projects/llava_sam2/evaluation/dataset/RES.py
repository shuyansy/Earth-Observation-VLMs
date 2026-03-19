import os
import numpy as np
import torch
from PIL import Image
from pycocotools import mask as _mask
from utils import REFER, Summary, AverageMeter, intersectionAndUnionGPU, master_only

DATASETS_ATTRIBUTES = {
    'refcoco': {'splitBy': "unc", 'dataset_name': 'refcoco'},
    'refcoco_plus': {'splitBy': "unc", 'dataset_name': 'refcoco+'},
    'refcocog': {'splitBy': "umd", 'dataset_name': 'refcocog'},
    'refsegrs': {'splitBy': "unc", 'dataset_name': 'refsegrs'},
    'rrsisd': {'splitBy': "unc", 'dataset_name': 'rrsisd'},
    'risbench': {'splitBy': "unc", 'dataset_name': 'risbench'},
    'sarnew': {'splitBy': "unc", 'dataset_name': 'sarnew'}
}




class RESDataset:
    METAINFO: dict = dict(name='Referring Expression Segmentation')

    def __init__(self,
                 image_folder,
                 dataset_name,
                 data_path=None,
                 split='val',
                 ):
        self.split = split
        self._set_attribute(dataset_name)
        json_datas = self.json_file_preprocess(data_path)
        self.json_datas = json_datas
        self.image_folder = image_folder

    def _set_attribute(self, dataset_name):
        attr_dict = DATASETS_ATTRIBUTES[dataset_name]
        self.splitBy = attr_dict['splitBy']
        self.dataset_name = attr_dict['dataset_name']

    def __len__(self):
        return len(self.json_datas)

    def real_len(self):
        return len(self.json_datas)

    def json_file_preprocess(self, data_path):
        splitBy = self.splitBy
        dataset_name = self.dataset_name
        refer_api = REFER(data_path, dataset_name, splitBy)
        ref_ids_train = refer_api.getRefIds(split=self.split)
        images_ids_train = refer_api.getImgIds(ref_ids=ref_ids_train)
        refs_train = refer_api.loadRefs(ref_ids=ref_ids_train)
        self.img2refs = self.create_img_to_refs_mapping(refs_train)

        image_infos = []
        loaded_images = refer_api.loadImgs(image_ids=images_ids_train)
        for item in loaded_images:
            item = item.copy()
            image_infos.append(item)

        self.annotations = refer_api.Anns
        refs = [self.img2refs[image_info['id']] for image_info in image_infos]

        ret = []
        for image_info, ref in zip(image_infos, refs):
            if len(ref) == 0:
                continue

            sents = []
            ann_ids = []
            for _ref in ref:
                for sent in _ref["sentences"]:
                    text = sent["sent"]
                    sents.append(text)
                    ann_ids.append(_ref["ann_id"])

            sampled_inds = list(range(len(sents)))
            sampled_sents = np.vectorize(sents.__getitem__)(sampled_inds).tolist()
            sampled_ann_ids = [ann_ids[ind] for ind in sampled_inds]
            selected_labels = sampled_sents
            ret.append(
                {'image_info': image_info,
                 'sampled_ann_id': sampled_ann_ids,
                 'selected_labels': selected_labels,
                 'image': image_info['file_name']
                 }
            )
        return ret

    def create_img_to_refs_mapping(self, refs_train):
        img2refs = {}
        for ref in refs_train:
            img2refs[ref["image_id"]] = img2refs.get(ref["image_id"], []) + [ref, ]
        return img2refs

    def decode_mask(self, annotations_ids, image_info):
        flag = False
        masks = []

        for ann_id in annotations_ids:
            if isinstance(ann_id, list):
                flag = True
                if -1 in ann_id:
                    assert len(ann_id) == 1
                    m = np.zeros((image_info["height"], image_info["width"])).astype(
                        np.uint8
                    )
                else:
                    m_final = np.zeros(
                        (image_info["height"], image_info["width"])
                    ).astype(np.uint8)
                    for ann_id_i in ann_id:
                        ann = self.annotations[ann_id_i]

                        if len(ann["segmentation"]) == 0:
                            m = np.zeros(
                                (image_info["height"], image_info["width"])
                            ).astype(np.uint8)
                        else:
                            if type(ann["segmentation"][0]) == list:  # polygon
                                rle = _mask.frPyObjects(
                                    ann["segmentation"], image_info["height"], image_info["width"], )
                            else:
                                rle = ann["segmentation"]
                                for i in range(len(rle)):
                                    if not isinstance(rle[i]["counts"], bytes):
                                        rle[i]["counts"] = rle[i]["counts"].encode()
                            m = _mask.decode(rle)
                            m = np.sum(
                                m, axis=2
                            )  # sometimes there are multiple binary map (corresponding to multiple segs)
                            m = m.astype(np.uint8)  # convert to np.uint8
                        m_final = m_final | m
                    m = m_final
                masks.append(m)
                continue

            ann = self.annotations[ann_id]

            if len(ann["segmentation"]) == 0:
                m = np.zeros((image_info["height"], image_info["width"])).astype(
                    np.uint8
                )
                masks.append(m)
                continue

            if type(ann["segmentation"][0]) == list:  # polygon
                rle = _mask.frPyObjects(
                    ann["segmentation"], image_info["height"], image_info["width"]
                )
            else:
                rle = ann["segmentation"]
                for i in range(len(rle)):
                    if not isinstance(rle[i]["counts"], bytes):
                        rle[i]["counts"] = rle[i]["counts"].encode()
            m = _mask.decode(rle)
            m = np.sum(m, axis=2)  # sometimes there are multiple binary map (corresponding to multiple segs)
            m = m.astype(np.uint8)  # convert to np.uint8
            masks.append(m)
        masks = np.stack(masks, axis=0)

        # if self.pad_image_to_square:
        masks = torch.from_numpy(masks)
        return masks

    def only_get_text_infos(self, json_data):
        return {'sampled_sents': json_data['selected_labels']}

    def get_questions(self, text_require_infos):
        sampled_sents = text_require_infos['sampled_sents']
        ret = []
        for sent in sampled_sents:
            ret.append("<image>\n Please segment {} in this image.".format(sent))
        return ret

    def filter_data_dict(self, data_dict):
        names = ['image', 'text', 'gt_masks', 'img_id']
        ret = {name: data_dict[name] for name in names}
        return ret

    def __getitem__(self, index):
        index = index % self.real_len()
        data_dict = self.json_datas[index]
        text_require_infos = self.only_get_text_infos(data_dict)
        questions = self.get_questions(text_require_infos)

        assert data_dict.get('image', None) is not None
        if data_dict.get('image', None) is not None:
            image_file = data_dict['image']
            image_file = os.path.join(self.image_folder, image_file)
            image = Image.open(image_file).convert('RGB')

            # process and get masks for evaluation
            masks = self.decode_mask(data_dict['sampled_ann_id'], data_dict['image_info'])
            data_dict['gt_masks'] = masks
            data_dict['image'] = image
            data_dict['text'] = questions
            data_dict['img_id'] = str(index)
        return self.filter_data_dict(data_dict)
    

    @master_only
    def evaluate(self, result, work_dir):
        trackers = {
            "intersection": AverageMeter("Intersec", ":6.3f", Summary.SUM),
            "union": AverageMeter("Union", ":6.3f", Summary.SUM),
            "gIoU": AverageMeter("gIoU", ":6.3f", Summary.SUM)
        }
        all_class_ious = []

        for pred_dict in result:
            accuracy_iou = 0.0
            masks = pred_dict['prediction_masks']
            _masks = []
            for mask in masks:
                if mask is not None:
                    mask = rle_to_mask(mask)
                _masks.append(mask)

            targets = pred_dict['gt_masks']
            _targets = rle_to_mask(targets)

            for i_item, _mask in enumerate(_masks):
                if _mask is None:
                    continue

                _target = _targets[i_item: i_item + 1]
                for prediction, target in zip(_mask, _target):
                    prediction = torch.from_numpy(prediction).int().cuda()
                    target = torch.from_numpy(target).int().cuda()

                    intersect, union_, _ = intersectionAndUnionGPU(
                        prediction.contiguous().clone(), target.contiguous(), 2, ignore_index=255
                    )

                    # 全局 intersection 和 union（用于 oIoU 计算）
                    trackers["intersection"].update(intersect.sum().item())
                    trackers["union"].update(union_.sum().item())

                    # 逐样本 IoU，用于 gIoU 和 mIoU 计算
                    accuracy_iou += intersect / (union_ + 1e-5)
                    accuracy_iou[union_ == 0] += 1.0

                    # 记录当前帧的每类 IoU，用于 mIoU
                    iou_per_class = (intersect.float() / (union_.float() + 1e-10)).cpu().numpy()
                    all_class_ious.append(iou_per_class)

            accuracy_iou = accuracy_iou.cpu().numpy() / _targets.shape[0]
            trackers["gIoU"].update(accuracy_iou, n=_targets.shape[0])

        # 获取总 intersection 与 union（都是 scalar）
        total_intersection = trackers["intersection"].sum
        total_union = trackers["union"].sum

        # 全局 IoU（oIoU）与平均 IoU（mIoU）
        oIoU = total_intersection / (total_union + 1e-10)
        mIoU = np.mean(np.array(all_class_ious), axis=0).mean()  # 所有类别 IoU 的平均

        # 用 gIoU 跟踪器记录的是逐样本平均的 IoU
        global_iou = trackers["gIoU"].avg

        # 逐类的总体 IoU（CIoU）
        class_iou = total_intersection / (total_union + 1e-10)

        print('============================================', 'current')
        print('CIoU: {}, GIoU: {}'.format(class_iou, global_iou), 'current')
        print('mIoU: {:.4f}, oIoU: {:.4f}'.format(mIoU, oIoU), 'current')
        print('============================================', 'current')
        print('RES_{}_{} successfully finished evaluating'.format(self.dataset_name, self.split), 'current')

        return {'Acc': class_iou, 'mIoU': mIoU, 'oIoU': oIoU}






class RESDataset_Multi:
    METAINFO: dict = dict(name='Referring Expression Segmentation')

    def __init__(self,
                 image_folder,
                 rgb_image_folder,
                 dataset_name,
                 data_path=None,
                 split='val',
                 ):
        self.split = split
        self._set_attribute(dataset_name)
        json_datas = self.json_file_preprocess(data_path)
        self.json_datas = json_datas
        self.image_folder = image_folder
        self.rgb_image_folder=rgb_image_folder

    def _set_attribute(self, dataset_name):
        attr_dict = DATASETS_ATTRIBUTES[dataset_name]
        self.splitBy = attr_dict['splitBy']
        self.dataset_name = attr_dict['dataset_name']

    def __len__(self):
        return len(self.json_datas)

    def real_len(self):
        return len(self.json_datas)

    def json_file_preprocess(self, data_path):
        splitBy = self.splitBy
        dataset_name = self.dataset_name
        refer_api = REFER(data_path, dataset_name, splitBy)
        ref_ids_train = refer_api.getRefIds(split=self.split)
        images_ids_train = refer_api.getImgIds(ref_ids=ref_ids_train)
        refs_train = refer_api.loadRefs(ref_ids=ref_ids_train)
        self.img2refs = self.create_img_to_refs_mapping(refs_train)

        image_infos = []
        loaded_images = refer_api.loadImgs(image_ids=images_ids_train)
        for item in loaded_images:
            item = item.copy()
            image_infos.append(item)

        self.annotations = refer_api.Anns
        refs = [self.img2refs[image_info['id']] for image_info in image_infos]

        ret = []
        for image_info, ref in zip(image_infos, refs):
            if len(ref) == 0:
                continue

            sents = []
            ann_ids = []
            for _ref in ref:
                for sent in _ref["sentences"]:
                    text = sent["sent"]
                    sents.append(text)
                    ann_ids.append(_ref["ann_id"])

            sampled_inds = list(range(len(sents)))
            sampled_sents = np.vectorize(sents.__getitem__)(sampled_inds).tolist()
            sampled_ann_ids = [ann_ids[ind] for ind in sampled_inds]
            selected_labels = sampled_sents
            ret.append(
                {'image_info': image_info,
                 'sampled_ann_id': sampled_ann_ids,
                 'selected_labels': selected_labels,
                 'image': image_info['file_name']
                 }
            )
        return ret

    def create_img_to_refs_mapping(self, refs_train):
        img2refs = {}
        for ref in refs_train:
            img2refs[ref["image_id"]] = img2refs.get(ref["image_id"], []) + [ref, ]
        return img2refs

    def decode_mask(self, annotations_ids, image_info):
        flag = False
        masks = []

        for ann_id in annotations_ids:
            if isinstance(ann_id, list):
                flag = True
                if -1 in ann_id:
                    assert len(ann_id) == 1
                    m = np.zeros((image_info["height"], image_info["width"])).astype(
                        np.uint8
                    )
                else:
                    m_final = np.zeros(
                        (image_info["height"], image_info["width"])
                    ).astype(np.uint8)
                    for ann_id_i in ann_id:
                        ann = self.annotations[ann_id_i]

                        if len(ann["segmentation"]) == 0:
                            m = np.zeros(
                                (image_info["height"], image_info["width"])
                            ).astype(np.uint8)
                        else:
                            if type(ann["segmentation"][0]) == list:  # polygon
                                rle = _mask.frPyObjects(
                                    ann["segmentation"], image_info["height"], image_info["width"], )
                            else:
                                rle = ann["segmentation"]
                                for i in range(len(rle)):
                                    if not isinstance(rle[i]["counts"], bytes):
                                        rle[i]["counts"] = rle[i]["counts"].encode()
                            m = _mask.decode(rle)
                            m = np.sum(
                                m, axis=2
                            )  # sometimes there are multiple binary map (corresponding to multiple segs)
                            m = m.astype(np.uint8)  # convert to np.uint8
                        m_final = m_final | m
                    m = m_final
                masks.append(m)
                continue

            ann = self.annotations[ann_id]

            if len(ann["segmentation"]) == 0:
                m = np.zeros((image_info["height"], image_info["width"])).astype(
                    np.uint8
                )
                masks.append(m)
                continue

            if type(ann["segmentation"][0]) == list:  # polygon
                rle = _mask.frPyObjects(
                    ann["segmentation"], image_info["height"], image_info["width"]
                )
            else:
                rle = ann["segmentation"]
                for i in range(len(rle)):
                    if not isinstance(rle[i]["counts"], bytes):
                        rle[i]["counts"] = rle[i]["counts"].encode()
            m = _mask.decode(rle)
            m = np.sum(m, axis=2)  # sometimes there are multiple binary map (corresponding to multiple segs)
            m = m.astype(np.uint8)  # convert to np.uint8
            masks.append(m)
        masks = np.stack(masks, axis=0)

        # if self.pad_image_to_square:
        masks = torch.from_numpy(masks)
        return masks

    def only_get_text_infos(self, json_data):
        return {'sampled_sents': json_data['selected_labels']}

    def get_questions(self, text_require_infos):
        sampled_sents = text_require_infos['sampled_sents']
        ret = []
        for sent in sampled_sents:
            ret.append("<image>\n Please segment {} in this image.".format(sent))
        return ret

    def filter_data_dict(self, data_dict):
        names = ['image', 'rgb_image', 'text', 'gt_masks', 'img_id']
        ret = {name: data_dict[name] for name in names}
        return ret

    def __getitem__(self, index):
        index = index % self.real_len()
        data_dict = self.json_datas[index]
        text_require_infos = self.only_get_text_infos(data_dict)
        questions = self.get_questions(text_require_infos)

        assert data_dict.get('image', None) is not None
        if data_dict.get('image', None) is not None:
            image_file = data_dict['image']
            image_file = os.path.join(self.image_folder, image_file)
            basename=image_file.split("/")[-1]
            rgb_image_file=  os.path.join(self.rgb_image_folder,  basename)
            image = Image.open(image_file).convert('RGB')
            rgb_image = Image.open(rgb_image_file).convert('RGB')

            # process and get masks for evaluation
            masks = self.decode_mask(data_dict['sampled_ann_id'], data_dict['image_info'])
            data_dict['gt_masks'] = masks
            data_dict['image'] = image
            data_dict["rgb_image"] = rgb_image
            data_dict['text'] = questions
            data_dict['img_id'] = str(index)
        return self.filter_data_dict(data_dict)
    

    @master_only
    def evaluate(self, result, work_dir):
        trackers = {
            "intersection": AverageMeter("Intersec", ":6.3f", Summary.SUM),
            "union": AverageMeter("Union", ":6.3f", Summary.SUM),
            "gIoU": AverageMeter("gIoU", ":6.3f", Summary.SUM)
        }
        all_class_ious = []

        for pred_dict in result:
            accuracy_iou = 0.0
            masks = pred_dict['prediction_masks']
            _masks = []
            for mask in masks:
                if mask is not None:
                    mask = rle_to_mask(mask)
                _masks.append(mask)

            targets = pred_dict['gt_masks']
            _targets = rle_to_mask(targets)

            for i_item, _mask in enumerate(_masks):
                if _mask is None:
                    continue

                _target = _targets[i_item: i_item + 1]
                for prediction, target in zip(_mask, _target):
                    prediction = torch.from_numpy(prediction).int().cuda()
                    target = torch.from_numpy(target).int().cuda()

                    intersect, union_, _ = intersectionAndUnionGPU(
                        prediction.contiguous().clone(), target.contiguous(), 2, ignore_index=255
                    )

                    # 全局 intersection 和 union（用于 oIoU 计算）
                    trackers["intersection"].update(intersect.sum().item())
                    trackers["union"].update(union_.sum().item())

                    # 逐样本 IoU，用于 gIoU 和 mIoU 计算
                    accuracy_iou += intersect / (union_ + 1e-5)
                    accuracy_iou[union_ == 0] += 1.0

                    # 记录当前帧的每类 IoU，用于 mIoU
                    iou_per_class = (intersect.float() / (union_.float() + 1e-10)).cpu().numpy()
                    all_class_ious.append(iou_per_class)

            accuracy_iou = accuracy_iou.cpu().numpy() / _targets.shape[0]
            trackers["gIoU"].update(accuracy_iou, n=_targets.shape[0])

        # 获取总 intersection 与 union（都是 scalar）
        total_intersection = trackers["intersection"].sum
        total_union = trackers["union"].sum

        # 全局 IoU（oIoU）与平均 IoU（mIoU）
        oIoU = total_intersection / (total_union + 1e-10)
        mIoU = np.mean(np.array(all_class_ious), axis=0).mean()  # 所有类别 IoU 的平均

        # 用 gIoU 跟踪器记录的是逐样本平均的 IoU
        global_iou = trackers["gIoU"].avg

        # 逐类的总体 IoU（CIoU）
        class_iou = total_intersection / (total_union + 1e-10)

        print('============================================', 'current')
        print('CIoU: {}, GIoU: {}'.format(class_iou, global_iou), 'current')
        print('mIoU: {:.4f}, oIoU: {:.4f}'.format(mIoU, oIoU), 'current')
        print('============================================', 'current')
        print('RES_{}_{} successfully finished evaluating'.format(self.dataset_name, self.split), 'current')

        return {'Acc': class_iou, 'mIoU': mIoU, 'oIoU': oIoU}





    # @master_only
    # def evaluate(self, result, work_dir):
    #     trackers = {
    #         "intersection": AverageMeter("Intersec", ":6.3f", Summary.SUM),
    #         "union": AverageMeter("Union", ":6.3f", Summary.SUM),
    #         "gIoU": AverageMeter("gIoU", ":6.3f", Summary.SUM)
    #     }
    #     for pred_dict in result:
    #         intersection, union, accuracy_iou = 0.0, 0.0, 0.0
    #         masks = pred_dict['prediction_masks']
    #         _masks = []
    #         for mask in masks:
    #             if mask is not None:
    #                 mask = rle_to_mask(mask)
    #             _masks.append(mask)
    #         targets = pred_dict['gt_masks']
    #         _targets = rle_to_mask(targets)

    #         for i_item, _mask in enumerate(_masks):
    #             if _mask is None:
    #                 continue

    #             _target = _targets[i_item: i_item+1]
    #             for prediction, target in zip(_mask, _target):
    #                 prediction = torch.from_numpy(prediction).int().cuda()
    #                 target = torch.from_numpy(target).int().cuda()
    #                 intersect, union_, _ = intersectionAndUnionGPU(
    #                     prediction.contiguous().clone(), target.contiguous(), 2, ignore_index=255
    #                 )
    #                 intersection += intersect
    #                 union += union_
    #                 accuracy_iou += intersect / (union_ + 1e-5)
    #                 accuracy_iou[union_ == 0] += 1.0

    #         intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
    #         accuracy_iou = accuracy_iou.cpu().numpy() / _targets.shape[0]
    #         trackers["intersection"].update(intersection)
    #         trackers["union"].update(union)
    #         trackers["gIoU"].update(accuracy_iou, n=_targets.shape[0])

    #     cur_results = {'pixel_intersection': trackers["intersection"].sum[1],
    #                    'pixel_union': trackers["union"].sum[1],
    #                    'gIoU': trackers["gIoU"].avg[1],
    #                    'mask_counts': trackers["gIoU"].count,
    #                    }
    #     class_iou = cur_results['pixel_intersection'] / (cur_results['pixel_union'] + 1e-10)
    #     global_iou = cur_results['gIoU']

    #     print('============================================', 'current')
    #     print('CIoU: {}, GIoU: {}'.format(class_iou, global_iou), 'current')
    #     print('============================================', 'current')
    #     print('RES_{}_{} successfully finished evaluating'.format(self.dataset_name, self.split),
    #               'current')
    #     return {'Acc': class_iou}


def rle_to_mask(rle):
    mask = []
    for r in rle:
        m = _mask.decode(r)
        m = np.uint8(m)
        mask.append(m)
    mask = np.stack(mask, axis=0)
    return mask


