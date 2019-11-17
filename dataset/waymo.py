import os
import pickle
from dataset import DatasetSplit, DatasetRegistry

__all__ = ["register_waymo"]


class WaymoDemo(DatasetSplit):
    def __init__(self, base_dir, split):
        
        assert split in ["train", "val"]
        split="waymo_" + split
        base_dir = os.path.expanduser(base_dir)
        self.imgdir = os.path.join(base_dir, split)
        self.meta_file = os.path.join(self.imgdir,"annotations.pickle")
        assert os.path.isdir(self.imgdir), self.imgdir

    def training_roidbs(self):
        ret = []
        with open(self.meta_file,'rb') as f:
            ret = pickle.load(f)

        return ret
    
    def inference_roidbs(self):
        ret = []
        with open(self.meta_file, 'rb') as f:
            a = pickle.load(f)
            ret = [{'file_name':os.path.join(self.imgdir,j['file_name']),'image_id':i} for i,j in enumerate(a)]
        return ret
    
    def eval_inference_results(self,results,output=None):
        return {}


def register_waymo(basedir):
    for split in ["train", "val"]:
        name = "waymo_" + split
        DatasetRegistry.register(name, lambda x=split: WaymoDemo(basedir, x))
        DatasetRegistry.register_metadata(name, "class_names", ["TYPE_BACKGROUND","TYPE_UNKNOWN", "TYPE_VEHICLE", "TYPE_PEDESTRIAN", "TYPE_SIGN", "TYPE_CYCLIST"])


if __name__ == '__main__':
    basedir = '~/data/balloon'
    roidbs = WaymoDemo(basedir, "train").training_roidbs()
    print("#images:", len(roidbs))

    from viz import draw_annotation
    from tensorpack.utils.viz import interactive_imshow as imshow
    import cv2
    for r in roidbs:
        im = cv2.imread(r["file_name"])
        vis = draw_annotation(im, r["boxes"], r["class"])
        imshow(vis)
