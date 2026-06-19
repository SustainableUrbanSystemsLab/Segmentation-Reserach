import json 
import math
import numpy as np
import os
import torch
from torchvision.models.detection.retinanet import RetinaNet, retinanet_resnet50_fpn
from torchvision.ops import nms


from tree_detector_inference_utils import \
    calculate_rectangle_size_from_batch_size,\
    convert_bounding_boxes_to_coord_list, \
    get_tile_size, \
    tile_to_batch, \
    variable_tile_size_check

class ChildObjectDetector:
    def initialize(self, model, model_as_file):
        if model_as_file:
            with open(model, "r") as f:
                self.json_info = json.load(f)
        else:
            self.json_info = json.loads(model)

        self.model_path = os.path.abspath(self.json_info["ModelFile"])
        #
        backbone = retinanet_resnet50_fpn(pretrained=False).backbone
        self.model = RetinaNet(backbone=backbone, num_classes=len(self.json_info['Classes']))
        self.model.load_state_dict(torch.load(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'NEON.pt')
        ))
        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.model = self.model.cuda()
        self.model.eval()

    def getParameterInfo(self, required_parameters):
        required_parameters.extend(
            [   
                {
                    "name": "padding",
                    "dataType": "numeric",
                    "value": self.json_info["ImageHeight"] // 4,
                    "required": False,
                    "displayName": "Padding",
                    "description": "Padding",
                },
                {
                    "name": "threshold",
                    "dataType": "numeric",
                    "value": 0.1,
                    "required": False,
                    "displayName": "Confidence Score Threshold [0.0, 1.0]",
                    "description": "Confidence score threshold value [0.0, 1.0]",
                },
                {
                    "name": "nms_overlap",
                    "dataType": "numeric",
                    "value": 0.1,
                    "required": False,
                    "displayName": "NMS Overlap",
                    "description": "Maximum allowed overlap within each chip",
                },
                {
                    "name": "batch_size",
                    "dataType": "numeric",
                    "required": False,
                    "value": 4,
                    "displayName": "Batch Size",
                    "description": "Batch Size",
                },
                {
                    "name": "exclude_pad_detections",
                    "dataType": "string",
                    "required": False,
                    "domain": ("True", "False"),
                    "value": "True",
                    "displayName": "Filter Outer Padding Detections",
                    "description": "Filter detections which are outside the specified padding",
                }
            ]
        )
        
        required_parameters = variable_tile_size_check(
            self.json_info, required_parameters
        )
        
        return required_parameters

    def getConfiguration(self, **scalars):
        self.padding = int(
            scalars.get("padding", self.json_info["ImageHeight"] // 4)
        )  ## Default padding Imageheight//4.
        self.nms_overlap = float(
            scalars.get("nms_overlap", 0.1)
        )  ## Default 0.1 NMS Overlap.
        self.batch_size = (
            int(scalars.get("batch_size", 1))
        )
        self.thres = float(scalars.get("threshold", 0.5))  ## Default 0.5 threshold.
        self.filter_outer_padding_detections = scalars.get(
            "exclude_pad_detections", "True"
        ).lower() in [
            "true",
            "1",
            "t",
            "y",
            "yes",
        ]  ## Default value True

        self.tytx = int(scalars.get("tile_size", self.json_info["ImageHeight"]))
        (
            self.rectangle_height,
            self.rectangle_width,
        ) = calculate_rectangle_size_from_batch_size(self.batch_size)

        ty, tx = get_tile_size(
            self.tytx,
            self.tytx,
            self.padding,
            self.rectangle_height,
            self.rectangle_width,
        )

        return {
            "extractBands": tuple(self.json_info["ExtractBands"]),
            "padding": self.padding,
            "threshold": self.thres,
            "nms_overlap": self.nms_overlap,
            "tx": tx,
            "ty": ty,
            "fixedTileSize": 1,
        }


    def vectorize(self, **pixelBlocks):
        input_image = pixelBlocks["raster_pixels"].astype(np.float32)
        #
        batch, batch_height, batch_width = tile_to_batch(
            input_image,
            self.tytx,
            self.tytx,
            self.padding,
            fixed_tile_size=True,
            batch_height=self.rectangle_height,
            batch_width=self.rectangle_width,
        )
        with torch.no_grad():
            prediction = self.model(torch.tensor(batch/255.0, device=self.device))
            # boxes from this model are in format: "xmin", "ymin", "xmax", "ymax"
            # Box coordinates are in range (1, tile_size)

        side = math.sqrt(self.batch_size)
        bounding_boxes = []
        scores = []
        classes = []
        for batch_idx in range(self.batch_size):
            i, j = batch_idx // side, batch_idx % side
            #
            boxes = prediction[batch_idx]['boxes'].cpu().numpy()
            _scores = prediction[batch_idx]['scores'].cpu().numpy()
            labels = prediction[batch_idx]['labels'].cpu().numpy()
            if self.filter_outer_padding_detections:
                # Exclude pad Detections
                mask = ( self.padding <= boxes[:, [0, 2]].mean(-1) ) & ( boxes[:, [0, 2]].mean(-1) <= self.tytx-self.padding ) & ( self.padding <= boxes[:, [1, 3]].mean(-1) ) & ( boxes[:, [1, 3]].mean(-1) <= self.tytx-self.padding )
                boxes = boxes[mask]
                if len(boxes) == 0:
                    continue
                _scores = _scores[mask]
                labels = labels[mask]
            #
            boxes[:, [0, 2]]+=(j*self.tytx)
            boxes[:, [1, 3]]+=(i*self.tytx)
            bounding_boxes.append(boxes)
            scores.append(_scores)
            classes.append(labels)

        if bounding_boxes:
            scores = np.concatenate(scores).astype(np.float32)
            mask = scores >= self.thres
            scores = scores[mask]
            bounding_boxes = np.concatenate(bounding_boxes)[mask]
            classes = np.concatenate(classes).astype(np.uint8)[mask]

            # Apply NMS
            keep_idxs = nms(torch.tensor(bounding_boxes), torch.tensor(scores), self.nms_overlap).numpy()
            bounding_boxes = bounding_boxes[keep_idxs]
            scores = scores[keep_idxs] * 100
            classes = classes[keep_idxs] + 1 #  indexed to 1 indexed

            # Switch to y1, x1, y2, x2 format as used in other raster functions
            bounding_boxes = bounding_boxes[:, [1, 0, 3, 2]]
            
            
            return (
                convert_bounding_boxes_to_coord_list(bounding_boxes),
                scores.tolist(),
                classes
            )
        return ([], [], [])
