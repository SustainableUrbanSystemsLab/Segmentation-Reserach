import math
import numpy as np


def calculate_rectangle_size_from_batch_size(batch_size):
    """
    calculate number of rows and cols to composite a rectangle given a batch size
    :param batch_size:
    :return: number of cols and number of rows
    """
    rectangle_height = int(math.sqrt(batch_size) + 0.5)
    rectangle_width = int(batch_size / rectangle_height)

    if rectangle_height * rectangle_width > batch_size:
        if rectangle_height >= rectangle_width:
            rectangle_height = rectangle_height - 1
        else:
            rectangle_width = rectangle_width - 1

    if (rectangle_height + 1) * rectangle_width <= batch_size:
        rectangle_height = rectangle_height + 1
    if (rectangle_width + 1) * rectangle_height <= batch_size:
        rectangle_width = rectangle_width + 1

    # swap col and row to make a horizontal rect
    if rectangle_height > rectangle_width:
        rectangle_height, rectangle_width = rectangle_width, rectangle_height

    if rectangle_height * rectangle_width != batch_size:
        return batch_size, 1

    return rectangle_height, rectangle_width


def convert_bounding_boxes_to_coord_list(bounding_boxes):
    """
    Convert bounding box numpy array to python list of point arrays.
    The points will represent the corners of a polygon.
    Parameters
    bounding_boxes: numpy array of shape [n, 4]
    return: python array of point numpy arrays, each point array is in shape [4,2]
            representing coordinates (y,x) of the polygon points starting from top-left corner
    """
    num_bounding_boxes = bounding_boxes.shape[0]
    bounding_box_coord_list = []
    for i in range(num_bounding_boxes):
        coord_array = np.empty(shape=(4, 2), dtype=float)
        coord_array[0][0] = bounding_boxes[i][0]
        coord_array[0][1] = bounding_boxes[i][1]

        coord_array[1][0] = bounding_boxes[i][0]
        coord_array[1][1] = bounding_boxes[i][3]

        coord_array[2][0] = bounding_boxes[i][2]
        coord_array[2][1] = bounding_boxes[i][3]

        coord_array[3][0] = bounding_boxes[i][2]
        coord_array[3][1] = bounding_boxes[i][1]

        bounding_box_coord_list.append(coord_array)

    return bounding_box_coord_list


def get_tile_size(model_height, model_width, padding, batch_height, batch_width):
    """
    Calculate request tile size given model and batch dimensions
    :param model_height:
    :param model_width:
    :param padding:
    :param batch_width:
    :param batch_height:
    :return: tile height and tile width
    """
    tile_height = (model_height - 2 * padding) * batch_height
    tile_width = (model_width - 2 * padding) * batch_width

    return tile_height, tile_width
    

# def scale_batch(
#     image_batch, model_info, normalization_stats=None, break_extract_bands=False
# ):
#     if normalization_stats is None:
#         normalization_stats = model_info.get("NormalizationStats", None)
#     if break_extract_bands:
#         # Only for change detection
#         # if subset of extract bands are specified fix this.
#         n_bands = len(model_info["ExtractBands"]) // 2
#         band_min_values = np.array(normalization_stats["band_min_values"])[
#             model_info["ExtractBands"][:n_bands]
#         ].reshape(1, -1, 1, 1)
#         band_max_values = np.array(normalization_stats["band_max_values"])[
#             model_info["ExtractBands"][:n_bands]
#         ].reshape(1, -1, 1, 1)
#     else:
#         if normalization_stats is None:
#             modtype = model_info.get("InferenceFunction", None)
#             if modtype == "ArcGISImageTranslation.py":
#                 band_min_values = np.full((image_batch.shape[1],), 0).reshape(
#                     1, -1, 1, 1
#                 )
#                 band_max_values = np.full((image_batch.shape[1],), 255).reshape(
#                     1, -1, 1, 1
#                 )
#         else:
#             band_min_values = np.array(normalization_stats["band_min_values"])[
#                 model_info["ExtractBands"]
#             ].reshape(1, -1, 1, 1)
#             band_max_values = np.array(normalization_stats["band_max_values"])[
#                 model_info["ExtractBands"]
#             ].reshape(1, -1, 1, 1)
#     img_scaled = (image_batch - band_min_values) / (band_max_values - band_min_values)
#     return img_scaled


# def normalize_batch(image_batch, model_info=None, normalization_stats=None):
#     if normalization_stats is None:
#         normalization_stats = model_info.get("NormalizationStats", None)
#     scaled_mean_values = np.array(normalization_stats["scaled_mean_values"])[
#         model_info["ExtractBands"]
#     ].reshape(1, -1, 1, 1)
#     scaled_std_values = np.array(normalization_stats["scaled_std_values"])[
#         model_info["ExtractBands"]
#     ].reshape(1, -1, 1, 1)
#     img_scaled = scale_batch(image_batch, model_info)
#     img_normed = (img_scaled - scaled_mean_values) / scaled_std_values
#     return img_normed
    

def tile_to_batch(
    pixel_block, model_height, model_width, padding, fixed_tile_size=True, **kwargs
):
    inner_width = model_width - 2 * padding
    inner_height = model_height - 2 * padding

    band_count, pb_height, pb_width = pixel_block.shape
    pixel_type = pixel_block.dtype

    if fixed_tile_size is True:
        batch_height = kwargs["batch_height"]
        batch_width = kwargs["batch_width"]
    else:
        batch_height = math.ceil((pb_height - 2 * padding) / inner_height)
        batch_width = math.ceil((pb_width - 2 * padding) / inner_width)

    batch = np.zeros(
        shape=(batch_width * batch_height, band_count, model_height, model_width),
        dtype=pixel_type,
    )
    for b in range(batch_width * batch_height):
        y = int(b / batch_width)
        x = int(b % batch_width)

        # pixel block might not be the shape (band_count, model_height, model_width)
        sub_pixel_block = pixel_block[
            :,
            y * inner_height : y * inner_height + model_height,
            x * inner_width : x * inner_width + model_width,
        ]
        sub_pixel_block_shape = sub_pixel_block.shape
        batch[
            b, :, : sub_pixel_block_shape[1], : sub_pixel_block_shape[2]
        ] = sub_pixel_block

    return batch, batch_height, batch_width


def variable_tile_size_check(json_info, parameters):
    if json_info.get("SupportsVariableTileSize", False):
        parameters.extend(
            [
                {
                    "name": "tile_size",
                    "dataType": "numeric",
                    "value": int(json_info["ImageHeight"]),
                    "required": False,
                    "displayName": "Tile Size",
                    "description": "Tile size used for inferencing",
                }
            ]
        )
    return parameters