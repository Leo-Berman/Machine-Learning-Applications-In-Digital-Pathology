import nedc_mladp_ann_tools as local_ann
import nedc_dpath_ann_tools

def generate_features(master:dict,
                      frame_dimensions:tuple,
                      window_dimensions:tuple,
                      image_path:str,
                      annotation_path:str) -> None:

    annotation_tool = nedc_dpath_ann_tools.AnnDpath()
    annotation_tool.load(image_path)
    
    master_dictionary[images_processed] = {
        'header':annotation_tool.get_header(),
        'frame_size':frame_size,
        'window_size':window_size,
    }
    
    
    
    local_ann.generateFeatures(annotation_tool.get_graph(),
                               master_dictionary[images_processed]['header'],
                               frame_size, window_size)
    
