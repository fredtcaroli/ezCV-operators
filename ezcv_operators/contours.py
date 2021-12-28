import cv2
from ezcv.operator import register_operator, Operator, EnumParameter, settings, BooleanParameter
from ezcv.pipeline import PipelineContext
from ezcv.typing import Image


@settings.GRAY_ONLY(True)
@register_operator
class FindContours(Operator):
    visual_feedback = BooleanParameter(False)
    mode = EnumParameter(
        possible_values=[
            'RETR_TREE',
            'RETR_LIST',
            'RETR_EXTERNAL',
            'RETR_CCOMP',
            # 'RETR_FLOODFILL'  # TODO: Flood fill apparently is not accepting CV_8U images. Disabling it for now
        ],
        default_value='RETR_TREE'
    )
    method = EnumParameter(
        possible_values=[
            'CHAIN_APPROX_SIMPLE',
            'CHAIN_APPROX_TC89_KCOS',
            'CHAIN_APPROX_TC89_L1',
            'CHAIN_APPROX_NONE'
        ],
        default_value='CHAIN_APPROX_SIMPLE'
    )

    def run(self, img: Image, ctx: PipelineContext) -> Image:
        mode = getattr(cv2, self.mode)
        method = getattr(cv2, self.method)
        im2, contours, hierarchy = cv2.findContours(img, mode=mode, method=method)
        ctx.add_info('contours', contours)
        ctx.add_info('hierarchy', hierarchy)
        if self.visual_feedback:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
        return img
