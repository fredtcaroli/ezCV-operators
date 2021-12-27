from typing import Type, Dict, Any

from ezcv.typing import Image
import hypothesis
import hypothesis.strategies as st
from ezcv.operator import Operator, IntegerParameter, DoubleParameter, EnumParameter, BooleanParameter
from ezcv.pipeline import PipelineContext
from ezcv.test_utils import parametrize_img
from ezcv.utils import is_image

from ezcv_operators.blur import GaussianBlur
from ezcv_operators.clahe import CLAHE
from ezcv_operators.color_space import ColorSpaceChange
from ezcv_operators.threshold import SimpleThreshold, AdaptiveThreshold


def make_hypothesis_parameters(op_class: Type[Operator]) -> Dict[str, Any]:
    params_specs = op_class.get_parameters_specs()
    r = dict()
    for param_name, param_spec in params_specs.items():
        if isinstance(param_spec, IntegerParameter):
            r[param_name] = st.integers(min_value=param_spec.lower, max_value=param_spec.upper)
        elif isinstance(param_spec, DoubleParameter):
            r[param_name] = st.floats(min_value=param_spec.lower, max_value=param_spec.upper)
        elif isinstance(param_spec, EnumParameter):
            r[param_name] = st.sampled_from(param_spec.possible_values)
        elif isinstance(param_spec, BooleanParameter):
            r[param_name] = st.booleans()
    return r


def run_operator(operator: Operator, img: Image) -> Image:
    ctx = PipelineContext(original_img=img)
    with ctx.scope('test'):
        r = operator.run(img, ctx)
    return r


@hypothesis.given(**make_hypothesis_parameters(GaussianBlur))
@parametrize_img
def test_blur(kernel_size: int, sigma: float, img: Image):
    op = GaussianBlur()
    op.kernel_size = kernel_size
    op.sigma = sigma
    r = run_operator(op, img)
    assert is_image(r)


@hypothesis.given(**make_hypothesis_parameters(CLAHE))
@parametrize_img
def test_clahe(clip_limit: float, tile_grid_size: int, img: Image):
    op = CLAHE()
    op.clip_limit = clip_limit
    op.tile_grid_size = tile_grid_size
    r = run_operator(op, img)
    assert is_image(r)


@hypothesis.given(**make_hypothesis_parameters(ColorSpaceChange))
@parametrize_img
def test_color_space(src: str, target: str, img: Image):
    hypothesis.assume((src == 'GRAY') == (img.ndim == 2))
    op = ColorSpaceChange()
    op.src = src
    op.target = target
    r = run_operator(op, img)
    assert is_image(r)


@hypothesis.given(**make_hypothesis_parameters(SimpleThreshold))
@parametrize_img(gray_only=True)
def test_simple_threshold(threshold_type: str, otsu: bool, threshold_value: int, max_value: int, img: Image):
    op = SimpleThreshold()
    op.threshold_type = threshold_type
    op.otsu = otsu
    op.threshold_value = threshold_value
    op.max_value = max_value
    r = run_operator(op, img)
    assert is_image(r)


@hypothesis.given(**make_hypothesis_parameters(AdaptiveThreshold))
@parametrize_img(gray_only=True)
def test_adaptive_threshold(
        threshold_type: str,
        adaptive_method: str,
        max_value: int,
        block_size: int,
        C: int,
        img: Image
):
    hypothesis.assume(block_size % 2 == 1)
    op = AdaptiveThreshold()
    op.threshold_type = threshold_type
    op.adaptive_method = adaptive_method
    op.max_value = max_value
    op.block_size = block_size
    op.C = C
    r = run_operator(op, img)
    assert is_image(r)
