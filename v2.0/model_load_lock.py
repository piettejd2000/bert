"""
Process-wide lock for loading models.

transformers' from_pretrained temporarily patches torch.nn.Module globally so new
parameters are created on the "meta" device (no data). Streamlit runs each session
and rerun in its own thread, so if two loads overlap, layers built by one thread
(e.g. the regressor in RobertaForGlobalDimensions) can come out as meta tensors and
fail with "Cannot copy out of meta tensor; no data!". Every model load in both
analyzer modules holds this one lock so loads never overlap.
"""
import threading

MODEL_LOAD_LOCK = threading.Lock()
