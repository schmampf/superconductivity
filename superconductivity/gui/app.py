from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Callable, Optional

import numpy as np

from ..evaluation.ivdata import IVTrace, IVTraces
from ..evaluation.offset import OffsetSpec, OffsetTrace, get_offset
from ..evaluation.psd import PSDSpec, PSDTrace, get_psd
from ..evaluation.sampling import (
    SamplingSpec,
    SamplingTrace,
    get_sampling,
)
from ..evaluation.smoothing import SmoothingSpec, get_smoothed_sampling
from ..optimizers.fit_model import SolutionDict
from ..optimizers.models import ParameterSpec, get_model_spec
from ..utilities.types import NDArray64
from .common import _import_panel
from .display import GUILeftMixin
from .state import (
    GUIStateDict,
    _DEFAULT_MODEL,
    _default_offset_spec,
    _default_sampling_spec,
)
from .tabs import GUITabsMixin

_ACTIVE_GUI_SERVER = None
_ACTIVE_GUI_PANEL: "GUIPanel | None" = None


class GUIPanel(GUILeftMixin, GUITabsMixin):
    def __init__(
        self,
        traces: IVTraces,
        *,
        model: str = _DEFAULT_MODEL,
        psd_nu_Hz: float = 13.7,
        offset_spec: Optional[OffsetSpec] = None,
        sampling_spec: Optional[SamplingSpec] = None,
        maxfev: Optional[int] = None,
        on_state_changed: Optional[
            Callable[[GUIStateDict], None]
        ] = None,
    ) -> None:
        self._pn = _import_panel()
        if len(traces) == 0:
            raise ValueError("traces must not be empty.")

        self.traces = traces
        self.active_index = 0
        self.model_key = model
        self.maxfev = maxfev
        self._on_state_changed = on_state_changed
        self._shared_nu_Hz = float(psd_nu_Hz)
        self._offset_spec = (
            _default_offset_spec(self._shared_nu_Hz)
            if offset_spec is None
            else replace(offset_spec)
        )
        self._sampling_spec = (
            _default_sampling_spec()
            if sampling_spec is None
            else replace(sampling_spec)
        )
        self._sampling_offset_override_enabled = np.zeros(
            len(traces),
            dtype=bool,
        )
        self._sampling_override_Voff_mV = np.zeros(
            len(traces),
            dtype=np.float64,
        )
        self._sampling_override_Ioff_nA = np.zeros(
            len(traces),
            dtype=np.float64,
        )
        self._initialize_sampling_offset_overrides(
            enable=bool(sampling_spec is not None),
        )
        self._sampling_spec = replace(
            self._sampling_spec,
            Voff_mV=0.0,
            Ioff_nA=0.0,
        )

        self._psd: PSDTrace | None = None
        self._offset: OffsetTrace | None = None
        self._sampling: SamplingTrace | None = None
        self._smoothed_sampling: SamplingTrace | None = None
        self._downsampled_trace: IVTrace | None = None
        self._fit_solution: Optional[SolutionDict] = None
        self._downsampled_t_s = np.empty((0,), dtype=np.float64)
        self._downsampled_V_mV = np.empty((0,), dtype=np.float64)
        self._downsampled_I_nA = np.empty((0,), dtype=np.float64)

        self._parameters = self._default_parameters()
        self._experimental_detrend = True
        self._smoothing_enabled = False
        self._smoothing_spec = SmoothingSpec()
        self._initial_curve = np.empty((0,), dtype=np.float64)
        self._fit_curve = np.empty((0,), dtype=np.float64)
        self._fit_running = False
        self._fit_started_at: float | None = None
        self._fit_timer = None
        self._fit_future = None
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._last_fit_error = ""
        self._last_fit_elapsed_s = 0.0
        self._last_fit_status = "idle"

        self._offset_status = self._pn.pane.Markdown(
            sizing_mode="stretch_width",
        )
        self._fit_state = self._pn.pane.Markdown(
            "Idle",
            sizing_mode="stretch_width",
        )

        self._build_control_widgets()
        self._recompute_pipeline(clear_fit=True)
        self._sync_control_widgets_from_specs()

        self._build_fit_widgets()
        self._build_plot_panes()
        self._build_left_controls()
        self._refresh_all_views()

        self.layout = self.get_layout()
        self._notify_state_changed()

    @property
    def state(self) -> GUIStateDict:
        if self._psd is None or self._offset is None or self._sampling is None:
            raise RuntimeError("GUI pipeline state is incomplete.")
        return {
            "active_index": self.active_index,
            "trace": self._active_trace(),
            "psd": self._psd,
            "offset": self._offset,
            "sampling": self._require_sampling(),
            "fit": self._fit_solution,
        }


    def get_layout(self):
        left_tabs = self._pn.Tabs(
            (
                "I(V)",
                self._pn.Column(
                    self._iv_pane,
                    self._didv_pane,
                    sizing_mode="stretch_width",
                ),
            ),
            (
                "V(I)",
                self._pn.Column(
                    self._vi_pane,
                    self._dvdi_pane,
                    sizing_mode="stretch_width",
                ),
            ),
            sizing_mode="stretch_width",
        )
        right_tabs = self._pn.Tabs(
            ("PSD Analysis", self._experimental_tab()),
            ("Offset Analysis", self._offset_tab()),
            ("Sampling", self._sampling_tab()),
            ("Fitting", self._fit_tab()),
            sizing_mode="stretch_width",
        )
        return self._pn.Row(
            self._pn.Column(
                self._trace_selector,
                self._left_stage_selector,
                left_tabs,
                min_width=720,
                sizing_mode="stretch_width",
            ),
            self._pn.Column(
                right_tabs,
                min_width=620,
                sizing_mode="stretch_width",
            ),
            sizing_mode="stretch_width",
        )


    def _active_trace(self) -> IVTrace:
        return self.traces[self.active_index]


    def _default_parameters(self) -> list[ParameterSpec]:
        return [
            replace(parameter)
            for parameter in get_model_spec(self.model_key).parameters
        ]


    def _sampling_offset_values(
        self,
        values: float | NDArray64,
        *,
        name: str,
    ) -> NDArray64:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim == 0 or array.size == 1:
            return np.full(len(self.traces), float(array.reshape(-1)[0]))
        if array.ndim != 1 or array.size != len(self.traces):
            raise ValueError(
                f"{name} must be scalar or have length {len(self.traces)}.",
            )
        return np.asarray(array, dtype=np.float64)


    def _initialize_sampling_offset_overrides(self, *, enable: bool) -> None:
        self._sampling_override_Voff_mV = self._sampling_offset_values(
            self._sampling_spec.Voff_mV,
            name="Voff_mV",
        )
        self._sampling_override_Ioff_nA = self._sampling_offset_values(
            self._sampling_spec.Ioff_nA,
            name="Ioff_nA",
        )
        if enable:
            self._sampling_offset_override_enabled[:] = True


    def _active_sampling_override_enabled(self) -> bool:
        return bool(self._sampling_offset_override_enabled[self.active_index])


    def _active_sampling_override_values(self) -> tuple[float, float]:
        return (
            float(self._sampling_override_Voff_mV[self.active_index]),
            float(self._sampling_override_Ioff_nA[self.active_index]),
        )


    def _active_sampling_offset_values(self) -> tuple[float, float]:
        if self._active_sampling_override_enabled():
            return self._active_sampling_override_values()
        if self._offset is not None:
            return (
                float(self._offset["Voff_mV"]),
                float(self._offset["Ioff_nA"]),
            )
        return (0.0, 0.0)


    def _set_active_sampling_override(
        self,
        *,
        enabled: bool,
        Voff_mV: float,
        Ioff_nA: float,
    ) -> None:
        self._sampling_offset_override_enabled[self.active_index] = bool(enabled)
        self._sampling_override_Voff_mV[self.active_index] = float(Voff_mV)
        self._sampling_override_Ioff_nA[self.active_index] = float(Ioff_nA)


    def _active_sampling_spec(self) -> SamplingSpec:
        Voff_mV, Ioff_nA = self._active_sampling_offset_values()
        return replace(
            self._sampling_spec,
            Voff_mV=Voff_mV,
            Ioff_nA=Ioff_nA,
        )


    def _current_guess(self) -> NDArray64:
        return np.asarray(
            [parameter.guess for parameter in self._parameters],
            dtype=np.float64,
        )


    def _require_psd(self) -> PSDTrace:
        if self._psd is None:
            raise RuntimeError("PSD state is not available.")
        return self._psd


    def _require_offset(self) -> OffsetTrace:
        if self._offset is None:
            raise RuntimeError("Offset state is not available.")
        return self._offset


    def _require_raw_sampling(self) -> SamplingTrace:
        if self._sampling is None:
            raise RuntimeError("Sampling state is not available.")
        return self._sampling


    def _require_sampling(self) -> SamplingTrace:
        if self._smoothing_enabled and self._smoothed_sampling is not None:
            return self._smoothed_sampling
        return self._require_raw_sampling()


    def _notify_state_changed(self) -> None:
        if self._on_state_changed is not None:
            self._on_state_changed(self.state)


    def _set_shared_nu_Hz(self, value: float) -> bool:
        value = float(value)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("nu_Hz must be finite and > 0.")
        changed = not np.isclose(value, self._shared_nu_Hz, rtol=0.0, atol=0.0)
        self._shared_nu_Hz = value
        return changed


    def _compute_psd_stage(self) -> None:
        trace = self._active_trace()
        downsampled_trace, self._psd = get_psd(
            trace,
            spec=PSDSpec(
                nu_Hz=self._shared_nu_Hz,
                detrend=self._experimental_detrend,
            ),
        )
        self._downsampled_trace = downsampled_trace
        self._downsampled_t_s = np.asarray(
            downsampled_trace["t_s"],
            dtype=np.float64,
        )
        self._downsampled_V_mV = np.asarray(
            downsampled_trace["V_mV"],
            dtype=np.float64,
        )
        self._downsampled_I_nA = np.asarray(
            downsampled_trace["I_nA"],
            dtype=np.float64,
        )


    def _compute_offset_stage(self) -> None:
        self._offset = get_offset(self._active_trace(), spec=self._offset_spec)


    def _require_downsampled_trace(self) -> IVTrace:
        if self._downsampled_trace is None:
            raise RuntimeError("Downsampled trace is not available.")
        return self._downsampled_trace


    def _compute_sampling_stage(self) -> None:
        self._sampling = get_sampling(
            self._require_downsampled_trace(),
            spec=self._active_sampling_spec(),
        )
        if self._smoothing_enabled:
            self._smoothed_sampling = get_smoothed_sampling(
                self._require_raw_sampling(),
                spec=self._smoothing_spec,
            )
        else:
            self._smoothed_sampling = None


    def _recompute_fit_curves(self) -> None:
        sampling = self._require_sampling()
        V_mV = np.asarray(sampling["Vbin_mV"], dtype=np.float64)
        function = get_model_spec(self.model_key).function
        self._initial_curve = np.asarray(
            function(V_mV, *self._current_guess()),
            dtype=np.float64,
        )
        if self._fit_solution is None:
            self._fit_curve = self._initial_curve.copy()
        else:
            self._fit_curve = np.asarray(
                self._fit_solution["I_fit_nA"],
                dtype=np.float64,
            )


    def _clear_fit_solution(self) -> None:
        self._fit_solution = None
        self._reset_fit_status()


    def _set_fit_solution(self, solution: Optional[SolutionDict]) -> None:
        self._fit_solution = solution


    def _recompute_pipeline(
        self,
        *,
        clear_fit: bool,
        recompute_psd: bool = True,
        recompute_offset: bool = True,
        recompute_sampling: bool = True,
    ) -> None:
        if recompute_psd:
            self._compute_psd_stage()
        if recompute_offset:
            self._compute_offset_stage()
        if recompute_sampling:
            self._compute_sampling_stage()
        if clear_fit:
            self._clear_fit_solution()
        self._recompute_fit_curves()


    def _refresh_all_views(self) -> None:
        self._refresh_left_plots()
        self._refresh_experimental_views()
        self._refresh_offset_views()
        self._refresh_sampling_views()
        self._refresh_fit_views()


    def _on_trace_changed(self, event: object) -> None:
        new_value = int(getattr(event, "new"))
        old_value = int(getattr(event, "old"))
        if new_value == old_value:
            return
        if self._fit_running:
            self._trace_selector.value = old_value
            return
        self.active_index = new_value
        self._recompute_pipeline(clear_fit=True)
        self._sync_control_widgets_from_specs()
        self._refresh_all_views()
        self._notify_state_changed()


def gui_app(
    traces: IVTraces,
    *,
    model: str = _DEFAULT_MODEL,
    psd_nu_Hz: float = 13.7,
    offset_spec: Optional[OffsetSpec] = None,
    sampling_spec: Optional[SamplingSpec] = None,
    maxfev: Optional[int] = None,
    on_state_changed: Optional[Callable[[GUIStateDict], None]] = None,
):
    panel = GUIPanel(
        traces,
        model=model,
        psd_nu_Hz=psd_nu_Hz,
        offset_spec=offset_spec,
        sampling_spec=sampling_spec,
        maxfev=maxfev,
        on_state_changed=on_state_changed,
    )
    layout = panel.layout
    setattr(layout, "_gui_panel", panel)
    return layout


def serve_gui(
    traces: IVTraces,
    *,
    model: str = _DEFAULT_MODEL,
    psd_nu_Hz: float = 13.7,
    offset_spec: Optional[OffsetSpec] = None,
    sampling_spec: Optional[SamplingSpec] = None,
    maxfev: Optional[int] = None,
    port: int = 0,
    open_browser: bool = True,
    threaded: bool = True,
    title: str = "Experimental Analysis GUI",
    verbose: bool = True,
    stop_existing: bool = True,
):
    global _ACTIVE_GUI_PANEL, _ACTIVE_GUI_SERVER

    pn = _import_panel()
    if (
        stop_existing
        and _ACTIVE_GUI_SERVER is not None
        and hasattr(_ACTIVE_GUI_SERVER, "stop")
    ):
        _ACTIVE_GUI_SERVER.stop()
        _ACTIVE_GUI_SERVER = None
        _ACTIVE_GUI_PANEL = None

    app = gui_app(
        traces,
        model=model,
        psd_nu_Hz=psd_nu_Hz,
        offset_spec=offset_spec,
        sampling_spec=sampling_spec,
        maxfev=maxfev,
    )
    _ACTIVE_GUI_PANEL = getattr(app, "_gui_panel")
    _ACTIVE_GUI_SERVER = pn.serve(
        app,
        port=port,
        show=open_browser,
        threaded=threaded,
        title=title,
        verbose=verbose,
    )
    return _ACTIVE_GUI_SERVER


def run_gui(
    traces: IVTraces,
    *,
    model: str = _DEFAULT_MODEL,
    psd_nu_Hz: float = 13.7,
    offset_spec: Optional[OffsetSpec] = None,
    sampling_spec: Optional[SamplingSpec] = None,
    maxfev: Optional[int] = None,
    port: int = 0,
    open_browser: bool = True,
    threaded: bool = True,
    title: str = "Experimental Analysis GUI",
    verbose: bool = True,
    stop_existing: bool = True,
    wait: bool = True,
    poll_interval: float = 0.5,
) -> Optional[GUIStateDict]:
    global _ACTIVE_GUI_PANEL

    server = serve_gui(
        traces,
        model=model,
        psd_nu_Hz=psd_nu_Hz,
        offset_spec=offset_spec,
        sampling_spec=sampling_spec,
        maxfev=maxfev,
        port=port,
        open_browser=open_browser,
        threaded=threaded,
        title=title,
        verbose=verbose,
        stop_existing=stop_existing,
    )
    if not wait:
        return None

    try:
        while getattr(server, "is_alive", lambda: False)():
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        if hasattr(server, "stop"):
            server.stop()
    return None if _ACTIVE_GUI_PANEL is None else _ACTIVE_GUI_PANEL.state


def gui(
    traces: IVTraces,
    *,
    model: str = _DEFAULT_MODEL,
    psd_nu_Hz: float = 13.7,
    offset_spec: Optional[OffsetSpec] = None,
    sampling_spec: Optional[SamplingSpec] = None,
    maxfev: Optional[int] = None,
    port: int = 0,
    open_browser: bool = True,
    threaded: bool = True,
    title: str = "Experimental Analysis GUI",
    verbose: bool = True,
    stop_existing: bool = True,
    wait: bool = True,
    poll_interval: float = 0.5,
) -> Optional[GUIStateDict]:
    if wait:
        return run_gui(
            traces,
            model=model,
            psd_nu_Hz=psd_nu_Hz,
            offset_spec=offset_spec,
            sampling_spec=sampling_spec,
            maxfev=maxfev,
            port=port,
            open_browser=open_browser,
            threaded=threaded,
            title=title,
            verbose=verbose,
            stop_existing=stop_existing,
            wait=True,
            poll_interval=poll_interval,
        )

    serve_gui(
        traces,
        model=model,
        psd_nu_Hz=psd_nu_Hz,
        offset_spec=offset_spec,
        sampling_spec=sampling_spec,
        maxfev=maxfev,
        port=port,
        open_browser=open_browser,
        threaded=threaded,
        title=title,
        verbose=verbose,
        stop_existing=stop_existing,
    )
    return None


__all__ = [
    "GUIPanel",
    "GUIStateDict",
    "gui",
    "gui_app",
    "run_gui",
    "serve_gui",
]
