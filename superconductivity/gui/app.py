from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Callable, Optional

import numpy as np

from ..evaluation.traces import FileSpec, Keys, KeysSpec, Trace, TraceSpec, Traces
from ..evaluation.analysis import (
    OffsetSpec,
    OffsetTrace,
    OffsetTraces,
    PSDTraces,
    offset_analysis,
)
from ..evaluation.analysis import PSDSpec, PSDTrace, psd_analysis
from ..evaluation.sampling import (
    Sample,
    Samples,
    SamplingSpec,
    downsample_trace,
    sample,
)
from ..evaluation.sampling import SmoothingSpec, smooth
from ..optimizers.bcs import (
    BCSModelConfig,
    ParameterSpec,
    SolutionDict,
    get_model_config,
    get_model_key,
    get_model_spec,
    make_bcs_parameters,
    make_noise_parameters,
    make_pat_addon_parameters,
)
from ..utilities.types import NDArray64
from .common import _import_panel
from .display import GUILeftMixin
from .state import (
    GUIStateDict,
    _DEFAULT_MODEL,
    _DEFAULT_SHARED_NU_HZ,
    _default_offset_spec,
    _default_sampling_spec,
)
from .tabs import GUITabsMixin

_ACTIVE_GUI_SERVER = None
_ACTIVE_GUI_PANEL: "GUIPanel | None" = None


class GUIPanel(GUILeftMixin, GUITabsMixin):
    def __init__(
        self,
        traces: Traces,
        *,
        filespec: Optional[FileSpec] = None,
        keysspec: Optional[KeysSpec] = None,
        tracespec: Optional[TraceSpec] = None,
        keys: Optional[Keys] = None,
        psdspec: Optional[PSDSpec] = None,
        psdanalysis: Optional[PSDTrace | PSDTraces] = None,
        offsetspec: Optional[OffsetSpec] = None,
        offsetanalysis: Optional[OffsetTrace | OffsetTraces] = None,
        samples: Optional[Sample | Samples] = None,
        samplingspec: Optional[SamplingSpec] = None,
        smoothingspec: Optional[SmoothingSpec] = None,
        on_state_changed: Optional[Callable[[GUIStateDict], None]] = None,
    ) -> None:
        self._pn = _import_panel()
        if len(traces) == 0:
            raise ValueError("traces must not be empty.")

        self.traces = traces
        self._filespec = filespec
        self._keysspec = keysspec
        self._tracespec = tracespec
        self._keys = keys
        self.active_index = 0
        self._fit_model_config = get_model_config(_DEFAULT_MODEL)
        self.model_key = get_model_key(self._fit_model_config)
        self._on_state_changed = on_state_changed
        resolved_psd_spec = PSDSpec() if psdspec is None else replace(psdspec)
        self._shared_nu_Hz = _DEFAULT_SHARED_NU_HZ
        self._experimental_detrend = bool(resolved_psd_spec.detrend)
        resolved_offset_spec = self._resolve_initial_offset_spec(
            offsetspec=offsetspec,
            offsetanalysis=offsetanalysis,
        )
        self._offset_spec = resolved_offset_spec
        self._sampling_spec = (
            _default_sampling_spec()
            if samplingspec is None
            else replace(samplingspec)
        )
        self._smoothing_enabled = smoothingspec is not None
        self._smoothing_spec = (
            SmoothingSpec() if smoothingspec is None else replace(smoothingspec)
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
        self._initialize_sampling_offset_overrides()

        self._psd: PSDTrace | None = None
        self._downsampled_psd: PSDTrace | None = None
        self._offset: OffsetTrace | None = None
        self._sampling: Sample | None = None
        self._smoothed_sampling: Sample | None = None
        self._downsampled_trace: Trace | None = None
        self._fit_solution: Optional[SolutionDict] = None
        self._downsampled_t_s = np.empty((0,), dtype=np.float64)
        self._downsampled_V_mV = np.empty((0,), dtype=np.float64)
        self._downsampled_I_nA = np.empty((0,), dtype=np.float64)

        self._bcs_parameters = [
            replace(parameter) for parameter in make_bcs_parameters()
        ]
        self._pat_parameters = [
            replace(parameter) for parameter in make_pat_addon_parameters()
        ]
        self._noise_parameters = [
            replace(parameter) for parameter in make_noise_parameters()
        ]
        self._parameters: list[ParameterSpec] = []
        self._sync_active_fit_model()
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
        self._init_psd_stage_state()
        self._init_offset_batch_state()
        self._init_sampling_stage_state()

        self._offset_status = self._pn.pane.Markdown(
            sizing_mode="stretch_width",
        )
        self._fit_state = self._pn.pane.Markdown(
            "Idle",
            sizing_mode="stretch_width",
        )

        self._build_control_widgets()
        self._build_left_controls()
        if psdanalysis is not None:
            self._load_psd_analysis_preset(psdanalysis)
        if offsetanalysis is not None:
            self._load_offset_analysis_preset(offsetanalysis)
        if samples is not None:
            self._load_samples_preset(samples)
        self._recompute_pipeline(clear_fit=True)
        self._sync_control_widgets_from_specs()

        self._build_fit_widgets()
        self._build_plot_panes()
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
                "(V)",
                self._pn.Column(
                    self._left_v_quantity_selector,
                    self._iv_pane,
                    sizing_mode="stretch_width",
                ),
            ),
            (
                "(I)",
                self._pn.Column(
                    self._left_i_quantity_selector,
                    self._vi_pane,
                    sizing_mode="stretch_width",
                ),
            ),
            sizing_mode="stretch_width",
        )
        right_tabs = self._pn.Tabs(
            ("Measurement", self._measurement_tab()),
            ("Data", self._data_tab()),
            ("PSD Analysis", self._experimental_tab()),
            ("Offset Analysis", self._offset_tab()),
            ("Sampling", self._sampling_tab()),
            ("BCS fitting", self._fit_tab()),
            sizing_mode="stretch_width",
        )
        self._right_tabs = right_tabs
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

    def _active_trace(self) -> Trace:
        return self.traces[self.active_index]

    def _copy_stage_preset_entries(
        self,
        preset: object,
        *,
        collection_type: type,
        single_name: str,
        copy_fn: Callable[[object], object],
    ) -> list[object]:
        """Copy one positional preset into GUI-owned per-trace entries."""
        if isinstance(preset, collection_type):
            if len(preset) != len(self.traces):
                raise ValueError(
                    f"{collection_type.__name__} preset length must match "
                    "GUI trace count.",
                )
            return [copy_fn(result) for result in preset]

        if len(self.traces) != 1:
            raise ValueError(
                f"A bare {single_name} preset can only be loaded when the GUI "
                f"has exactly one trace. Use {collection_type.__name__} for "
                "multi-trace presets.",
            )
        return [copy_fn(preset)]

    @staticmethod
    def _copy_psd_trace(result: object) -> PSDTrace:
        """Return a GUI-owned copy of one PSD result."""
        psd = result
        return {
            "f_Hz": np.asarray(psd["f_Hz"], dtype=np.float64).copy(),
            "I_psd_nA2_per_Hz": np.asarray(
                psd["I_psd_nA2_per_Hz"],
                dtype=np.float64,
            ).copy(),
            "V_psd_mV2_per_Hz": np.asarray(
                psd["V_psd_mV2_per_Hz"],
                dtype=np.float64,
            ).copy(),
            "nu_Hz": float(psd["nu_Hz"]),
            "nyquist_Hz": float(psd["nyquist_Hz"]),
        }

    @staticmethod
    def _copy_sample_trace(result: object) -> Sample:
        """Return a GUI-owned copy of one sampled result."""
        sample_result = result
        return {
            "meta": sample_result["meta"],
            "Vbins_mV": np.asarray(
                sample_result["Vbins_mV"],
                dtype=np.float64,
            ).copy(),
            "Ibins_nA": np.asarray(
                sample_result["Ibins_nA"],
                dtype=np.float64,
            ).copy(),
            "I_nA": np.asarray(sample_result["I_nA"], dtype=np.float64).copy(),
            "V_mV": np.asarray(sample_result["V_mV"], dtype=np.float64).copy(),
            "dG_G0": np.asarray(sample_result["dG_G0"], dtype=np.float64).copy(),
            "dR_R0": np.asarray(sample_result["dR_R0"], dtype=np.float64).copy(),
        }

    def _resolve_initial_offset_spec(
        self,
        *,
        offsetspec: Optional[OffsetSpec],
        offsetanalysis: Optional[OffsetTrace | OffsetTraces],
    ) -> OffsetSpec:
        if offsetanalysis is not None and offsetspec is None:
            raise ValueError(
                "offsetspec is required when passing offsetanalysis.",
            )
        if offsetspec is None:
            return _default_offset_spec(self._shared_nu_Hz)
        return replace(offsetspec)

    def _init_psd_stage_state(self) -> None:
        self._psd_stage_spec: PSDSpec | None = None
        self._psd_stage_results: list[PSDTrace | None] = [
            None for _ in range(len(self.traces))
        ]

    def _init_sampling_stage_state(self) -> None:
        self._sampling_stage_spec: SamplingSpec | None = None
        self._sampling_stage_smoothing_enabled: bool | None = None
        self._sampling_stage_smoothing_spec: SmoothingSpec | None = None
        self._sampling_stage_results: list[Sample | None] = [
            None for _ in range(len(self.traces))
        ]

    @staticmethod
    def _psd_specs_match(left: PSDSpec, right: PSDSpec) -> bool:
        return bool(left.detrend is right.detrend)

    @staticmethod
    def _sampling_specs_match(left: SamplingSpec, right: SamplingSpec) -> bool:
        return bool(
            np.array_equal(left.Vbins_mV, right.Vbins_mV)
            and np.array_equal(left.Ibins_nA, right.Ibins_nA)
            and float(left.nu_Hz) == float(right.nu_Hz)
            and int(left.upsample) == int(right.upsample)
        )

    @staticmethod
    def _smoothing_specs_match(left: SmoothingSpec, right: SmoothingSpec) -> bool:
        return bool(
            int(left.median_bins) == int(right.median_bins)
            and float(left.sigma_bins) == float(right.sigma_bins)
            and str(left.mode) == str(right.mode)
        )

    def _current_psd_stage_spec(self) -> PSDSpec:
        return PSDSpec(detrend=self._experimental_detrend)

    def _current_sampling_stage_signature(
        self,
    ) -> tuple[SamplingSpec, bool, SmoothingSpec | None]:
        return (
            replace(self._sampling_spec),
            bool(self._smoothing_enabled),
            (replace(self._smoothing_spec) if self._smoothing_enabled else None),
        )

    def _clear_psd_stage_cache(self) -> None:
        self._psd_stage_spec = None
        self._psd_stage_results = [None for _ in range(len(self.traces))]

    def _clear_sampling_stage_cache(
        self,
        *,
        indices: list[int] | None = None,
    ) -> None:
        if indices is None:
            self._sampling_stage_spec = None
            self._sampling_stage_smoothing_enabled = None
            self._sampling_stage_smoothing_spec = None
            self._sampling_stage_results = [None for _ in range(len(self.traces))]
            return

        for index in indices:
            if 0 <= int(index) < len(self._sampling_stage_results):
                self._sampling_stage_results[int(index)] = None

    def _get_cached_psd_stage_result(
        self,
        index: int,
    ) -> PSDTrace | None:
        if self._psd_stage_spec is None:
            return None
        if not self._psd_specs_match(
            self._psd_stage_spec,
            self._current_psd_stage_spec(),
        ):
            return None
        if index < 0 or index >= len(self._psd_stage_results):
            return None
        return self._psd_stage_results[index]

    def _get_cached_sampling_stage_result(
        self,
        index: int,
    ) -> Sample | None:
        if self._sampling_stage_spec is None:
            return None
        current_spec, current_enabled, current_smoothing = (
            self._current_sampling_stage_signature()
        )
        if not self._sampling_specs_match(self._sampling_stage_spec, current_spec):
            return None
        if self._sampling_stage_smoothing_enabled != current_enabled:
            return None
        if current_enabled:
            stored = self._sampling_stage_smoothing_spec
            if stored is None or current_smoothing is None:
                return None
            if not self._smoothing_specs_match(stored, current_smoothing):
                return None
        if index < 0 or index >= len(self._sampling_stage_results):
            return None
        return self._sampling_stage_results[index]

    def _stage_psd_result(self, index: int, result: PSDTrace) -> None:
        self._psd_stage_spec = replace(self._current_psd_stage_spec())
        self._psd_stage_results[index] = self._copy_psd_trace(result)

    def _stage_sampling_result(self, index: int, result: Sample) -> None:
        sampling_spec, smoothing_enabled, smoothing_spec = (
            self._current_sampling_stage_signature()
        )
        self._sampling_stage_spec = sampling_spec
        self._sampling_stage_smoothing_enabled = smoothing_enabled
        self._sampling_stage_smoothing_spec = smoothing_spec
        self._sampling_stage_results[index] = self._copy_sample_trace(result)

    def _load_psd_analysis_preset(
        self,
        psd_analysis: PSDTrace | PSDTraces,
    ) -> None:
        copied = self._copy_stage_preset_entries(
            psd_analysis,
            collection_type=PSDTraces,
            single_name="PSDTrace",
            copy_fn=self._copy_psd_trace,
        )
        self._psd_stage_spec = replace(self._current_psd_stage_spec())
        self._psd_stage_results = [None for _ in range(len(self.traces))]
        for index, result in enumerate(copied):
            self._psd_stage_results[index] = result

    def _load_samples_preset(
        self,
        samples: Sample | Samples,
    ) -> None:
        copied = self._copy_stage_preset_entries(
            samples,
            collection_type=Samples,
            single_name="Sample",
            copy_fn=self._copy_sample_trace,
        )
        (
            self._sampling_stage_spec,
            self._sampling_stage_smoothing_enabled,
            self._sampling_stage_smoothing_spec,
        ) = self._current_sampling_stage_signature()
        self._sampling_stage_results = [None for _ in range(len(self.traces))]
        for index, result in enumerate(copied):
            self._sampling_stage_results[index] = result

    def _sync_active_fit_model(self) -> None:
        self.model_key = get_model_key(self._fit_model_config)
        parameters = list(self._bcs_parameters)
        if self._fit_model_config.pat_enabled:
            parameters.extend(self._pat_parameters)
        if self._fit_model_config.noise_enabled:
            parameters.extend(self._noise_parameters)
        self._parameters = parameters

    def _initialize_sampling_offset_overrides(self) -> None:
        self._sampling_override_Voff_mV = np.zeros(len(self.traces), dtype=np.float64)
        self._sampling_override_Ioff_nA = np.zeros(len(self.traces), dtype=np.float64)

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

    def _active_sampling_offsetanalysis(self) -> OffsetTrace | None:
        if self._active_sampling_override_enabled():
            Voff_mV, Ioff_nA = self._active_sampling_override_values()
            return {
                "dGerr_G0": np.asarray([], dtype=np.float64),
                "dRerr_R0": np.asarray([], dtype=np.float64),
                "Voff_mV": float(Voff_mV),
                "Ioff_nA": float(Ioff_nA),
            }
        if self._offset is None:
            return None
        return self._offset

    def _current_guess(self) -> NDArray64:
        return np.asarray(
            [parameter.guess for parameter in self._parameters],
            dtype=np.float64,
        )

    def _require_psd(self) -> PSDTrace:
        if self._psd is None:
            raise RuntimeError("PSD state is not available.")
        return self._psd

    def _require_downsampled_psd(self) -> PSDTrace:
        if self._downsampled_psd is None:
            raise RuntimeError("Downsampled PSD state is not available.")
        return self._downsampled_psd

    def _require_offset(self) -> OffsetTrace:
        if self._offset is None:
            raise RuntimeError("Offset state is not available.")
        return self._offset

    def _require_raw_sampling(self) -> Sample:
        if self._sampling is None:
            raise RuntimeError("Sampling state is not available.")
        return self._sampling

    def _require_sampling(self) -> Sample:
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
        cached_raw_psd = self._get_cached_psd_stage_result(self.active_index)
        if cached_raw_psd is not None:
            self._psd = self._copy_psd_trace(cached_raw_psd)
        else:
            self._psd = psd_analysis(
                trace,
                spec=self._current_psd_stage_spec(),
            )
        downsampled_trace = downsample_trace(trace, nu_Hz=self._shared_nu_Hz)
        self._downsampled_psd = psd_analysis(
            downsampled_trace,
            spec=PSDSpec(detrend=self._experimental_detrend),
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
        cached_offset = self._get_cached_offset_batch_result(self.active_index)
        if cached_offset is not None:
            self._offset = cached_offset
            return
        self._offset = offset_analysis(self._active_trace(), spec=self._offset_spec)

    def _require_downsampled_trace(self) -> Trace:
        if self._downsampled_trace is None:
            raise RuntimeError("Downsampled trace is not available.")
        return self._downsampled_trace

    def _compute_sampling_stage(self) -> None:
        cached_sample = self._get_cached_sampling_stage_result(self.active_index)
        if cached_sample is not None:
            self._sampling = self._copy_sample_trace(cached_sample)
            self._smoothed_sampling = (
                self._copy_sample_trace(cached_sample)
                if self._smoothing_enabled
                else None
            )
            return

        self._sampling = sample(
            self._active_trace(),
            samplingspec=self._sampling_spec,
            offsetanalysis=self._active_sampling_offsetanalysis(),
            show_progress=False,
        )
        if self._smoothing_enabled:
            self._smoothed_sampling = smooth(
                self._require_raw_sampling(),
                smoothingspec=self._smoothing_spec,
            )
        else:
            self._smoothed_sampling = None

    def _recompute_fit_curves(self) -> None:
        sampling = self._require_sampling()
        V_mV = np.asarray(sampling["Vbins_mV"], dtype=np.float64)
        function = get_model_spec(self._fit_model_config).function
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
        self._refresh_data_views()
        self._refresh_experimental_views()
        self._refresh_offset_views()
        self._refresh_offset_batch_views()
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
    traces: Traces,
    *,
    filespec: Optional[FileSpec] = None,
    keysspec: Optional[KeysSpec] = None,
    tracespec: Optional[TraceSpec] = None,
    keys: Optional[Keys] = None,
    psdspec: Optional[PSDSpec] = None,
    psdanalysis: Optional[PSDTrace | PSDTraces] = None,
    offsetspec: Optional[OffsetSpec] = None,
    offsetanalysis: Optional[OffsetTrace | OffsetTraces] = None,
    samples: Optional[Sample | Samples] = None,
    samplingspec: Optional[SamplingSpec] = None,
    smoothingspec: Optional[SmoothingSpec] = None,
    on_state_changed: Optional[Callable[[GUIStateDict], None]] = None,
):
    panel = GUIPanel(
        traces,
        filespec=filespec,
        keysspec=keysspec,
        tracespec=tracespec,
        keys=keys,
        psdspec=psdspec,
        psdanalysis=psdanalysis,
        offsetspec=offsetspec,
        offsetanalysis=offsetanalysis,
        samples=samples,
        samplingspec=samplingspec,
        smoothingspec=smoothingspec,
        on_state_changed=on_state_changed,
    )
    layout = panel.layout
    setattr(layout, "_gui_panel", panel)
    return layout


def serve_gui(
    traces: Traces,
    *,
    filespec: Optional[FileSpec] = None,
    keysspec: Optional[KeysSpec] = None,
    tracespec: Optional[TraceSpec] = None,
    keys: Optional[Keys] = None,
    psdspec: Optional[PSDSpec] = None,
    psdanalysis: Optional[PSDTrace | PSDTraces] = None,
    offsetspec: Optional[OffsetSpec] = None,
    offsetanalysis: Optional[OffsetTrace | OffsetTraces] = None,
    samples: Optional[Sample | Samples] = None,
    samplingspec: Optional[SamplingSpec] = None,
    smoothingspec: Optional[SmoothingSpec] = None,
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
        filespec=filespec,
        keysspec=keysspec,
        tracespec=tracespec,
        keys=keys,
        psdspec=psdspec,
        psdanalysis=psdanalysis,
        offsetspec=offsetspec,
        offsetanalysis=offsetanalysis,
        samples=samples,
        samplingspec=samplingspec,
        smoothingspec=smoothingspec,
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
    traces: Traces,
    *,
    filespec: Optional[FileSpec] = None,
    keysspec: Optional[KeysSpec] = None,
    tracespec: Optional[TraceSpec] = None,
    keys: Optional[Keys] = None,
    psdspec: Optional[PSDSpec] = None,
    psdanalysis: Optional[PSDTrace | PSDTraces] = None,
    offsetspec: Optional[OffsetSpec] = None,
    offsetanalysis: Optional[OffsetTrace | OffsetTraces] = None,
    samples: Optional[Sample | Samples] = None,
    samplingspec: Optional[SamplingSpec] = None,
    smoothingspec: Optional[SmoothingSpec] = None,
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
        filespec=filespec,
        keysspec=keysspec,
        tracespec=tracespec,
        keys=keys,
        psdspec=psdspec,
        psdanalysis=psdanalysis,
        offsetspec=offsetspec,
        offsetanalysis=offsetanalysis,
        samples=samples,
        samplingspec=samplingspec,
        smoothingspec=smoothingspec,
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
    traces: Traces,
    *,
    filespec: Optional[FileSpec] = None,
    keysspec: Optional[KeysSpec] = None,
    tracespec: Optional[TraceSpec] = None,
    keys: Optional[Keys] = None,
    psdspec: Optional[PSDSpec] = None,
    psdanalysis: Optional[PSDTrace | PSDTraces] = None,
    offsetspec: Optional[OffsetSpec] = None,
    offsetanalysis: Optional[OffsetTrace | OffsetTraces] = None,
    samples: Optional[Sample | Samples] = None,
    samplingspec: Optional[SamplingSpec] = None,
    smoothingspec: Optional[SmoothingSpec] = None,
    port: int = 0,
    open_browser: bool = True,
    threaded: bool = True,
    title: str = "Experimental Analysis GUI",
    verbose: bool = True,
    stop_existing: bool = True,
    wait: bool = True,
    poll_interval: float = 0.5,
):
    if wait:
        return run_gui(
            traces,
            filespec=filespec,
            keysspec=keysspec,
            tracespec=tracespec,
            keys=keys,
            psdspec=psdspec,
            psdanalysis=psdanalysis,
            offsetspec=offsetspec,
            offsetanalysis=offsetanalysis,
            samples=samples,
            samplingspec=samplingspec,
            smoothingspec=smoothingspec,
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
        filespec=filespec,
        keysspec=keysspec,
        tracespec=tracespec,
        keys=keys,
        psdspec=psdspec,
        psdanalysis=psdanalysis,
        offsetspec=offsetspec,
        offsetanalysis=offsetanalysis,
        samples=samples,
        samplingspec=samplingspec,
        smoothingspec=smoothingspec,
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
