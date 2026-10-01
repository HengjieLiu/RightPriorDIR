"""Audited registration/data kernels; see docs/methods.md for protocol scope."""

from __future__ import annotations
import argparse
import json
import math
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple
import numpy as np
import torch
from . import oasis as c

CheckpointSampler = object


def _json_clean(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return str(tuple(value.shape))
    if isinstance(value, dict):
        return {str(k): _json_clean(v) for (k, v) in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(v) for v in value]
    return value


@contextmanager
def _null_section() -> Iterator[None]:
    yield


class TimingRecorder:
    """Hierarchical wall-clock timing with optional CUDA synchronization."""

    def __init__(
        self, detail: str = "off", device: Optional[torch.device] = None
    ) -> None:
        self.detail = str(detail)
        self.enabled = self.detail != "off"
        self.device = device
        self.root_id = "script"
        self.root_start = time.perf_counter()
        self.root_end: Optional[float] = None
        self.stack: List[str] = [self.root_id]
        self.events: List[Dict[str, object]] = []
        self._counter = 0
        self._mark_counter = 0
        self.notes: List[Dict[str, object]] = []
        self.marks: List[Dict[str, object]] = []
        self.stage_phase_summaries: List[Dict[str, object]] = []
        self.components = None
        if self.detail == "components":
            from scripts.experiments.mrdbscp842_acceleration.components import ACTIVE

            self.components = ACTIVE.get()
            if self.components is None:
                raise RuntimeError(
                    "Component timing requires the main instrumentation context"
                )
            self.components.origin = self.root_start

    def set_device(self, device: torch.device) -> None:
        self.device = device
        if self.components is not None:
            self.components.device = device

    def sync(self) -> None:
        if not self.enabled:
            return
        if (
            self.device is not None
            and self.device.type == "cuda"
            and torch.cuda.is_available()
        ):
            torch.cuda.synchronize(self.device)

    def start(
        self,
        name: str,
        *,
        category: str = "",
        metadata: Optional[Dict[str, object]] = None,
        sync: bool = True,
    ) -> Optional[Dict[str, object]]:
        if not self.enabled or (self.detail == "endpoints" and name != "setup.device"):
            return None
        if self.components is not None:
            sync = False
        if sync:
            self.sync()
        now = time.perf_counter()
        self._counter += 1
        token: Dict[str, object] = {
            "id": f"event_{self._counter:05d}",
            "name": str(name),
            "parent_id": self.stack[-1],
            "parent_name": self.stack[-1],
            "category": str(category),
            "metadata": _json_clean(metadata or {}),
            "start_perf": now,
            "start_offset_sec": now - self.root_start,
            "depth": max(0, len(self.stack) - 1),
            "sync": bool(sync),
        }
        if self.components is not None:
            token["component_token"] = self.components.start(str(name))
        self.stack.append(str(token["id"]))
        return token

    def stop(
        self, token: Optional[Dict[str, object]], *, sync: Optional[bool] = None
    ) -> float:
        if not self.enabled or token is None:
            return 0.0
        do_sync = bool(token.get("sync")) if sync is None else bool(sync)
        if do_sync:
            self.sync()
        now = time.perf_counter()
        if self.components is not None:
            self.components.stop(token["component_token"])
        event_id = str(token["id"])
        if not self.stack or self.stack[-1] != event_id:
            raise RuntimeError(f"Timing stack mismatch while stopping {event_id}.")
        self.stack.pop()
        elapsed = now - float(token["start_perf"])
        parent_id = str(token["parent_id"])
        parent_name = "script" if parent_id == self.root_id else parent_id
        event = {
            "id": event_id,
            "name": str(token["name"]),
            "parent_id": parent_id,
            "parent_name": parent_name,
            "category": str(token.get("category", "")),
            "metadata": token.get("metadata", {}),
            "start_offset_sec": float(token["start_offset_sec"]),
            "end_offset_sec": now - self.root_start,
            "elapsed_sec": float(elapsed),
            "depth": int(token["depth"]),
        }
        self.events.append(event)
        return float(elapsed)

    @contextmanager
    def section(
        self,
        name: str,
        *,
        category: str = "",
        metadata: Optional[Dict[str, object]] = None,
        sync: bool = True,
    ) -> Iterator[None]:
        token = self.start(name, category=category, metadata=metadata, sync=sync)
        try:
            yield
        finally:
            self.stop(token)

    def add_note(self, name: str, payload: Dict[str, object]) -> None:
        if self.enabled:
            self.notes.append({"name": str(name), "payload": _json_clean(payload)})

    def mark(
        self,
        name: str,
        *,
        metadata: Optional[Dict[str, object]] = None,
        sync: bool = True,
    ) -> Optional[Dict[str, object]]:
        if not self.enabled:
            return None
        if sync:
            self.sync()
        now = time.perf_counter()
        self._mark_counter += 1
        parent_id = self.stack[-1]
        parent_name = "script" if parent_id == self.root_id else parent_id
        row = {
            "id": f"mark_{self._mark_counter:05d}",
            "name": str(name),
            "parent_id": parent_id,
            "parent_name": parent_name,
            "offset_sec": float(now - self.root_start),
            "metadata": _json_clean(metadata or {}),
            "sync": bool(sync),
        }
        self.marks.append(row)
        if self.components is not None:
            self.components.marks[str(name)] = now - self.components.origin
        return row

    def add_stage_phase_summary(self, summary: Dict[str, object]) -> None:
        if self.enabled:
            self.stage_phase_summaries.append(_json_clean(summary))

    def finalize(self) -> None:
        if self.enabled:
            self.sync()
            self.root_end = time.perf_counter()

    def reconciliation(self) -> List[Dict[str, object]]:
        if not self.enabled:
            return []
        root_elapsed = (self.root_end or time.perf_counter()) - self.root_start
        elapsed_by_id = {
            str(event["id"]): float(event["elapsed_sec"]) for event in self.events
        }
        elapsed_by_id[self.root_id] = float(root_elapsed)
        names_by_id = {str(event["id"]): str(event["name"]) for event in self.events}
        names_by_id[self.root_id] = "script"
        children_by_parent: Dict[str, float] = {}
        counts_by_parent: Dict[str, int] = {}
        for event in self.events:
            parent_id = str(event["parent_id"])
            children_by_parent[parent_id] = children_by_parent.get(
                parent_id, 0.0
            ) + float(event["elapsed_sec"])
            counts_by_parent[parent_id] = counts_by_parent.get(parent_id, 0) + 1
        rows: List[Dict[str, object]] = []
        for node_id, elapsed in elapsed_by_id.items():
            children = children_by_parent.get(node_id, 0.0)
            child_count = int(counts_by_parent.get(node_id, 0))
            self_unaccounted = elapsed - children
            tolerance_sec = max(0.5, 0.01 * max(abs(elapsed), 1e-09))
            rows.append(
                {
                    "id": node_id,
                    "name": names_by_id.get(node_id, node_id),
                    "elapsed_sec": float(elapsed),
                    "child_elapsed_sec": float(children),
                    "self_unaccounted_sec": float(self_unaccounted),
                    "abs_self_unaccounted_sec": float(abs(self_unaccounted)),
                    "child_count": child_count,
                    "tolerance_sec": float(tolerance_sec),
                    "within_tolerance": bool(
                        child_count == 0 or abs(self_unaccounted) <= tolerance_sec
                    ),
                }
            )
        return sorted(
            rows,
            key=lambda row: (0 if row["id"] == self.root_id else 1, str(row["id"])),
        )

    def payload(
        self, *, args: argparse.Namespace, run_dir: Path, run_name: str
    ) -> Dict[str, object]:
        root_elapsed = (self.root_end or time.perf_counter()) - self.root_start
        return {
            "enabled": bool(self.enabled),
            "detail": self.detail,
            "created_at": c.now_text(),
            "run_name": str(run_name),
            "run_dir": str(run_dir),
            "method": str(args.method),
            "runtime_candidate": str(args.runtime_candidate) or None,
            "output_mode": str(getattr(args, "output_mode", "full")),
            "case_rank": int(args.case_rank),
            "cudnn_requested": {
                "benchmark": bool(getattr(args, "cudnn_benchmark", True)),
                "deterministic": bool(getattr(args, "cudnn_deterministic", False)),
            },
            "cuda_backend": c.cuda_backend_metadata(self.device),
            "script_total_sec": float(root_elapsed),
            "events": sorted(
                self.events, key=lambda row: float(row["start_offset_sec"])
            ),
            "marks": sorted(self.marks, key=lambda row: float(row["offset_sec"])),
            "reconciliation": self.reconciliation(),
            "stage_phase_summaries": list(self.stage_phase_summaries),
            "notes": list(self.notes),
            **(
                {"components": self.components.payload()}
                if self.components is not None
                else {}
            ),
        }


class CudaPhaseAccumulator:
    """Summarize optimizer-loop CUDA phase events without per-epoch log rows."""

    def __init__(self, enabled: bool, device: torch.device) -> None:
        self.enabled = bool(
            enabled and device.type == "cuda" and torch.cuda.is_available()
        )
        self.device = device
        self.events: Dict[str, List[Tuple[torch.cuda.Event, torch.cuda.Event]]] = {}
        self.wall_sec: Dict[str, float] = {}

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        if not self.enabled:
            start_wall = time.perf_counter()
            try:
                yield
            finally:
                self.wall_sec[name] = self.wall_sec.get(name, 0.0) + (
                    time.perf_counter() - start_wall
                )
            return
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            self.events.setdefault(str(name), []).append((start, end))

    def summary(self) -> Dict[str, float]:
        if self.enabled:
            torch.cuda.synchronize(self.device)
            return {
                str(name): float(
                    sum((start.elapsed_time(end) for (start, end) in pairs)) / 1000.0
                )
                for (name, pairs) in sorted(self.events.items())
            }
        return {
            str(name): float(value) for (name, value) in sorted(self.wall_sec.items())
        }


def timed_evaluate_from_dense_coords(
    dense_coords: torch.Tensor,
    batch: Dict[str, torch.Tensor | str | int | Dict[str, object]],
    timer: Optional[TimingRecorder] = None,
) -> Dict[str, object]:
    if timer is None or not timer.enabled:
        return c.evaluate_from_dense_coords(dense_coords, batch)
    fixed = batch["fixed"]
    moving = batch["moving"]
    moving_seg = batch["moving_seg"]
    fixed_seg = batch["fixed_seg"]
    fixed_mask = batch["fixed_mask"]
    assert isinstance(fixed, torch.Tensor)
    assert isinstance(moving, torch.Tensor)
    assert isinstance(moving_seg, torch.Tensor)
    assert isinstance(fixed_seg, torch.Tensor)
    assert isinstance(fixed_mask, torch.Tensor)
    with torch.no_grad():
        with timer.section(
            "post_train_evaluation.image_warp", category="fixed_post_training"
        ):
            warped = c.warp_full(moving, dense_coords, mode="bilinear")
        with timer.section(
            "post_train_evaluation.seg_warp", category="fixed_post_training"
        ):
            warped_seg = c.warp_full(moving_seg, dense_coords, mode="nearest")
        with timer.section(
            "post_train_evaluation.dice", category="fixed_post_training", sync=False
        ):
            dice = c.dice_per_label(warped_seg, fixed_seg, moving_seg)
        with timer.section(
            "post_train_evaluation.hd95_asd", category="fixed_post_training", sync=False
        ):
            surface = c.surface_metrics_per_label(warped_seg, fixed_seg, moving_seg)
        with timer.section(
            "post_train_evaluation.dense_ncc", category="fixed_post_training"
        ):
            image_loss = float(
                c.ncc_loss(fixed[None, None], warped[None, None]).detach().cpu()
            )
    metrics: Dict[str, object] = {
        "dense_coords": dense_coords.detach(),
        "warped": warped.detach(),
        "warped_seg": warped_seg.detach(),
        "dice": dice,
        "image_loss": image_loss,
        "dice_mean": c.safe_nanmean(dice),
        "foreground_voxels": int(fixed_mask.sum().detach().cpu()),
    }
    metrics.update(surface)
    return metrics


def method_cps(method: str) -> Optional[int]:
    if method in {"mr_sinr842_composed", "mr_sinr842_requery"}:
        return 2
    if method == "direct_cp_8_4_2_be_bspline_analytic_multistage":
        return 2
    if method in {
        "sinr_cps2_be_bspline_analytic",
        "direct_cp_cps2_be_bspline_analytic",
    }:
        return 2
    if method in {
        "sinr_cps4_be_bspline_analytic",
        "direct_cp_cps4_be_bspline_analytic",
    }:
        return 4
    return None


def default_lr(method: str) -> float:
    if method.startswith("direct_cp_"):
        return 0.001
    return 0.0001


def idir_gradient(
    input_coords: torch.Tensor, output: torch.Tensor, *, create_graph: bool = True
) -> torch.Tensor:
    return torch.autograd.grad(
        output,
        [input_coords],
        grad_outputs=torch.ones_like(output),
        create_graph=create_graph,
        retain_graph=True,
        only_inputs=True,
    )[0]


def idir_jacobian_matrix(
    input_coords: torch.Tensor, output: torch.Tensor
) -> torch.Tensor:
    jac = torch.zeros(
        input_coords.shape[0], 3, 3, dtype=output.dtype, device=output.device
    )
    for i in range(3):
        jac[:, i, :] = idir_gradient(input_coords, output[:, i], create_graph=True)
    return jac


def idir_bending_energy(
    input_coords: torch.Tensor, output_rel: torch.Tensor, batch_size: int
) -> torch.Tensor:
    jac = idir_jacobian_matrix(input_coords, output_rel)
    dx_xyz = torch.zeros(
        input_coords.shape[0], 3, 3, dtype=output_rel.dtype, device=output_rel.device
    )
    dy_xyz = torch.zeros_like(dx_xyz)
    dz_xyz = torch.zeros_like(dx_xyz)
    for i in range(3):
        dx_xyz[:, i, :] = idir_gradient(input_coords, jac[:, i, 0], create_graph=True)
        dy_xyz[:, i, :] = idir_gradient(input_coords, jac[:, i, 1], create_graph=True)
        dz_xyz[:, i, :] = idir_gradient(input_coords, jac[:, i, 2], create_graph=True)
    dx_xyz = dx_xyz.square()
    dy_xyz = dy_xyz.square()
    dz_xyz = dz_xyz.square()
    loss = (
        torch.mean(dx_xyz[:, :, 0])
        + torch.mean(dy_xyz[:, :, 1])
        + torch.mean(dz_xyz[:, :, 2])
    )
    loss = (
        loss
        + 2 * torch.mean(dx_xyz[:, :, 1])
        + 2 * torch.mean(dx_xyz[:, :, 2])
        + torch.mean(dy_xyz[:, :, 2])
    )
    return loss / float(batch_size)


def make_idir_training_coords(
    batch: Dict[str, torch.Tensor | str | int | Dict[str, object]], device: torch.device
) -> torch.Tensor:
    coords_all = c.make_coordinate_tensor(c.IMAGE_SHAPE, device)
    fixed_mask = batch["fixed_mask"]
    assert isinstance(fixed_mask, torch.Tensor)
    selected = coords_all[fixed_mask.detach().reshape(-1).bool()]
    if selected.numel() == 0:
        raise RuntimeError("OASIS fixed image-positive mask is empty.")
    return selected.contiguous()


def evaluate_dense_model(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor | str | int | Dict[str, object]],
    device: torch.device,
    chunk_size: int,
) -> Dict[str, object]:
    coords_all = c.make_coordinate_tensor(c.IMAGE_SHAPE, device)
    chunks: List[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, coords_all.shape[0], int(chunk_size)):
            coords = coords_all[start : start + int(chunk_size)]
            chunks.append(coords + model(coords))
    dense_coords = torch.cat(chunks, dim=0).reshape(*c.IMAGE_SHAPE, 3)
    return c.evaluate_from_dense_coords(dense_coords, batch)


def should_image_sim_eval(args: argparse.Namespace, epoch: int) -> bool:
    interval = 0
    if interval <= 0:
        return False
    return (
        int(epoch) == 1 or int(epoch) % interval == 0 or int(epoch) == int(args.epochs)
    )


def fixed_image_positive_ncc_fields(
    fixed: torch.Tensor, warped: torch.Tensor, fixed_mask: torch.Tensor | None
) -> Dict[str, float | str | int]:
    if fixed_mask is None:
        return {}
    with torch.no_grad():
        mask = fixed_mask.detach().bool()
        voxels = int(mask.sum().detach().cpu())
        if voxels < 2:
            raise RuntimeError(
                "Cannot compute fixed-image-positive NCC on an empty/singleton mask."
            )
        fixed_masked = fixed.detach()[mask]
        warped_masked = warped.detach()[mask]
        loss = c.ncc_loss(fixed_masked, warped_masked)
        value = float(loss.detach().cpu())
    return {
        "fixed_image_positive_ncc_loss": value,
        "fixed_image_positive_ncc_value": -value,
        "fixed_image_positive_one_minus_ncc": 1.0 + value,
        "fixed_image_positive_mask_voxels": voxels,
        "fixed_image_positive_diagnostic_name": "fixed_image_positive_global_ncc",
        "fixed_image_positive_eval_domain": "fixed_raw_image_gt_0",
    }


def evaluate_idir_full_image_ncc(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor | str | int | Dict[str, object]],
    device: torch.device,
    chunk_size: int,
    epoch: int,
) -> Dict[str, float | str | bool | int]:
    fixed = batch["fixed"]
    moving = batch["moving"]
    fixed_mask = batch["fixed_mask"]
    assert isinstance(fixed, torch.Tensor)
    assert isinstance(moving, torch.Tensor)
    assert isinstance(fixed_mask, torch.Tensor)
    eval_start = time.time()
    coords_all = c.make_coordinate_tensor(c.IMAGE_SHAPE, device)
    chunks: List[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, coords_all.shape[0], int(chunk_size)):
            coords = coords_all[start : start + int(chunk_size)]
            chunks.append(coords + model(coords))
        dense_coords = torch.cat(chunks, dim=0).reshape(*c.IMAGE_SHAPE, 3)
        warped = c.warp_full(moving, dense_coords, mode="bilinear")
        loss = c.ncc_loss(fixed[None, None], warped[None, None])
    if device.type == "cuda":
        torch.cuda.synchronize()
    value = float(loss.detach().cpu())
    fixed_positive_fields = fixed_image_positive_ncc_fields(fixed, warped, fixed_mask)
    del coords_all, chunks, dense_coords, warped, loss
    return {
        "epoch": float(epoch),
        "global_epoch": float(epoch),
        "similarity_loss": value,
        "similarity_value": -value,
        "similarity_name": "full_image_global_ncc",
        "eval_domain": "full_image",
        "affects_training": False,
        "eval_elapsed_sec": time.time() - eval_start,
        **fixed_positive_fields,
    }


def train_idir(
    args: argparse.Namespace,
    batch: Dict[str, torch.Tensor | str | int | Dict[str, object]],
    device: torch.device,
    lr: float,
    sampler: Optional[CheckpointSampler] = None,
) -> Dict[str, object]:
    model = c.build_siren(args.hidden_dim, args.hidden_layers, args.idir_omega).to(
        device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    coords_train = make_idir_training_coords(batch, device)
    n_train = int(coords_train.shape[0])
    batch_size = min(int(args.coords_batch_size), n_train)
    moving = batch["moving"]
    fixed_image = batch["fixed"]
    fixed_mask = batch["fixed_mask"]
    assert isinstance(moving, torch.Tensor)
    assert isinstance(fixed_image, torch.Tensor)
    assert isinstance(fixed_mask, torch.Tensor)
    history: List[Dict[str, float]] = []
    image_similarity_eval: List[Dict[str, float | str | bool]] = []
    image_sim_chunk_size = int(args.image_sim_eval_chunk_size or args.eval_chunk_size)
    global_ncc_diagnostic: List[Dict[str, float | str | bool]] = []
    global_ncc_chunk_size = int(
        args.global_ncc_diagnostic_chunk_size or args.eval_chunk_size
    )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    train_start = time.time()
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        indices = torch.randperm(n_train, device=device)[:batch_size]
        coords = coords_train[indices].detach().requires_grad_(True)
        disp = model(coords)
        moved_coords = coords + disp
        warped = c.sample_volume(moving, moved_coords, mode="bilinear")
        fixed_samples = c.sample_volume(fixed_image, coords, mode="bilinear")
        data_loss = c.ncc_loss(fixed_samples, warped)
        reg_loss = idir_bending_energy(coords, disp, batch_size=batch_size)
        loss = data_loss + float(args.alpha) * reg_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        entry = {
            "epoch": float(epoch),
            "loss": float(loss.detach().cpu()),
            "data_loss": float(data_loss.detach().cpu()),
            "reg_loss": float(reg_loss.detach().cpu()),
        }
        history.append(entry)
        if should_image_sim_eval(args, epoch):
            sim_row = evaluate_idir_full_image_ncc(
                model, batch, device, image_sim_chunk_size, epoch
            )
            image_similarity_eval.append(sim_row)
            print(json.dumps({"image_similarity_eval": sim_row}), flush=True)
        if (
            epoch == 1
            or epoch % int(args.eval_interval) == 0
            or epoch == int(args.epochs)
        ):
            print(json.dumps(entry), flush=True)
    torch.cuda.synchronize()
    training_elapsed_sec = time.time() - train_start
    training_peak_mem_gb = torch.cuda.max_memory_allocated() / 1024**3
    metrics = evaluate_dense_model(model, batch, device, args.eval_chunk_size)
    metrics["history"] = history
    metrics["image_similarity_eval"] = image_similarity_eval
    metrics["global_ncc_diagnostic"] = global_ncc_diagnostic
    metrics["model"] = model
    metrics["training_elapsed_sec"] = training_elapsed_sec
    metrics["training_peak_mem_gb"] = training_peak_mem_gb
    metrics["train_coords"] = n_train
    metrics["coords_batch_size"] = batch_size
    metrics["param_count"] = c.count_trainable_params(model)
    return metrics


def build_spline_generator(
    args: argparse.Namespace, control_shape: Tuple[int, int, int]
) -> torch.nn.Module:
    if args.method.startswith("direct_cp_"):
        return c.DirectCPGenerator(control_shape)
    return c.NetworkGenerator(
        c.build_siren(
            args.hidden_dim, args.hidden_layers, args.spline_omega, upstream="SINR"
        )
    )


def save_full_mode_dvf_endpoint(
    args: argparse.Namespace,
    batch: Dict[str, torch.Tensor | str | int | Dict[str, object]],
    dense_disp_norm: torch.Tensor,
    run_dir: Path,
    *,
    timer: Optional[TimingRecorder] = None,
    total_start: Optional[float] = None,
    case_load_complete_sec: Optional[float] = None,
) -> Dict[str, object]:
    """Save the final DVF before full evaluation so the input-to-DVF endpoint is timed."""
    fixed_id = str(batch["fixed_id"])
    moving_id = str(batch["moving_id"])
    path = Path(run_dir) / f"disp_norm_{fixed_id}_{moving_id}.npy"
    if timer is not None:
        timer.mark(
            "dvf_in_memory_ready",
            metadata={
                "definition": "full-mode endpoint immediately after final dense displacement exists, before evaluation",
                "output_mode": "full",
            },
            sync=True,
        )
    prep_token = (
        timer.start(
            "method_execution.full_mode_dvf_file_prep",
            category="dvf_file_output",
            metadata={
                "definition": "detach final dense displacement and move to CPU float32 numpy before evaluation"
            },
        )
        if timer is not None
        else None
    )
    dense_np = dense_disp_norm.detach().cpu().numpy().astype(np.float32, copy=False)
    if timer is not None:
        timer.stop(prep_token)
    write_token = (
        timer.start(
            "method_execution.full_mode_dvf_file_write",
            category="dvf_file_output",
            metadata={
                "path": str(path),
                "definition": "pre-evaluation dense DVF .npy write in full output mode",
            },
        )
        if timer is not None
        else None
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, dense_np)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    full_script_to_dvf_file_sec = (
        float(time.time() - float(total_start))
        if total_start is not None
        else (1e309 - 1e309)
    )
    loaded_images_to_dvf_file_sec = (
        full_script_to_dvf_file_sec - float(case_load_complete_sec)
        if case_load_complete_sec is not None
        and math.isfinite(full_script_to_dvf_file_sec)
        else (1e309 - 1e309)
    )
    if timer is not None:
        timer.stop(write_token)
        timer.mark(
            "dvf_file_ready",
            metadata={
                "path": str(path),
                "definition": "full-mode endpoint immediately after dense DVF .npy write, before evaluation",
                "output_mode": "full",
            },
            sync=False,
        )
    return {
        "pre_evaluation_dvf_file_saved": True,
        "dvf_file_path": str(path),
        "dense_output_units": "normalized_grid_xyz_displacement",
        "dense_output_shape_zyx3": list(dense_np.shape),
        "dense_output_dtype": str(dense_np.dtype),
        "dense_output_file_format": "numpy_npy",
        "full_script_to_dvf_file_sec": full_script_to_dvf_file_sec,
        "loaded_images_to_dvf_file_sec": loaded_images_to_dvf_file_sec,
    }


def train_spline(
    args: argparse.Namespace,
    batch: Dict[str, torch.Tensor | str | int | Dict[str, object]],
    device: torch.device,
    lr: float,
    sampler: Optional[CheckpointSampler] = None,
    timer: Optional[TimingRecorder] = None,
    run_dir: Optional[Path] = None,
    total_start: Optional[float] = None,
    case_load_complete_sec: Optional[float] = None,
) -> Dict[str, object]:
    cps = method_cps(args.method)
    if cps is None:
        raise ValueError(f"No cps for method {args.method}")
    single_stage_early_stop_mode = str(args.stage_early_stop_mode)
    if str(args.stage_min_epochs).strip():
        single_stage_min_epochs = c.parse_stage_ints(str(args.stage_min_epochs), 1)[0]
    else:
        single_stage_min_epochs = int(args.epochs)
    single_stage_patience = c.parse_stage_ints(str(args.stage_patience), 1)[0]
    single_stage_check_interval = c.parse_stage_ints(str(args.stage_check_interval), 1)[
        0
    ]
    if str(args.stage_min_delta).strip():
        single_stage_min_delta = c.parse_float_list(str(args.stage_min_delta), 1, 0.0)[
            0
        ]
    else:
        single_stage_min_delta = 0.0
    dense_grid_token = (
        timer.start(
            "training.initial_dense_grid",
            category="fixed_stage_overhead",
            metadata={"shape": list(c.IMAGE_SHAPE), "cps": int(cps)},
        )
        if timer is not None
        else None
    )
    (control_coords, control_shape) = c.bspline_control_coords(
        c.IMAGE_SHAPE, cps, device
    )
    id_dense = c.make_coordinate_grid(c.IMAGE_SHAPE, device).detach()
    if timer is not None:
        timer.stop(dense_grid_token)
    moving = batch["moving"]
    fixed_image = batch["fixed"]
    fixed_mask = batch["fixed_mask"]
    assert isinstance(moving, torch.Tensor)
    assert isinstance(fixed_image, torch.Tensor)
    assert isinstance(fixed_mask, torch.Tensor)
    train_token = (
        timer.start(
            "training",
            category="training",
            metadata={
                "stage_full_cps": [int(cps)],
                "stage_downsample": [1],
                "stage_requested_epochs": [int(args.epochs)],
            },
        )
        if timer is not None
        else None
    )
    stage_token = (
        timer.start(
            "stage_0",
            category="training_stage",
            metadata={
                "stage_idx": 0,
                "full_cps": int(cps),
                "downsample": 1,
                "requested_epochs": int(args.epochs),
                "lr": float(lr),
                "single_stage_sinr_or_directcp": bool(
                    not args.method.startswith("direct_cp_")
                ),
            },
        )
        if timer is not None
        else None
    )
    setup_token = (
        timer.start(
            "stage_0.stage_setup",
            category="fixed_stage_overhead",
            metadata={"stage_idx": 0, "cps": int(cps)},
        )
        if timer is not None
        else None
    )
    generator = build_spline_generator(args, control_shape).to(device)
    optimizer = torch.optim.Adam(generator.parameters(), lr=lr)
    if timer is not None:
        stage_setup_sec = timer.stop(setup_token)
    else:
        stage_setup_sec = 0.0
    history: List[Dict[str, float]] = []
    global_ncc_diagnostic: List[Dict[str, float | str | bool]] = []
    best_loss = 1e309
    best_data_loss = 1e309 - 1e309
    best_reg_loss = 1e309 - 1e309
    best_epoch = 0
    early_stop_epoch: Optional[int] = None
    early_stop_reason = "requested_epochs_complete"
    pre_loop_token = (
        timer.start(
            "stage_0.pre_loop_cuda_sync_and_memory_reset",
            category="fixed_stage_overhead",
            metadata={
                "stage_idx": 0,
                "definition": "CUDA synchronization and peak-memory-stat reset before epoch timing starts",
            },
        )
        if timer is not None
        else None
    )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    if timer is not None:
        pre_loop_sync_sec = timer.stop(pre_loop_token)
    else:
        pre_loop_sync_sec = 0.0
    train_start = time.time()
    loop_token = (
        timer.start(
            "stage_0.optimizer_loop",
            category="step_proportional",
            metadata={"stage_idx": 0, "requested_epochs": int(args.epochs)},
        )
        if timer is not None
        else None
    )
    phase_acc = CudaPhaseAccumulator(
        bool(timer is not None and timer.enabled and (timer.detail == "deep")), device
    )
    loop_logging_wall_sec = 0.0
    for epoch in range(1, int(args.epochs) + 1):
        generator.train()
        with phase_acc.phase("generator_forward"):
            output = generator(control_coords, control_shape)
        with phase_acc.phase("control_grid"):
            control_disp = output.reshape(*control_shape, 3)
        with phase_acc.phase("expand"):
            dense_disp = c.expand_bspline_controls(
                output, control_shape, c.IMAGE_SHAPE, cps
            )
        with phase_acc.phase("warp_ncc"):
            warped = c.warp_full(moving, id_dense + dense_disp, mode="bilinear")
            data_loss = c.ncc_loss(fixed_image[None, None], warped[None, None])
        with phase_acc.phase("regularizer"):
            reg_loss = c.analytic_bspline_sampled_be(
                control_disp, c.IMAGE_SHAPE, cps, component_reduction="mean"
            )
        loss = data_loss + float(args.alpha) * reg_loss
        with phase_acc.phase("backward_step"):
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        logging_start = time.perf_counter()
        entry = {
            "epoch": float(epoch),
            "global_epoch": float(epoch),
            "stage_idx": 0.0,
            "stage_epoch": float(epoch),
            "full_cps": float(cps),
            "downsample": 1.0,
            "domain_cps": float(cps),
            "loss": float(loss.detach().cpu()),
            "data_loss": float(data_loss.detach().cpu()),
            "reg_loss": float(reg_loss.detach().cpu()),
            "alpha": float(args.alpha),
            "lr": float(lr),
        }
        history.append(entry)
        current_loss = float(entry["loss"])
        if best_epoch == 0 or best_loss - current_loss >= float(single_stage_min_delta):
            best_loss = current_loss
            best_data_loss = float(entry["data_loss"])
            best_reg_loss = float(entry["reg_loss"])
            best_epoch = int(epoch)
        if (
            epoch == 1
            or epoch % int(args.eval_interval) == 0
            or epoch == int(args.epochs)
        ):
            print(json.dumps(entry), flush=True)
        loop_logging_wall_sec += time.perf_counter() - logging_start
        if str(single_stage_early_stop_mode) == "adaptive":
            can_stop = (
                int(epoch) >= int(single_stage_min_epochs)
                and int(epoch) % max(1, int(single_stage_check_interval)) == 0
                and (int(epoch) - int(best_epoch) >= max(1, int(single_stage_patience)))
            )
            if can_stop:
                early_stop_epoch = int(epoch)
                early_stop_reason = f"adaptive_plateau:epoch={int(epoch)};best_epoch={int(best_epoch)};patience={int(single_stage_patience)};min_delta={float(single_stage_min_delta):.6g};best_loss={float(best_loss):.8g};current_loss={float(current_loss):.8g}"
                print(
                    json.dumps(
                        {
                            "single_stage_stop": 0,
                            "epoch": int(epoch),
                            "requested_epochs": int(args.epochs),
                            "stop_reason": early_stop_reason,
                        }
                    ),
                    flush=True,
                )
                break
    if timer is not None:
        optimizer_loop_sec = timer.stop(loop_token)
    else:
        optimizer_loop_sec = 0.0
    actual_epochs = int(history[-1]["epoch"]) if history else 0
    phase_summary = phase_acc.summary()
    phase_summary["logging_early_stop_wall_sec"] = float(loop_logging_wall_sec)
    phase_summary["actual_epochs"] = float(actual_epochs)
    phase_summary["stage_idx"] = 0.0
    phase_summary["full_cps"] = float(cps)
    phase_summary["downsample"] = 1.0
    if timer is not None:
        timer.add_stage_phase_summary(
            {
                "stage_idx": 0,
                "full_cps": int(cps),
                "downsample": 1,
                "requested_epochs": int(args.epochs),
                "actual_epochs": int(actual_epochs),
                "optimizer_loop_sec": float(optimizer_loop_sec),
                "phase_totals_sec": phase_summary,
            }
        )
    final_dense_token = (
        timer.start(
            "stage_0.final_output_dense_expand",
            category="final_output_dense_expand",
            metadata={"stage_idx": 0, "full_cps": int(cps)},
        )
        if timer is not None
        else None
    )
    generator.eval()
    with torch.no_grad():
        final_output = generator(control_coords, control_shape)
        final_control_disp = final_output.reshape(*control_shape, 3).detach()
        final_dense_disp = c.expand_bspline_controls(
            final_output, control_shape, c.IMAGE_SHAPE, cps
        ).detach()
    if timer is not None:
        final_output_dense_expand_sec = timer.stop(final_dense_token)
        if str(getattr(args, "output_mode", "full")) == "dvf_only":
            timer.mark(
                "dvf_in_memory_ready",
                metadata={
                    "stage_idx": 0,
                    "full_cps": int(cps),
                    "definition": "immediately after single-stage final dense displacement expansion",
                },
                sync=False,
            )
    else:
        final_output_dense_expand_sec = 0.0
    stats_token = (
        timer.start(
            "stage_0.control_summary_stats",
            category="fixed_stage_overhead",
            metadata={"stage_idx": 0},
        )
        if timer is not None
        else None
    )
    ctrl_abs = final_control_disp.detach().abs().reshape(-1)
    control_abs_max_norm = float(ctrl_abs.max().detach().cpu())
    control_abs_p95_norm = float(torch.quantile(ctrl_abs, 0.95).detach().cpu())
    if timer is not None:
        control_summary_stats_sec = timer.stop(stats_token)
    else:
        control_summary_stats_sec = 0.0
    torch.cuda.synchronize()
    training_elapsed_sec = time.time() - train_start
    training_peak_mem_gb = torch.cuda.max_memory_allocated() / 1024**3
    stage_timing = {
        "loss_volume_setup_sec": 0.0,
        "stage_setup_sec": float(stage_setup_sec + pre_loop_sync_sec),
        "model_optimizer_setup_sec": float(stage_setup_sec),
        "pre_loop_cuda_sync_and_memory_reset_sec": float(pre_loop_sync_sec),
        "optimizer_loop_sec": float(optimizer_loop_sec),
        "intermediate_dense_expand_sec": 0.0,
        "final_output_dense_expand_sec": float(final_output_dense_expand_sec),
        "final_dense_expand_sec": float(final_output_dense_expand_sec),
        "dense_expand_role": "final_output",
        "control_summary_stats_sec": float(control_summary_stats_sec),
    }
    if timer is not None:
        stage_timing["stage_total_sec"] = timer.stop(stage_token)
        timer.stop(train_token)
    dvf_endpoint_info: Dict[str, object] = {}
    if str(getattr(args, "output_mode", "full")) == "full" and run_dir is not None:
        dvf_endpoint_info = save_full_mode_dvf_endpoint(
            args,
            batch,
            final_dense_disp,
            run_dir,
            timer=timer,
            total_start=total_start,
            case_load_complete_sec=case_load_complete_sec,
        )
    if str(getattr(args, "output_mode", "full")) == "dvf_only":
        metrics: Dict[str, object] = {
            "output_mode": "dvf_only",
            "no_evaluation": True,
            "evaluation_skipped": [
                "image_warp_evaluation",
                "segmentation_warp",
                "dice",
                "hd95_asd",
                "dense_ncc",
                "regularity_metrics",
                "model_control_checkpoint_saves",
                "full_csv_packaging",
            ],
        }
    else:
        eval_start = time.time()
        metrics = timed_evaluate_from_dense_coords(
            id_dense + final_dense_disp, batch, timer=timer
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        metrics["post_dvf_evaluation_sec"] = float(time.time() - eval_start)
    metrics.update(dvf_endpoint_info)
    stage_summary = {
        "stage_idx": 0,
        "full_cps": int(cps),
        "downsample": 1,
        "domain_cps": int(cps),
        "control_shape_zyx": list((int(v) for v in control_shape)),
        "eval_shape_zyx": list((int(v) for v in c.IMAGE_SHAPE)),
        "n_control": int(np.prod(tuple((int(v) for v in control_shape)))),
        "epochs": int(actual_epochs),
        "requested_epochs": int(args.epochs),
        "actual_epochs": int(actual_epochs),
        "lr": float(lr),
        "alpha": float(args.alpha),
        "early_stop_mode": str(single_stage_early_stop_mode),
        "min_epochs": int(single_stage_min_epochs),
        "patience": (
            int(single_stage_patience)
            if str(single_stage_early_stop_mode) == "adaptive"
            else 0
        ),
        "check_interval": int(single_stage_check_interval),
        "rel_delta_total": 0.0,
        "rel_delta_data": 0.0,
        "min_delta": float(single_stage_min_delta),
        "stop_reason": str(early_stop_reason),
        "early_stop_epoch": (
            int(early_stop_epoch) if early_stop_epoch is not None else None
        ),
        "best_loss": float(best_loss),
        "best_loss_epoch": int(best_epoch),
        "best_data_loss": float(best_data_loss),
        "best_reg_loss": float(best_reg_loss),
        "loss_mode": "sinr_ncc_bspline_be",
        "antialias_sigma": 0.0,
        "antialias_kernel_size": 0,
        "elapsed_sec": float(training_elapsed_sec),
        "timing_breakdown": stage_timing,
        "timing_phase_totals_sec": phase_summary,
        "control_abs_max_norm": control_abs_max_norm,
        "control_abs_p95_norm": control_abs_p95_norm,
    }
    metrics["history"] = history
    metrics["stage_history"] = history
    metrics["global_ncc_diagnostic"] = global_ncc_diagnostic
    metrics["stage_summaries"] = [stage_summary]
    metrics["prolongation_diagnostics"] = []
    metrics["generator"] = generator
    metrics["control_shape"] = control_shape
    metrics["control_coords"] = control_coords.detach()
    metrics["control_disp"] = final_control_disp
    metrics["dense_disp"] = final_dense_disp
    metrics["training_elapsed_sec"] = training_elapsed_sec
    metrics["training_peak_mem_gb"] = training_peak_mem_gb
    metrics["param_count"] = c.count_trainable_params(generator)
    metrics["cps"] = cps
    metrics["stage_full_cps"] = [int(cps)]
    metrics["stage_downsample"] = [1]
    metrics["stage_epochs"] = [int(args.epochs)]
    metrics["stage_requested_epochs"] = [int(args.epochs)]
    metrics["stage_actual_epochs"] = [int(actual_epochs)]
    metrics["stage_lr"] = [float(lr)]
    metrics["stage_early_stop_mode"] = str(single_stage_early_stop_mode)
    metrics["stage_min_epochs"] = [int(single_stage_min_epochs)]
    metrics["stage_patience"] = [
        (
            int(single_stage_patience)
            if str(single_stage_early_stop_mode) == "adaptive"
            else 0
        )
    ]
    metrics["stage_check_interval"] = [int(single_stage_check_interval)]
    metrics["stage_rel_delta_total"] = [0.0]
    metrics["stage_rel_delta_data"] = [0.0]
    metrics["stage_min_delta"] = [float(single_stage_min_delta)]
    metrics["stage_stop_reasons"] = [str(early_stop_reason)]
    metrics["early_stop_epoch"] = (
        int(early_stop_epoch) if early_stop_epoch is not None else None
    )
    metrics["early_stop_reason"] = str(early_stop_reason)
    metrics["best_loss"] = float(best_loss)
    metrics["best_loss_epoch"] = int(best_epoch)
    metrics["best_data_loss"] = float(best_data_loss)
    metrics["best_reg_loss"] = float(best_reg_loss)
    metrics["prolongation_preserve_check"] = False
    metrics["save_intermediate_dense"] = False
    metrics["nested_control_shapes_zyx"] = {
        str(int(cps)): list((int(v) for v in control_shape))
    }
    metrics["prolongation"] = "none_single_stage"
    metrics["actual_epochs"] = int(actual_epochs)
    metrics["requested_epochs"] = int(args.epochs)
    return metrics


def prolongate_control_grid(
    prev_control: torch.Tensor,
    prev_shape: Sequence[int],
    target_shape: Sequence[int],
    target_cps: int,
    image_shape: Sequence[int],
    stage_idx: int,
    *,
    prev_dense_full: Optional[torch.Tensor] = None,
    preserve_check: bool = False,
    timer: Optional[TimingRecorder] = None,
) -> Tuple[torch.Tensor, List[Dict[str, object]]]:
    knot_token = (
        timer.start(
            f"stage_{stage_idx}.prolongation.pure_knot_insertion",
            category="pure_knot_insertion",
            metadata={"stage_idx": int(stage_idx), "target_cps": int(target_cps)},
        )
        if timer is not None
        else None
    )
    with torch.no_grad():
        prev_grid = c.flat_to_grid(prev_control.detach(), prev_shape)
        exact_grid = c.cubic_bspline_dyadic_knot_insert_3d(prev_grid, target_shape)
    pure_knot_sec = timer.stop(knot_token) if timer is not None else 0.0
    row: Dict[str, object] = {
        "stage_idx": int(stage_idx),
        "target_cps": int(target_cps),
        "method": "exact_knot",
        "source_shape_zyx": list((int(v) for v in prev_shape)),
        "target_shape_zyx": list((int(v) for v in target_shape)),
        "preserve_check_enabled": bool(preserve_check),
        "timing_pure_knot_insertion_sec": float(pure_knot_sec),
        "timing_prolongation_preserve_dense_expand_sec": 0.0,
        "timing_prolongation_preserve_metrics_sec": 0.0,
    }
    if preserve_check:
        if prev_dense_full is None:
            raise RuntimeError(
                "prolongation preserve check requires previous-stage dense field."
            )
        dense_token = (
            timer.start(
                f"stage_{stage_idx}.prolongation.preserve_dense_expand",
                category="prolongation_preserve_dense_expand",
                metadata={"stage_idx": int(stage_idx), "target_cps": int(target_cps)},
            )
            if timer is not None
            else None
        )
        with torch.no_grad():
            dense = c.expand_on_shape(
                c.grid_to_flat(exact_grid), target_shape, target_cps, image_shape, 1
            )
        row["timing_prolongation_preserve_dense_expand_sec"] = (
            timer.stop(dense_token) if timer is not None else 0.0
        )
        metrics_token = (
            timer.start(
                f"stage_{stage_idx}.prolongation.preserve_metrics",
                category="prolongation_preserve_metrics",
                metadata={"stage_idx": int(stage_idx)},
            )
            if timer is not None
            else None
        )
        row.update(
            c.field_difference_metrics(dense - prev_dense_full, prefix="preserve_")
        )
        row["timing_prolongation_preserve_metrics_sec"] = (
            timer.stop(metrics_token) if timer is not None else 0.0
        )
    rows: List[Dict[str, object]] = [row]
    return (c.grid_to_flat(exact_grid).detach(), rows)


def stage_stop_decision(
    rows: Sequence[Dict[str, object]],
    *,
    min_epochs: int,
    patience: int,
    check_interval: int,
    rel_delta_total: float,
    rel_delta_data: float,
) -> Tuple[bool, str]:
    if not rows:
        return (False, "")
    epoch = int(float(rows[-1]["epoch"]))
    if epoch < int(min_epochs) or epoch % int(check_interval) != 0:
        return (False, "")
    window = max(1, int(check_interval))
    patience = max(window, int(patience))
    if len(rows) < patience + window:
        return (False, "")

    def median_metric(metric: str, start: int, stop: int) -> float:
        values = [float(row[metric]) for row in rows[start:stop]]
        return float(np.median(np.asarray(values, dtype=np.float64)))

    recent_start = len(rows) - window
    prev_stop = len(rows) - patience
    prev_start = max(0, prev_stop - window)
    recent_loss = median_metric("loss", recent_start, len(rows))
    prev_loss = median_metric("loss", prev_start, prev_stop)
    recent_data = median_metric("data_loss", recent_start, len(rows))
    prev_data = median_metric("data_loss", prev_start, prev_stop)
    total_improvement = prev_loss - recent_loss
    data_improvement = prev_data - recent_data
    rel_total = total_improvement / max(abs(prev_loss), 1e-08)
    rel_data = data_improvement / max(abs(prev_data), 1e-08)
    if rel_total < float(rel_delta_total) and rel_data < float(rel_delta_data):
        return (
            True,
            f"adaptive_plateau:epoch={epoch};patience={patience};window={window};rel_total={rel_total:.6g};rel_data={rel_data:.6g};threshold_total={float(rel_delta_total):.6g};threshold_data={float(rel_delta_data):.6g}",
        )
    return (False, "")


def optimize_multistage_stage(
    args: argparse.Namespace,
    fixed_image: torch.Tensor,
    moving: torch.Tensor,
    fixed_mask: torch.Tensor,
    control: torch.nn.Parameter,
    control_shape: Sequence[int],
    full_cps: int,
    downsample: int,
    epochs: int,
    lr: float,
    stage_idx: int,
    global_epoch_offset: int,
    total_epochs: int,
    global_ncc_diagnostic: List[Dict[str, float | str | bool]],
    early_stop_mode: str,
    min_epochs: int,
    patience: int,
    check_interval: int,
    rel_delta_total: float,
    rel_delta_data: float,
    sampler: Optional[CheckpointSampler] = None,
    timer: Optional[TimingRecorder] = None,
    is_final_stage: bool = False,
    require_stage_dense: bool = True,
) -> Tuple[Optional[torch.Tensor], Dict[str, object], List[Dict[str, object]]]:
    stage_timing: Dict[str, float] = {}
    stage_token = (
        timer.start(
            f"stage_{stage_idx}",
            category="training_stage",
            metadata={
                "stage_idx": int(stage_idx),
                "full_cps": int(full_cps),
                "downsample": int(downsample),
                "requested_epochs": int(epochs),
                "lr": float(lr),
            },
        )
        if timer is not None
        else None
    )
    loss_token = (
        timer.start(
            f"stage_{stage_idx}.loss_volume_setup",
            category="fixed_stage_overhead",
            metadata={"stage_idx": int(stage_idx), "downsample": int(downsample)},
        )
        if timer is not None
        else None
    )
    (fixed_l, blur_sigma, blur_kernel) = c.make_loss_volume(
        fixed_image,
        int(downsample),
        float(args.antialias_sigma_scale),
        float(args.antialias_radius_sigma),
    )
    (moving_l, _sigma_m, _kernel_m) = c.make_loss_volume(
        moving,
        int(downsample),
        float(args.antialias_sigma_scale),
        float(args.antialias_radius_sigma),
    )
    if timer is not None:
        stage_timing["loss_volume_setup_sec"] = timer.stop(loss_token)
    setup_token = (
        timer.start(
            f"stage_{stage_idx}.stage_setup",
            category="fixed_stage_overhead",
            metadata={"stage_idx": int(stage_idx)},
        )
        if timer is not None
        else None
    )
    eval_shape = tuple((int(v) for v in fixed_l.shape))
    domain_cps = c.domain_cps_from_full(int(full_cps), int(downsample))
    id_grid = c.make_coordinate_grid(eval_shape, fixed_image.device).detach()
    components = timer.components if timer is not None else None
    with (
        components.scope("optimizer_creation", cuda=False)
        if components is not None
        else _null_section()
    ):
        optimizer = torch.optim.Adam([control], lr=float(lr))
    if timer is not None:
        stage_timing["stage_setup_sec"] = timer.stop(setup_token)
    rows: List[Dict[str, object]] = []
    loss_buffer = None
    if getattr(args, "loss_history_mode", "cpu") == "gpu_buffer":
        from scripts.experiments.mrdbscp842_acceleration.backends import LossBuffer

        loss_buffer = LossBuffer(int(epochs), fixed_image.device)
    snapshot = getattr(args, "_control_snapshot_callback", None)
    if snapshot is not None:
        snapshot(stage_idx, 0, control, control_shape, eval_shape, domain_cps, False)
    stop_reason = "requested_epochs_complete"
    stage_start = time.time()
    loop_token = (
        timer.start(
            f"stage_{stage_idx}.optimizer_loop",
            category="step_proportional",
            metadata={"stage_idx": int(stage_idx), "requested_epochs": int(epochs)},
        )
        if timer is not None
        else None
    )
    phase_acc = CudaPhaseAccumulator(
        bool(timer is not None and timer.enabled and (timer.detail == "deep")),
        fixed_image.device,
    )
    loop_logging_early_stop_wall_sec = 0.0
    print(
        json.dumps(
            {
                "stage_start": int(stage_idx),
                "full_cps": int(full_cps),
                "downsample": int(downsample),
                "domain_cps": int(domain_cps),
                "control_shape_zyx": list((int(v) for v in control_shape)),
                "eval_shape_zyx": list(eval_shape),
                "epochs": int(epochs),
                "lr": float(lr),
                "early_stop_mode": str(early_stop_mode),
                "min_epochs": int(min_epochs),
                "patience": int(patience),
                "check_interval": int(check_interval),
                "rel_delta_total": float(rel_delta_total),
                "rel_delta_data": float(rel_delta_data),
                "image_loss": "sinr_ncc",
                "regularizer": "analytic_cubic_bspline_be",
            }
        ),
        flush=True,
    )
    for epoch in range(1, int(epochs) + 1):
        iteration_token = None
        if components is not None:
            components.epoch = epoch
            iteration_token = components.start("iteration")
            (dense_disp, data_loss, reg_loss, loss) = components.training_step(
                c,
                control,
                control_shape,
                full_cps,
                eval_shape,
                downsample,
                moving_l,
                fixed_l,
                id_grid,
                domain_cps,
                args.alpha,
                optimizer,
            )
        else:
            with phase_acc.phase("control_grid"):
                control_grid = c.flat_to_grid(control, control_shape)
            with phase_acc.phase("expand"):
                dense_disp = c.expand_on_shape(
                    control, control_shape, int(full_cps), eval_shape, int(downsample)
                )
            with phase_acc.phase("warp_ncc"):
                warped = c.warp_full(moving_l, id_grid + dense_disp, mode="bilinear")
                data_loss = c.ncc_loss(fixed_l[None, None], warped[None, None])
            with phase_acc.phase("regularizer"):
                reg_loss = c.analytic_bspline_sampled_be(
                    control_grid,
                    eval_shape,
                    int(domain_cps),
                    component_reduction="mean",
                )
            loss = data_loss + float(args.alpha) * reg_loss
            with phase_acc.phase("backward_step"):
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        logging_start = time.perf_counter()
        host_token = (
            components.start("host_logging_and_checks", cuda=False)
            if components is not None
            else None
        )
        scalar_token = (
            components.start("scalar_transfers", cuda=False)
            if components is not None
            else None
        )
        row = {
            "stage_idx": float(stage_idx),
            "epoch": float(epoch),
            "global_epoch": float(int(global_epoch_offset) + int(epoch)),
            "full_cps": float(full_cps),
            "downsample": float(downsample),
            "domain_cps": float(domain_cps),
            "loss": float(loss.detach().cpu()) if loss_buffer is None else None,
            "data_loss": (
                float(data_loss.detach().cpu()) if loss_buffer is None else None
            ),
            "reg_loss": float(reg_loss.detach().cpu()) if loss_buffer is None else None,
            "alpha": float(args.alpha),
            "lr": float(lr),
        }
        rows.append(row)
        if components is not None:
            components.stop(scalar_token)
        if loss_buffer is not None:
            loss_buffer.append(epoch - 1, loss, data_loss, reg_loss)
            check_due = (
                str(early_stop_mode) == "adaptive"
                and epoch >= int(min_epochs)
                and (epoch % int(check_interval) == 0)
            )
        if snapshot is not None:
            snapshot(
                stage_idx, epoch, control, control_shape, eval_shape, domain_cps, False
            )
        if epoch == 1 or epoch % int(args.eval_interval) == 0 or epoch == int(epochs):
            print(json.dumps(row), flush=True)
        should_break = False
        if str(early_stop_mode) == "adaptive":
            stop_token = (
                components.start("early_stop_check", cuda=False)
                if components is not None
                else None
            )
            (should_stop, reason) = stage_stop_decision(
                rows,
                min_epochs=int(min_epochs),
                patience=int(patience),
                check_interval=int(check_interval),
                rel_delta_total=float(rel_delta_total),
                rel_delta_data=float(rel_delta_data),
            )
            if components is not None:
                components.stop(stop_token)
            if should_stop:
                stop_reason = reason
                print(
                    json.dumps(
                        {
                            "stage_stop": int(stage_idx),
                            "epoch": int(epoch),
                            "requested_epochs": int(epochs),
                            "stop_reason": stop_reason,
                        }
                    ),
                    flush=True,
                )
                should_break = True
        loop_logging_early_stop_wall_sec += time.perf_counter() - logging_start
        if components is not None:
            components.stop(host_token)
            components.stop(iteration_token)
        if should_break:
            break
    if components is not None:
        components.epoch = 0
    if loss_buffer is not None:
        loss_buffer.flush(rows)
    if snapshot is not None:
        snapshot(
            stage_idx, len(rows), control, control_shape, eval_shape, domain_cps, True
        )
    if timer is not None:
        stage_timing["optimizer_loop_sec"] = timer.stop(loop_token)
    phase_summary = phase_acc.summary()
    phase_summary["logging_early_stop_wall_sec"] = float(
        loop_logging_early_stop_wall_sec
    )
    phase_summary["actual_epochs"] = float(int(rows[-1]["epoch"]) if rows else 0)
    phase_summary["stage_idx"] = float(stage_idx)
    phase_summary["full_cps"] = float(full_cps)
    phase_summary["downsample"] = float(downsample)
    if timer is not None:
        timer.add_stage_phase_summary(
            {
                "stage_idx": int(stage_idx),
                "full_cps": int(full_cps),
                "downsample": int(downsample),
                "requested_epochs": int(epochs),
                "actual_epochs": int(rows[-1]["epoch"]) if rows else 0,
                "optimizer_loop_sec": float(
                    stage_timing.get("optimizer_loop_sec", 0.0)
                ),
                "phase_totals_sec": phase_summary,
            }
        )
    dense_full: Optional[torch.Tensor] = None
    stage_timing["intermediate_dense_expand_sec"] = 0.0
    stage_timing["final_output_dense_expand_sec"] = 0.0
    stage_timing["final_dense_expand_sec"] = 0.0
    stage_timing["dense_expand_role"] = "skipped_intermediate"
    if bool(require_stage_dense):
        dense_name = (
            "final_output_dense_expand"
            if is_final_stage
            else "intermediate_dense_expand"
        )
        dense_category = (
            "final_output_dense_expand"
            if is_final_stage
            else "intermediate_dense_diagnostic"
        )
        dense_token = (
            timer.start(
                f"stage_{stage_idx}.{dense_name}",
                category=dense_category,
                metadata={"stage_idx": int(stage_idx), "full_cps": int(full_cps)},
            )
            if timer is not None
            else None
        )
        with torch.no_grad():
            dense_full = c.expand_on_shape(
                control, control_shape, int(full_cps), c.IMAGE_SHAPE, 1
            )
        dense_elapsed = timer.stop(dense_token) if timer is not None else 0.0
        if is_final_stage:
            stage_timing["final_output_dense_expand_sec"] = float(dense_elapsed)
            stage_timing["dense_expand_role"] = "final_output"
        else:
            stage_timing["intermediate_dense_expand_sec"] = float(dense_elapsed)
            stage_timing["dense_expand_role"] = "intermediate_diagnostic_or_save"
        stage_timing["final_dense_expand_sec"] = float(dense_elapsed)
        if (
            is_final_stage
            and str(getattr(args, "output_mode", "full")) == "dvf_only"
            and (timer is not None)
        ):
            timer.mark(
                "dvf_in_memory_ready",
                metadata={
                    "stage_idx": int(stage_idx),
                    "full_cps": int(full_cps),
                    "definition": "immediately after final-stage dense displacement expansion",
                },
                sync=False,
            )
    stats_token = (
        timer.start(
            f"stage_{stage_idx}.control_summary_stats",
            category="fixed_stage_overhead",
            metadata={"stage_idx": int(stage_idx)},
        )
        if timer is not None
        else None
    )
    ctrl_abs = control.detach().abs().reshape(-1)
    control_abs_max_norm = float(ctrl_abs.max().detach().cpu())
    control_abs_p95_norm = float(torch.quantile(ctrl_abs, 0.95).detach().cpu())
    if timer is not None:
        stage_timing["control_summary_stats_sec"] = timer.stop(stats_token)
    actual_epochs = int(rows[-1]["epoch"]) if rows else 0
    summary = {
        "stage_idx": int(stage_idx),
        "full_cps": int(full_cps),
        "downsample": int(downsample),
        "domain_cps": int(domain_cps),
        "control_shape_zyx": list((int(v) for v in control_shape)),
        "eval_shape_zyx": list(eval_shape),
        "n_control": int(np.prod(tuple((int(v) for v in control_shape)))),
        "epochs": int(actual_epochs),
        "requested_epochs": int(epochs),
        "actual_epochs": int(actual_epochs),
        "lr": float(lr),
        "alpha": float(args.alpha),
        "early_stop_mode": str(early_stop_mode),
        "min_epochs": int(min_epochs),
        "patience": int(patience),
        "check_interval": int(check_interval),
        "rel_delta_total": float(rel_delta_total),
        "rel_delta_data": float(rel_delta_data),
        "stop_reason": str(stop_reason),
        "loss_mode": "sinr_ncc_bspline_be",
        "antialias_sigma": float(blur_sigma),
        "antialias_kernel_size": int(blur_kernel),
        "elapsed_sec": float(time.time() - stage_start),
        "timing_breakdown": stage_timing,
        "timing_phase_totals_sec": phase_summary,
        "control_abs_max_norm": control_abs_max_norm,
        "control_abs_p95_norm": control_abs_p95_norm,
    }
    if timer is not None:
        stage_timing["stage_total_sec"] = timer.stop(stage_token)
    return (dense_full.detach() if dense_full is not None else None, summary, rows)


def train_multistage_direct_cp(
    args: argparse.Namespace,
    batch: Dict[str, torch.Tensor | str | int | Dict[str, object]],
    device: torch.device,
    lr: float,
    sampler: Optional[CheckpointSampler] = None,
    timer: Optional[TimingRecorder] = None,
    run_dir: Optional[Path] = None,
    total_start: Optional[float] = None,
    case_load_complete_sec: Optional[float] = None,
) -> Dict[str, object]:
    stage_cps = c.parse_int_list(str(args.stage_full_cps))
    stage_downsample = c.parse_stage_ints(str(args.stage_downsample), len(stage_cps))
    stage_epochs = c.parse_stage_ints(str(args.stage_epochs), len(stage_cps))
    stage_lr = c.parse_float_list(str(args.stage_lr), len(stage_cps), float(lr))
    stage_early_stop_mode = str(args.stage_early_stop_mode)
    if str(args.stage_min_epochs).strip():
        stage_min_epochs = c.parse_stage_ints(
            str(args.stage_min_epochs), len(stage_cps)
        )
    else:
        stage_min_epochs = list(stage_epochs)
    stage_patience = c.parse_stage_ints(str(args.stage_patience), len(stage_cps))
    stage_check_interval = c.parse_stage_ints(
        str(args.stage_check_interval), len(stage_cps)
    )
    stage_rel_delta_total = c.parse_float_list(
        str(args.stage_rel_delta_total), len(stage_cps), 5e-05
    )
    stage_rel_delta_data = c.parse_float_list(
        str(args.stage_rel_delta_data), len(stage_cps), 2e-05
    )
    base_spacing = int(stage_cps[0])
    nested_shapes = c.build_nested_cp_shapes(
        c.IMAGE_SHAPE, base_spacing=base_spacing, spacings=stage_cps
    )
    c.assert_nested_schedule(nested_shapes, stage_cps)
    moving = batch["moving"]
    fixed_image = batch["fixed"]
    fixed_mask = batch["fixed_mask"]
    assert isinstance(moving, torch.Tensor)
    assert isinstance(fixed_image, torch.Tensor)
    assert isinstance(fixed_mask, torch.Tensor)
    dense_grid_token = (
        timer.start(
            "training.initial_dense_grid",
            category="fixed_stage_overhead",
            metadata={"shape": list(c.IMAGE_SHAPE)},
        )
        if timer is not None
        else None
    )
    id_dense = c.make_coordinate_grid(c.IMAGE_SHAPE, device).detach()
    if timer is not None:
        timer.stop(dense_grid_token)
    stage_summaries: List[Dict[str, object]] = []
    history: List[Dict[str, object]] = []
    global_ncc_diagnostic: List[Dict[str, float | str | bool]] = []
    prolong_rows: List[Dict[str, object]] = []
    actual_stage_epochs: List[int] = []
    prev_control: Optional[torch.Tensor] = None
    prev_shape: Optional[Tuple[int, int, int]] = None
    prev_dense_full: Optional[torch.Tensor] = None
    final_dense: Optional[torch.Tensor] = None
    final_control: Optional[torch.Tensor] = None
    final_control_shape: Optional[Tuple[int, int, int]] = None
    final_cps: Optional[int] = None
    train_token = (
        timer.start(
            "training",
            category="training",
            metadata={
                "stage_full_cps": stage_cps,
                "stage_downsample": stage_downsample,
                "stage_requested_epochs": stage_epochs,
            },
        )
        if timer is not None
        else None
    )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    train_start = time.time()
    global_epoch_offset = 0
    requested_total_epochs = int(sum(stage_epochs))
    for stage_idx, (full_cps, downsample, epochs, stage_lr_value) in enumerate(
        zip(stage_cps, stage_downsample, stage_epochs, stage_lr)
    ):
        control_shape = tuple((int(v) for v in nested_shapes[int(full_cps)]))
        components = timer.components if timer is not None else None
        if components is not None:
            (components.stage, components.epoch) = (stage_idx, 0)
        with (
            components.scope("control_initialization")
            if components is not None
            else _null_section()
        ):
            control = torch.nn.Parameter(
                torch.zeros(
                    int(np.prod(control_shape)), 3, device=device, dtype=torch.float32
                )
            )
        if prev_control is not None and prev_shape is not None:
            prolong_token = (
                timer.start(
                    f"stage_{stage_idx}.prolongation",
                    category="fixed_stage_overhead",
                    metadata={"stage_idx": int(stage_idx), "target_cps": int(full_cps)},
                )
                if timer is not None
                else None
            )
            (init_flat, rows) = prolongate_control_grid(
                prev_control,
                prev_shape,
                control_shape,
                int(full_cps),
                c.IMAGE_SHAPE,
                int(stage_idx),
                prev_dense_full=prev_dense_full,
                preserve_check=False,
                timer=timer,
            )
            with torch.no_grad():
                control.copy_(init_flat)
            prolong_elapsed = timer.stop(prolong_token) if timer is not None else 0.0
            for row in rows:
                row["timing_prolongation_sec"] = float(prolong_elapsed)
            prolong_rows.extend(rows)
            print(json.dumps({"prolongation": rows}, allow_nan=True), flush=True)
        is_final_stage = stage_idx == len(stage_cps) - 1
        require_stage_dense = bool(is_final_stage or False or False)
        (dense_full, summary, rows) = optimize_multistage_stage(
            args,
            fixed_image,
            moving,
            fixed_mask,
            control,
            control_shape,
            int(full_cps),
            int(downsample),
            int(epochs),
            float(stage_lr_value),
            int(stage_idx),
            int(global_epoch_offset),
            int(requested_total_epochs),
            global_ncc_diagnostic,
            stage_early_stop_mode,
            int(stage_min_epochs[stage_idx]),
            int(stage_patience[stage_idx]),
            int(stage_check_interval[stage_idx]),
            float(stage_rel_delta_total[stage_idx]),
            float(stage_rel_delta_data[stage_idx]),
            sampler=sampler,
            timer=timer,
            is_final_stage=is_final_stage,
            require_stage_dense=require_stage_dense,
        )
        if False and (not is_final_stage):
            if dense_full is None:
                raise RuntimeError(
                    "Intermediate dense save requested but no stage dense field was computed."
                )
            if run_dir is None:
                raise RuntimeError(
                    "Intermediate dense save requested without a run directory."
                )
            save_token = (
                timer.start(
                    f"stage_{stage_idx}.intermediate_dense_save",
                    category="intermediate_dense_diagnostic",
                    metadata={"stage_idx": int(stage_idx), "full_cps": int(full_cps)},
                )
                if timer is not None
                else None
            )
            fixed_id = str(batch["fixed_id"])
            moving_id = str(batch["moving_id"])
            dense_path = (
                Path(run_dir)
                / "intermediate_dense"
                / f"disp_norm_stage{stage_idx + 1}_cps{int(full_cps)}_{fixed_id}_{moving_id}.npy"
            )
            dense_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(
                dense_path,
                dense_full.detach().cpu().numpy().astype(np.float32, copy=False),
            )
            save_elapsed = timer.stop(save_token) if timer is not None else 0.0
            summary["intermediate_dense_path"] = str(dense_path)
            timing_breakdown = summary.get("timing_breakdown")
            if isinstance(timing_breakdown, dict):
                timing_breakdown["intermediate_dense_save_sec"] = float(save_elapsed)
        actual_epochs = int(summary.get("actual_epochs", epochs))
        actual_stage_epochs.append(actual_epochs)
        global_epoch_offset += actual_epochs
        prev_control = control.detach()
        prev_shape = control_shape
        prev_dense_full = None
        if is_final_stage:
            final_dense = dense_full
        final_control = control.detach()
        final_control_shape = control_shape
        final_cps = int(full_cps)
        stage_summaries.append(summary)
        history.extend(rows)
        if components is not None:
            with components.scope("instrumentation_stage_resolve", cuda=False):
                components.resolve()
    torch.cuda.synchronize()
    training_elapsed_sec = time.time() - train_start
    training_peak_mem_gb = torch.cuda.max_memory_allocated() / 1024**3
    if timer is not None:
        timer.stop(train_token)
    if (
        final_dense is None
        or final_control is None
        or final_control_shape is None
        or (final_cps is None)
    ):
        raise RuntimeError("Multistage training produced no final control field.")
    is_dvf_only = str(getattr(args, "output_mode", "full")) == "dvf_only"
    dvf_endpoint_info: Dict[str, object] = {}
    if not is_dvf_only and run_dir is not None:
        dvf_endpoint_info = save_full_mode_dvf_endpoint(
            args,
            batch,
            final_dense.detach(),
            run_dir,
            timer=timer,
            total_start=total_start,
            case_load_complete_sec=case_load_complete_sec,
        )
    if is_dvf_only:
        metrics: Dict[str, object] = {
            "output_mode": "dvf_only",
            "evaluation_skipped": [
                "image_warp_evaluation",
                "segmentation_warp",
                "dice",
                "hd95_asd",
                "dense_ncc",
                "regularity_metrics",
                "model_control_checkpoint_saves",
                "full_csv_packaging",
            ],
        }
    else:
        eval_token = (
            timer.start(
                "post_train_evaluation",
                category="fixed_post_training",
                metadata={"stage_count": len(stage_summaries)},
            )
            if timer is not None
            else None
        )
        eval_start = time.time()
        metrics = timed_evaluate_from_dense_coords(
            id_dense + final_dense.detach(), batch, timer=timer
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        metrics["post_dvf_evaluation_sec"] = float(time.time() - eval_start)
        if timer is not None:
            timer.stop(eval_token)
        final_control_grid = c.flat_to_grid(final_control, final_control_shape).detach()
        metrics["control_coords"] = torch.empty(0, 3, dtype=torch.float32)
        metrics["control_disp"] = final_control_grid
    metrics.update(dvf_endpoint_info)
    metrics["history"] = [
        {**row, "epoch": float(row["global_epoch"]), "stage_epoch": float(row["epoch"])}
        for row in history
    ]
    metrics["stage_history"] = history
    metrics["global_ncc_diagnostic"] = global_ncc_diagnostic
    metrics["stage_summaries"] = stage_summaries
    metrics["prolongation_diagnostics"] = prolong_rows
    metrics["control_shape"] = final_control_shape
    metrics["dense_disp"] = final_dense.detach()
    metrics["training_elapsed_sec"] = training_elapsed_sec
    metrics["training_peak_mem_gb"] = training_peak_mem_gb
    metrics["param_count"] = int(final_control.numel())
    metrics["cps"] = int(final_cps)
    metrics["stage_full_cps"] = stage_cps
    metrics["stage_downsample"] = stage_downsample
    metrics["stage_epochs"] = stage_epochs
    metrics["stage_requested_epochs"] = stage_epochs
    metrics["stage_actual_epochs"] = actual_stage_epochs
    metrics["stage_lr"] = stage_lr
    metrics["stage_early_stop_mode"] = stage_early_stop_mode
    metrics["stage_min_epochs"] = stage_min_epochs
    metrics["stage_patience"] = stage_patience
    metrics["stage_check_interval"] = stage_check_interval
    metrics["stage_rel_delta_total"] = stage_rel_delta_total
    metrics["stage_rel_delta_data"] = stage_rel_delta_data
    metrics["stage_stop_reasons"] = [
        str(row.get("stop_reason")) for row in stage_summaries
    ]
    metrics["requested_epochs"] = requested_total_epochs
    metrics["actual_epochs"] = int(sum(actual_stage_epochs))
    metrics["prolongation_preserve_check"] = False
    metrics["save_intermediate_dense"] = False
    metrics["nested_control_shapes_zyx"] = {
        str(k): list(v) for (k, v) in nested_shapes.items()
    }
    metrics["prolongation"] = "exact crop-aligned cubic B-spline dyadic knot insertion"
    return metrics
